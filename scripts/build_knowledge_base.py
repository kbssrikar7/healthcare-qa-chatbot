#!/usr/bin/env python3
"""
Build the medical knowledge base from downloaded datasets.
Optimized for handling large datasets with progress tracking.

Usage examples
--------------
# Default (legacy MedicalTextChunker, KB v1):
    python scripts/build_knowledge_base.py

# KB v2 with RecursiveSentenceChunker:
    python scripts/build_knowledge_base.py \
        --chunker recursive \
        --output-dir data/knowledge_base_v2 \
        --collection medical_knowledge_v2
"""
import argparse
import sys
import gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from src.data_pipeline.loaders.dataset_loader import MedicalDatasetLoader
from src.data_pipeline.preprocessors.text_cleaner import MedicalTextCleaner
from src.data_pipeline.preprocessors.chunker import MedicalTextChunker, RecursiveSentenceChunker
from src.embeddings.embedding_models import MedicalEmbedder
from src.embeddings.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Build medical knowledge base")
    parser.add_argument("--chunker", choices=["fixed", "recursive"], default="fixed",
                        help="Chunker type: 'fixed' = original MedicalTextChunker, "
                             "'recursive' = new RecursiveSentenceChunker for KB v2")
    parser.add_argument("--output-dir", default="data/knowledge_base",
                        help="ChromaDB persist directory")
    parser.add_argument("--collection", default="medical_knowledge",
                        help="ChromaDB collection name")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  BUILDING MEDICAL KNOWLEDGE BASE")
    print(f"  chunker={args.chunker}  output={args.output_dir}  collection={args.collection}")
    print("=" * 60)

    # Initialize components
    print("\n[1/5] Initializing components...")
    loader = MedicalDatasetLoader()
    cleaner = MedicalTextCleaner()

    if args.chunker == "recursive":
        chunker = RecursiveSentenceChunker(chunk_overlap=100, min_chunk_size=80)
        print("  Using RecursiveSentenceChunker (domain-adaptive, sentence-boundary-aware)")
    else:
        chunker = MedicalTextChunker(chunk_size=512, chunk_overlap=50)
        print("  Using MedicalTextChunker (fixed 512-token chunks)")

    embedder = MedicalEmbedder(model_name="all-minilm")
    vector_store = VectorStore(
        collection_name=args.collection,
        persist_directory=args.output_dir,
    )
    
    # Show dataset statistics
    print("\n[2/5] Checking available datasets...")
    try:
        stats = loader.get_stats()
        print("  Available datasets:")
        for name, count in stats.items():
            if name != "total" and count > 0:
                print(f"    - {name}: {count:,} entries")
        print(f"  Total raw entries: {stats.get('total', 0):,}")
    except Exception as e:
        print(f"  Could not get stats: {e}")
    
    # Load documents with streaming to handle large datasets
    print("\n[3/5] Loading and processing documents...")
    all_chunks = []
    doc_count = 0
    
    # Process in streaming fashion to reduce memory usage
    for doc in tqdm(loader.get_documents_for_knowledge_base(), desc="Processing documents"):
        try:
            cleaned_content = cleaner.clean(doc["content"])
            if len(cleaned_content.strip()) < 50:  # Skip very short content
                continue
                
            chunks = chunker.chunk_document({
                "content": cleaned_content,
                "source": doc["source"],
                "metadata": doc.get("metadata", {})
            })
            all_chunks.extend(chunks)
            doc_count += 1
            
            # Periodic garbage collection for large datasets
            if doc_count % 50000 == 0:
                gc.collect()
                print(f"    Processed {doc_count:,} documents, {len(all_chunks):,} chunks so far...")
                
        except Exception as e:
            # Skip problematic documents
            continue
    
    print(f"  Processed {doc_count:,} documents")
    print(f"  Created {len(all_chunks):,} text chunks")
    
    # Generate embeddings and add to vector store
    print("\n[4/5] Generating embeddings and indexing...")
    
    batch_size = 500  # Smaller batches for better memory management
    total_chunks = len(all_chunks)
    
    for i in tqdm(range(0, total_chunks, batch_size), desc="Indexing batches"):
        batch = all_chunks[i : i + batch_size]
        texts = [chunk.content for chunk in batch]
        
        try:
            # Generate embeddings for batch
            embeddings = embedder.embed_documents(texts, batch_size=32)
            
            # Prepare metadata
            metadatas = [
                {
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                    "total_chunks": chunk.total_chunks,
                    **chunk.metadata
                }
                for chunk in batch
            ]
            
            # Add batch to vector store
            vector_store.add_documents(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas
            )
        except Exception as e:
            print(f"\n  Warning: Failed to process batch at {i}: {e}")
            continue
        
        # Periodic garbage collection
        if (i // batch_size) % 100 == 0:
            gc.collect()
    
    # Verify and summarize
    print("\n[5/5] Finalizing...")
    
    try:
        final_stats = vector_store.get_stats()
    except Exception as e:
        print(f"[warn] Could not retrieve final stats: {e}")
        final_stats = {"count": "unknown"}
    
    print("\n" + "=" * 60)
    print("  KNOWLEDGE BASE BUILD COMPLETE")
    print("=" * 60)
    print(f"  Documents processed: {doc_count:,}")
    print(f"  Chunks created: {len(all_chunks):,}")
    print(f"  Vector store stats: {final_stats}")
    print(f"  Location: {args.output_dir}")
    print("\nThe knowledge base is ready for use!")

    # Record embedding model metadata as sidecar JSON
    try:
        vector_store.record_embedding_model(
            model_name=embedder._model_path,
            dimension=embedder.embedding_dimension,
        )
        print(f"  Embedding metadata written to {args.output_dir}/embedding_metadata.json")
    except Exception as e:
        print(f"  [warn] Could not write embedding metadata: {e}")


if __name__ == "__main__":
    main()
