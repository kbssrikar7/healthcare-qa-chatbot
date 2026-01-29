#!/usr/bin/env python3
"""
Build the medical knowledge base from downloaded datasets.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from src.data_pipeline.loaders.dataset_loader import MedicalDatasetLoader
from src.data_pipeline.preprocessors.text_cleaner import MedicalTextCleaner
from src.data_pipeline.preprocessors.chunker import MedicalTextChunker
from src.embeddings.embedding_models import MedicalEmbedder
from src.embeddings.vector_store import VectorStore

def main():
    print("🏗️ Building Medical Knowledge Base\n")
    print("=" * 50)
    
    # Initialize components
    print("1️⃣ Initializing components...")
    loader = MedicalDatasetLoader()
    cleaner = MedicalTextCleaner()
    chunker = MedicalTextChunker(chunk_size=512, chunk_overlap=50)
    embedder = MedicalEmbedder(model_name="all-minilm")
    vector_store = VectorStore(
        collection_name="medical_knowledge",
        persist_directory="data/knowledge_base"
    )
    
    # Load and process documents
    print("\n2️⃣ Loading documents...")
    documents = list(loader.get_documents_for_knowledge_base())
    print(f"   Loaded {len(documents)} documents")
    
    # Clean and chunk
    print("\n3️⃣ Cleaning and chunking...")
    all_chunks = []
    for doc in tqdm(documents, desc="Processing"):
        cleaned_content = cleaner.clean(doc["content"])
        chunks = chunker.chunk_document({
            "content": cleaned_content,
            "source": doc["source"],
            "metadata": doc.get("metadata", {})
        })
        all_chunks.extend(chunks)
    
    print(f"   Created {len(all_chunks)} chunks")
    
    # Process in batches to allow incremental availability
    print("\n4️⃣ & 5️⃣ Generating embeddings and adding to store in batches...")
    
    batch_size = 1000
    total_chunks = len(all_chunks)
    
    for i in tqdm(range(0, total_chunks, batch_size), desc="Batch Processing"):
        batch = all_chunks[i : i + batch_size]
        texts = [chunk.content for chunk in batch]
        
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
    
    # Summary
    print("\n" + "=" * 50)
    print("✅ Knowledge Base Built Successfully!")
    print(f"   📊 Total chunks: {len(all_chunks)}")
    print(f"   📁 Location: data/knowledge_base")
    print(f"   📈 Vector store stats: {vector_store.get_stats()}")

if __name__ == "__main__":
    main()
