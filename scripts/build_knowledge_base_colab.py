#!/usr/bin/env python3
"""
Colab-compatible script to build the medical knowledge base.
Run this in Google Colab for stable environment.

Usage:
1. Upload your final_project folder to Colab or mount Google Drive
2. Run: !pip install chromadb sentence-transformers pandas pyarrow tqdm
3. Run this script
"""
import sys
import gc
from pathlib import Path

# Add project to path
import os
# Allow overriding PROJECT_ROOT via environment variable for portability
project_root_env = os.environ.get("PROJECT_ROOT")
PROJECT_ROOT = Path(project_root_env) if project_root_env else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Generator
import hashlib

def generate_deterministic_id(source: str, content: str, chunk_id: int = 0) -> str:
    """Generate a stable document ID from source + chunk_id + content hash."""
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    key = f"{source}::chunk{chunk_id}::{content_hash}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


class SimpleEmbedder:
    """Simple sentence-transformers embedder."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Loaded embedding model. Dimension: {self.dimension}")
    
    def embed_documents(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return np.array(embeddings)


class SimpleVectorStore:
    """Simplified ChromaDB vector store."""
    
    def __init__(self, collection_name: str, persist_directory: str):
        import chromadb
        from chromadb.config import Settings
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Vector store initialized. Documents: {self.collection.count()}")
    
    def add_documents(self, documents: List[str], embeddings: List[List[float]], 
                      metadatas: List[Dict], ids: List[str]):
        # Clean metadata
        clean_metadatas = []
        for meta in metadatas:
            clean_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                elif v is None:
                    clean_meta[k] = ""
                else:
                    clean_meta[k] = str(v)
            clean_metadatas.append(clean_meta)
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=clean_metadatas
        )
    
    def count(self):
        return self.collection.count()


class TextChunk:
    def __init__(self, content, source, chunk_id, total_chunks, metadata):
        self.content = content
        self.source = source
        self.chunk_id = chunk_id
        self.total_chunks = total_chunks
        self.metadata = metadata


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Simple text chunking."""
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
        if end >= len(words):
            break
    
    return chunks if chunks else [text]


def load_all_qa_pairs(data_dir: Path) -> Generator[Dict, None, None]:
    """Load all QA pairs from parquet files."""
    
    # MedQuAD
    path = data_dir / "mediqa" / "medquad.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            yield {
                "question": row.get("Question", row.get("question", "")),
                "answer": row.get("Answer", row.get("answer", "")),
                "source": "MedQuAD"
            }
        print(f"  Loaded MedQuAD: {len(df):,}")
    
    # PubMedQA
    path = data_dir / "pubmed" / "pubmedqa_labeled.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            yield {
                "question": row.get("question", ""),
                "answer": row.get("long_answer", ""),
                "source": "PubMedQA"
            }
        print(f"  Loaded PubMedQA: {len(df):,}")
    
    # MedMCQA
    path = data_dir / "mediqa" / "medmcqa_train.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        count = 0
        for _, row in df.iterrows():
            answer = row.get("exp")
            if answer and not pd.isna(answer):
                yield {
                    "question": row.get("question", ""),
                    "answer": str(answer),
                    "source": f"MedMCQA"
                }
                count += 1
        print(f"  Loaded MedMCQA: {count:,}")
    
    # HealthCareMagic
    path = data_dir / "mediqa" / "healthcare_magic.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            question = row.get("input", row.get("instruction", ""))
            yield {
                "question": question,
                "answer": row.get("output", ""),
                "source": "HealthCareMagic"
            }
        print(f"  Loaded HealthCareMagic: {len(df):,}")
    
    # MedQA USMLE
    for filename in ["medqa_usmle_train.parquet", "medqa_usmle_test.parquet"]:
        path = data_dir / "medqa" / filename
        if path.exists():
            df = pd.read_parquet(path)
            for _, row in df.iterrows():
                question = row.get("question", row.get("sent1", ""))
                answer = row.get("answer", "")
                
                options = row.get("options", [])
                answer_idx = row.get("answer_idx", row.get("label", -1))
                if options and isinstance(answer_idx, int) and 0 <= answer_idx < len(options):
                    answer = options[answer_idx]
                
                if question and answer:
                    yield {
                        "question": question,
                        "answer": str(answer),
                        "source": "MedQA-USMLE"
                    }
            print(f"  Loaded {filename}: {len(df):,}")
    
    # ChatDoctor
    for filename in ["chatdoctor_icliniq.parquet", "chatdoctor_healthcaremagic.parquet"]:
        path = data_dir / "chatdoctor" / filename
        if path.exists():
            df = pd.read_parquet(path)
            for _, row in df.iterrows():
                question = row.get("input", row.get("instruction", row.get("question", "")))
                answer = row.get("output", row.get("answer", ""))
                if question and answer:
                    yield {
                        "question": question,
                        "answer": answer,
                        "source": "ChatDoctor"
                    }
            print(f"  Loaded {filename}: {len(df):,}")
    
    # Medical Meadow
    meadow_dir = data_dir / "medical_meadow"
    if meadow_dir.exists():
        for parquet_file in meadow_dir.glob("*.parquet"):
            df = pd.read_parquet(parquet_file)
            for _, row in df.iterrows():
                instruction = row.get("instruction", "")
                input_text = row.get("input", "")
                output_text = row.get("output", "")
                
                question = instruction
                if input_text:
                    question = f"{instruction}\n\n{input_text}" if instruction else input_text
                
                if question and output_text:
                    yield {
                        "question": question,
                        "answer": output_text,
                        "source": f"MedicalMeadow"
                    }
            print(f"  Loaded {parquet_file.name}: {len(df):,}")


def main():
    print("\n" + "=" * 60)
    print("  BUILDING MEDICAL KNOWLEDGE BASE (Colab Version)")
    print("=" * 60)
    
    DATA_DIR = PROJECT_ROOT / "data" / "raw"
    KB_DIR = PROJECT_ROOT / "data" / "knowledge_base_new"
    
    # Initialize components
    print("\n[1/4] Initializing components...")
    embedder = SimpleEmbedder("all-MiniLM-L6-v2")
    vector_store = SimpleVectorStore(
        collection_name="medical_knowledge",
        persist_directory=str(KB_DIR)
    )
    
    # Process documents
    print("\n[2/4] Loading and processing documents...")
    all_chunks = []
    doc_count = 0
    
    for qa in tqdm(load_all_qa_pairs(DATA_DIR), desc="Processing"):
        content = f"Question: {qa['question']}\n\nAnswer: {qa['answer']}"
        
        # Skip very short content
        if len(content.strip()) < 50:
            continue
        
        # Chunk the content
        chunks = chunk_text(content, chunk_size=512, overlap=50)
        
        for i, chunk in enumerate(chunks):
            all_chunks.append(TextChunk(
                content=chunk,
                source=qa['source'],
                chunk_id=i + 1,
                total_chunks=len(chunks),
                metadata={"type": "qa_pair"}
            ))
        
        doc_count += 1
        
        # Periodic garbage collection
        if doc_count % 50000 == 0:
            gc.collect()
            print(f"    Processed {doc_count:,} documents, {len(all_chunks):,} chunks...")
    
    print(f"\n  Total documents: {doc_count:,}")
    print(f"  Total chunks: {len(all_chunks):,}")
    
    # Generate embeddings and index
    print("\n[3/4] Generating embeddings and indexing...")
    
    batch_size = 500
    total_chunks = len(all_chunks)
    
    for i in tqdm(range(0, total_chunks, batch_size), desc="Indexing"):
        batch = all_chunks[i : i + batch_size]
        texts = [chunk.content for chunk in batch]
        
        try:
            embeddings = embedder.embed_documents(texts, batch_size=32)
            
            metadatas = [
                {
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                    "total_chunks": chunk.total_chunks,
                    **chunk.metadata
                }
                for chunk in batch
            ]
            
            ids = [
                generate_deterministic_id(
                    source=chunk.source,
                    content=chunk.content,
                    chunk_id=chunk.chunk_id
                )
                for chunk in batch
            ]
            
            vector_store.add_documents(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            print(f"\n  Error at batch {i}: {e}")
            continue
        
        if (i // batch_size) % 100 == 0:
            gc.collect()
    
    # Done
    print("\n[4/4] Finalizing...")
    final_count = vector_store.count()
    
    print("\n" + "=" * 60)
    print("  BUILD COMPLETE!")
    print("=" * 60)
    print(f"  Documents processed: {doc_count:,}")
    print(f"  Chunks indexed: {final_count:,}")
    print(f"  Location: {KB_DIR}")
    print("\nDownload the knowledge_base_new folder and replace your local one!")


if __name__ == "__main__":
    main()
