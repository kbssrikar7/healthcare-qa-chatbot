import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings.embedding_models import MedicalEmbedder
from src.embeddings.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever

def debug_retrieval():
    print("🔄 Initializing retrieval components...")
    
    # Initialize components
    embedder = MedicalEmbedder(model_name="all-minilm")
    vector_store = VectorStore(
        collection_name="medical_knowledge",
        persist_directory="data/knowledge_base"
    )
    retriever = HybridRetriever(embedder, vector_store)
    
    query = "Symptoms of Type 2 Diabetes"
    print(f"\n❓ Querying: '{query}'")
    
    # 1. Test Dense Retrieval
    print("\n🔎 Testing Dense Retrieval (Vector Only)...")
    dense_results = vector_store.search(query, k=3)
    for i, res in enumerate(dense_results):
        print(f"  {i+1}. Score: {res.score:.4f} | Content: {res.content[:100]}...")
        
    # 2. Test Hybrid Retrieval
    print("\n🔎 Testing Hybrid Retrieval (RRF)...")
    hybrid_results = retriever.retrieve(query, k=3)
    for i, res in enumerate(hybrid_results):
        print(f"  {i+1}. Score: {res.score:.4f} | Source: {res.source} | Content: {res.content[:100]}...")

if __name__ == "__main__":
    debug_retrieval()
