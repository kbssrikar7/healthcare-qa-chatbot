#!/usr/bin/env python3
"""
Ingest curated drug data into the ChromaDB knowledge base.

This script reads drug information from JSON and adds it to the existing
medical_knowledge collection so the retriever can find accurate drug-specific
information when users ask about medications.

Usage:
    python scripts/ingest_drug_data.py
"""
import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.embedding_models import MedicalEmbedder
from src.embeddings.vector_store import VectorStore


def load_drug_data(json_path: str) -> list:
    """Load drug data from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def drug_to_documents(drug: dict) -> list:
    """
    Convert a single drug entry into multiple Q&A-style documents
    for better retrieval. Each drug generates several documents
    covering different aspects (overview, side effects, dosage, etc.)
    """
    name = drug["drug_name"]
    generic = drug["generic_name"]
    drug_class = drug["drug_class"]
    
    documents = []
    
    # 1. General overview document
    documents.append({
        "content": (
            f"Question: What is {name}? What type of drug is {name}?\n\n"
            f"Answer: {name} (generic name: {generic}) is a {drug_class}. "
            f"Composition: {drug['composition']}. "
            f"It is manufactured by {drug['manufacturer']} and available as: {drug['available_forms']}. "
            f"Uses: {drug['uses']}"
        ),
        "source": "Drug Reference Database",
        "metadata": {
            "type": "drug_info",
            "drug_name": name,
            "generic_name": generic,
            "drug_class": drug_class,
            "aspect": "overview"
        }
    })
    
    # 2. Side effects document
    documents.append({
        "content": (
            f"Question: What are the side effects of {name} ({generic})? "
            f"Can {name} cause rash or allergic reactions?\n\n"
            f"Answer: {name} ({generic}) belongs to the {drug_class} class. "
            f"Side effects of {name}: {drug['side_effects']} "
            f"Warnings: {drug['warnings']}"
        ),
        "source": "Drug Reference Database",
        "metadata": {
            "type": "drug_info",
            "drug_name": name,
            "generic_name": generic,
            "drug_class": drug_class,
            "aspect": "side_effects"
        }
    })
    
    # 3. Dosage document
    documents.append({
        "content": (
            f"Question: What is the dosage of {name}? How to take {name}?\n\n"
            f"Answer: {name} ({generic}) dosage: {drug['dosage']} "
            f"Warnings: {drug['warnings']}"
        ),
        "source": "Drug Reference Database",
        "metadata": {
            "type": "drug_info",
            "drug_name": name,
            "generic_name": generic,
            "drug_class": drug_class,
            "aspect": "dosage"
        }
    })
    
    # 4. Drug interactions document
    documents.append({
        "content": (
            f"Question: What are the drug interactions of {name}? "
            f"What should not be taken with {name}?\n\n"
            f"Answer: {name} ({generic}, {drug_class}) interactions: {drug['interactions']} "
            f"Warnings: {drug['warnings']}"
        ),
        "source": "Drug Reference Database",
        "metadata": {
            "type": "drug_info",
            "drug_name": name,
            "generic_name": generic,
            "drug_class": drug_class,
            "aspect": "interactions"
        }
    })
    
    return documents


def main():
    drug_data_path = Path("data/drug_knowledge/common_drugs.json")
    
    if not drug_data_path.exists():
        print(f"❌ Drug data not found at {drug_data_path}")
        sys.exit(1)
    
    # Load drug data
    print("📋 Loading drug data...")
    drugs = load_drug_data(str(drug_data_path))
    print(f"   Found {len(drugs)} drugs")
    
    # Convert to documents
    print("📝 Converting to documents...")
    all_documents = []
    for drug in drugs:
        docs = drug_to_documents(drug)
        all_documents.extend(docs)
    print(f"   Generated {len(all_documents)} documents")
    
    # Initialize embedder
    print("🔄 Loading embedding model...")
    embedder = MedicalEmbedder(model_name="all-minilm")
    
    # Initialize vector store (connects to existing collection)
    print("🔄 Connecting to vector store...")
    vector_store = VectorStore(
        collection_name="medical_knowledge",
        persist_directory="data/knowledge_base"
    )
    
    # Get current stats
    stats_before = vector_store.get_stats()
    print(f"   Current documents in collection: {stats_before['count']}")
    
    # Generate embeddings
    print("🔄 Generating embeddings for drug documents...")
    contents = [doc["content"] for doc in all_documents]
    embeddings = embedder.embed_documents(contents, batch_size=32)
    
    # Prepare metadata
    metadatas = [doc["metadata"] for doc in all_documents]
    for i, doc in enumerate(all_documents):
        metadatas[i]["source"] = doc["source"]
    
    # Add to vector store
    print("💾 Adding to knowledge base...")
    ids = vector_store.add_documents(
        documents=contents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )
    
    # Verify
    stats_after = vector_store.get_stats()
    print(f"\n✅ Ingestion complete!")
    print(f"   Documents before: {stats_before['count']}")
    print(f"   Documents added:  {len(ids)}")
    print(f"   Documents after:  {stats_after['count']}")
    
    # Quick verification - search for a drug
    print("\n🔍 Verification search: 'Dolo 650 side effects rash'")
    query_emb = embedder.embed_query("Dolo 650 side effects rash")
    results = vector_store.search(query_emb.tolist(), n_results=3)
    
    if results["documents"] and results["documents"][0]:
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            print(f"\n   Result {i+1} (source: {meta.get('source', 'unknown')}):")
            print(f"   {doc[:150]}...")
    else:
        print("   ⚠️ No results found - something may be wrong")


if __name__ == "__main__":
    main()
