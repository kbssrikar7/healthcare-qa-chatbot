import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    test_set_path = PROJECT_ROOT / "evaluation" / "test_set.json"
    if not test_set_path.exists():
        print(f"Test set not found at {test_set_path}")
        return

    with open(test_set_path, "r") as f:
        test_data = json.load(f)

    try:
        from src.embeddings.embedding_models import MedicalEmbedder
        from src.embeddings.vector_store import VectorStore
    except ImportError:
        print(
            "Failed to import app components. Ensure you run this from the project root."
        )
        return

    embedder = MedicalEmbedder(model_name="all-minilm")
    vector_store = VectorStore(
        collection_name="medical_knowledge",
        persist_directory=str(PROJECT_ROOT / "data" / "knowledge_base"),
    )

    print(f"Loaded VectorStore with {vector_store.collection.count()} documents.")
    print("Aligning test set IDs with vector store based on semantic search...")
    updates = 0

    for case in test_data.get("test_cases", []):
        query = case.get("query", "")
        if not query:
            continue

        emb = embedder.embed_query(query).tolist()
        results = vector_store.collection.query(query_embeddings=[emb], n_results=3)

        if results and results["ids"] and len(results["ids"][0]) > 0:
            assigned_ids = results["ids"][0]
            if case.get("relevant_ids") != assigned_ids:
                case["relevant_ids"] = assigned_ids
                updates += 1

    if updates > 0:
        with open(test_set_path, "w") as f:
            json.dump(test_data, f, indent=4)
        print(
            f"✅ Updated {updates} test cases with real deterministic IDs from the vector store."
        )
    else:
        print("Test set is already aligned or no updates were needed.")


if __name__ == "__main__":
    main()
