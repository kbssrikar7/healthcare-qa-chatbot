"""
Build a small, curated knowledge-base subset for the mobile (Android) spike.

Derives the subset from what the desktop system's dense retriever actually
returns for the 97-question paper eval set (evaluation/test_set_v2.json).
See project_paperwork/scratch/mobile_port_notes.md ("KB Subset Construction")
for why this method was chosen over hand-picking documents, and for why the
originally-planned "neighbor chunk" expansion was dropped: the KB's `source`
metadata field is a dataset-level label (e.g. "MedQuAD"), not a per-document
ID, and `chunk_id` only resets to 0 within one original document — so
(source, chunk_id) does not uniquely identify a document's neighbor chunk and
matches thousands of unrelated documents instead. There's no stable
per-document ID in the metadata to do this safely, so this script ships the
direct top-k-per-question set only.

Output: evaluation/mobile_kb_subset.jsonl — one JSON object per line:
    {"id": str, "text": str, "source": str, "chunk_id": int, "embedding": [384 floats]}

Usage:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 venv/bin/python evaluation/build_mobile_kb_subset.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import EmbeddingConfig
from src.embeddings.embedding_models import MedicalEmbedder
from src.embeddings.vector_store import VectorStore

TOP_K = 15
TEST_SET_PATH = Path(__file__).parent / "test_set_v2.json"
OUTPUT_PATH = Path(__file__).parent / "mobile_kb_subset.jsonl"
KB_PERSIST_DIR = "data/knowledge_base_v2"


def main() -> None:
    print(f"Loading embedder ({EmbeddingConfig().model_name})...")
    embedder = MedicalEmbedder(model_name=EmbeddingConfig().model_name)

    print(f"Loading vector store ({KB_PERSIST_DIR})...")
    vs = VectorStore(persist_directory=KB_PERSIST_DIR)

    test_set = json.loads(TEST_SET_PATH.read_text())
    questions = [tc["query"] for tc in test_set["test_cases"]]
    print(f"Loaded {len(questions)} eval questions from {TEST_SET_PATH.name}")

    # Step 1: collect top-k retrieved doc ids per question.
    seen_ids: set[str] = set()

    for i, question in enumerate(questions, 1):
        q_emb = embedder.embed_query(question)
        results = vs.search(q_emb.tolist(), n_results=TOP_K)
        ids = results["ids"][0]
        seen_ids.update(ids)
        if i % 20 == 0 or i == len(questions):
            print(f"  retrieved for {i}/{len(questions)} questions "
                  f"({len(seen_ids)} unique docs so far)")

    print(f"Final subset size: {len(seen_ids)} unique chunks "
          f"(top-{TOP_K} per question, {len(questions)} questions).")

    # Step 2: batch-fetch full records (text + metadata + embedding) for the
    # final id set. ChromaDB get() has no hard batch limit here, but chunk it
    # anyway to keep memory bounded and give progress feedback.
    all_ids = sorted(seen_ids)
    BATCH = 100
    written = 0
    with open(OUTPUT_PATH, "w") as f:
        for start in range(0, len(all_ids), BATCH):
            batch_ids = all_ids[start:start + BATCH]
            got = vs._chroma_col.get(
                ids=batch_ids,
                include=["documents", "metadatas", "embeddings"],
            )
            for doc_id, text, meta, emb in zip(
                got["ids"], got["documents"], got["metadatas"], got["embeddings"]
            ):
                record = {
                    "id": doc_id,
                    "text": text,
                    "source": meta.get("source", ""),
                    "chunk_id": meta.get("chunk_id", 0),
                    "embedding": [round(float(x), 6) for x in emb],
                }
                f.write(json.dumps(record) + "\n")
                written += 1
            print(f"  wrote {written}/{len(all_ids)} records")

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nDone. Wrote {written} records to {OUTPUT_PATH} ({size_mb:.1f} MB).")


if __name__ == "__main__":
    main()
