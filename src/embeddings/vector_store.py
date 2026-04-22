"""
Vector store management — Qdrant Cloud with local ChromaDB fallback.

Tries Qdrant Cloud first. If the cloud is unreachable (DNS / network error),
automatically falls back to the local ChromaDB persist directory.
"""

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger


def generate_deterministic_id(
    source: str,
    content: str,
    chunk_id: int = 0,
) -> str:
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    key = f"{source}::chunk{chunk_id}::{content_hash}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


class VectorStore:
    """
    Vector store with Qdrant Cloud primary and local ChromaDB fallback.
    Exposes the same method signatures regardless of backend.
    """

    def __init__(
        self,
        collection_name: str = "medical_knowledge_v2",
        persist_directory: str = "data/knowledge_base_v2",
        embedding_function=None,
        hnsw_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self._meta_path = self.persist_directory / "embedding_metadata.json"
        self.collection = self  # ChromaDB-style interface alias

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            logger.warning(
                "QDRANT_URL or QDRANT_API_KEY not set — using local ChromaDB fallback. "
                "Note: ChromaDB fallback requires KB_PERSIST_DIR to exist locally."
            )

        # Try Qdrant Cloud first; fall back to local ChromaDB on any connection error
        if qdrant_url and qdrant_api_key:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, PointStruct, VectorParams

                client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=10)
                # Ping to confirm connectivity
                client.get_collections()
                self._backend = "qdrant"
                self._qdrant = client
                self._qdrant_VectorParams = VectorParams
                self._qdrant_Distance = Distance
                self._qdrant_PointStruct = PointStruct
                if not client.collection_exists(collection_name):
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                    )
                col_info = client.get_collection(collection_name)
                logger.info(f"Vector store: Qdrant Cloud — {collection_name} ({col_info.points_count} docs)")
            except Exception as e:
                logger.warning(f"Qdrant Cloud unreachable ({e}), falling back to local ChromaDB")
                self._backend = "chromadb"
                self._init_chromadb(hnsw_metadata)
        else:
            self._backend = "chromadb"
            self._init_chromadb(hnsw_metadata)

    def _init_chromadb(self, hnsw_metadata: Optional[Dict] = None):
        import chromadb
        from chromadb.config import Settings

        settings = Settings(anonymized_telemetry=False)
        chroma_client = chromadb.PersistentClient(
            path=str(self.persist_directory), settings=settings
        )
        meta = hnsw_metadata or {}
        try:
            self._chroma_col = chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata=meta or None,
            )
        except Exception:
            self._chroma_col = chroma_client.get_collection(name=self.collection_name)
        logger.info(
            f"Vector store: local ChromaDB — {self.collection_name} ({self._chroma_col.count()} docs)"
        )

    # ── ChromaDB-style count/get (used by BM25 warm-up) ──────────────────────

    def count(self) -> int:
        if self._backend == "qdrant":
            try:
                return self._qdrant.get_collection(self.collection_name).points_count or 0
            except Exception:
                return 0
        return self._chroma_col.count()

    def get(self, limit: int = 100, offset: int = 0, include: List[str] = None) -> Dict:
        if self._backend == "qdrant":
            try:
                scroll_results, _ = self._qdrant.scroll(
                    collection_name=self.collection_name,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=bool(include and "embeddings" in include),
                )
                docs, metas, ids, embs = [], [], [], []
                for pt in scroll_results:
                    payload = pt.payload or {}
                    docs.append(payload.get("page_content", ""))
                    m = payload.copy()
                    m.pop("page_content", None)
                    metas.append(m)
                    ids.append(payload.get("chroma_id", str(pt.id)))
                    if pt.vector:
                        embs.append(pt.vector)
                return {"ids": ids, "documents": docs, "metadatas": metas, "embeddings": embs}
            except Exception as e:
                logger.error(f"Qdrant scroll error: {e}")
                return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
        # ChromaDB
        inc = include or ["documents", "metadatas"]
        return self._chroma_col.get(limit=limit, offset=offset, include=inc)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        filter_metadata: Optional[Dict] = None,
    ) -> Dict:
        if self._backend == "qdrant":
            return self._search_qdrant(query_embedding, n_results)
        return self._search_chromadb(query_embedding, n_results, filter_metadata)

    def _search_qdrant(self, query_embedding, n_results):
        try:
            res = self._qdrant.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=n_results,
                with_payload=True,
            ).points
        except Exception as e:
            logger.warning(f"query_points failed ({e}), using scroll fallback")
            res = self._qdrant.scroll(
                collection_name=self.collection_name, limit=n_results, with_payload=True
            )[0]
        docs, metas, dists, ids = [], [], [], []
        for r in res:
            payload = r.payload or {}
            docs.append(payload.get("page_content", ""))
            m = payload.copy(); m.pop("page_content", None)
            metas.append(m)
            dists.append(1.0 - max(0.0, getattr(r, "score", 0.0)))
            ids.append(payload.get("chroma_id", str(r.id)))
        return {"documents": [docs], "metadatas": [metas], "distances": [dists], "ids": [ids]}

    def _search_chromadb(self, query_embedding, n_results, filter_metadata):
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self._chroma_col.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata
        return self._chroma_col.query(**kwargs)

    def mmr_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter_metadata: Optional[Dict] = None,
    ) -> Dict:
        if self._backend == "qdrant":
            return self._mmr_qdrant(query_embedding, k, fetch_k, lambda_mult)
        # ChromaDB: fetch more then apply MMR manually
        raw = self._search_chromadb(query_embedding, fetch_k, filter_metadata)
        docs = raw["documents"][0]
        metas = raw["metadatas"][0]
        dists = raw["distances"][0]
        if not docs:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        # Greedy MMR: relevance to query minus redundancy to already-selected docs
        # Re-fetch with embeddings included for doc-doc similarity
        raw_with_emb = self._chroma_col.query(
            query_embeddings=[query_embedding],
            n_results=min(fetch_k, self._chroma_col.count() or 1),
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        emb_list = raw_with_emb.get("embeddings", [[]])[0]
        emb_matrix = np.array(emb_list) if emb_list else None

        selected, remaining = [], list(range(len(docs)))
        while len(selected) < k and remaining:
            scores = []
            for idx in remaining:
                rel = 1.0 - dists[idx]
                if emb_matrix is not None and selected:
                    sel_embs = emb_matrix[selected]
                    idx_emb = emb_matrix[idx]
                    norms = np.linalg.norm(sel_embs, axis=1) * np.linalg.norm(idx_emb) + 1e-9
                    red = float((sel_embs @ idx_emb / norms).max())
                else:
                    red = 0.0
                scores.append((lambda_mult * rel - (1 - lambda_mult) * red, idx))
            best = max(scores)[1]
            selected.append(best)
            remaining.remove(best)
        return {
            "documents": [[docs[i] for i in selected]],
            "metadatas": [[metas[i] for i in selected]],
            "distances": [[dists[i] for i in selected]],
        }

    def _mmr_qdrant(self, query_embedding, k, fetch_k, lambda_mult):
        try:
            res = self._qdrant.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=fetch_k,
                with_payload=True,
                with_vectors=True,
            ).points
        except Exception:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        if not res:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        e_docs, e_metas, e_dists, e_embs = [], [], [], []
        for r in res:
            payload = r.payload or {}
            e_docs.append(payload.get("page_content", ""))
            m = payload.copy(); m.pop("page_content", None)
            e_metas.append(m)
            e_dists.append(1.0 - max(0.0, r.score))
            e_embs.append(r.vector)
        emb_arr = np.array(e_embs)
        sims = 1 - np.array(e_dists)
        selected, remaining = [], list(range(len(e_docs)))
        while len(selected) < k and remaining:
            mmr = []
            for idx in remaining:
                rel = sims[idx]
                red = max(np.dot(emb_arr[selected], emb_arr[idx])) if selected else 0.0
                mmr.append((lambda_mult * rel - (1 - lambda_mult) * red, idx))
            best = max(mmr)[1]
            selected.append(best)
            remaining.remove(best)
        return {
            "documents": [[e_docs[i] for i in selected]],
            "metadatas": [[e_metas[i] for i in selected]],
            "distances": [[e_dists[i] for i in selected]],
        }

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        if metadatas is None:
            metadatas = [{}] * len(documents)
        if ids is None:
            ids = [
                generate_deterministic_id(
                    source=meta.get("source", ""),
                    content=doc,
                    chunk_id=meta.get("chunk_id", i),
                )
                for i, (doc, meta) in enumerate(zip(documents, metadatas))
            ]
        if self._backend == "qdrant":
            from qdrant_client.models import PointStruct
            points = []
            for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                payload = meta.copy()
                payload["page_content"] = doc
                payload["chroma_id"] = ids[i]
                q_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, ids[i]))
                points.append(PointStruct(id=q_id, vector=embeddings[i], payload=payload))
            self._qdrant.upsert(collection_name=self.collection_name, points=points)
        else:
            self._chroma_col.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        return ids

    # ── Metadata helpers ──────────────────────────────────────────────────────

    def verify_embedding_compatibility(self, model_name: str, dimension: int) -> bool:
        try:
            if self._meta_path.exists():
                with open(self._meta_path) as f:
                    meta = json.load(f)
                stored = meta.get("embedding_model")
                if stored and stored != model_name:
                    logger.warning("Embedding model mismatch. Quality may drop.")
                    return False
        except Exception:
            pass
        return True

    def record_embedding_model(self, model_name: str, dimension: int) -> None:
        import datetime
        try:
            meta = {
                "embedding_model": model_name,
                "embedding_dimension": dimension,
                "recorded_at": datetime.datetime.now(datetime.UTC).isoformat(),
            }
            with open(self._meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not write embedding metadata: {e}")

    def get_stats(self) -> Dict:
        return {"name": self.collection_name, "count": self.count()}

    def delete_collection(self):
        if self._backend == "qdrant":
            self._qdrant.delete_collection(self.collection_name)
        else:
            self._chroma_col._client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")
