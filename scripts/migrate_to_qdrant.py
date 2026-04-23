import os
import chromadb
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from tqdm import tqdm
from loguru import logger

# Configuration
CHROMA_DIR = "/home/kbs/Documents/final_project/data/knowledge_base_v2"
CHROMA_COLLECTION = "medical_knowledge_v2"
QDRANT_URL = "https://5769e335-3865-41d8-bf3c-c6fd67c240fc.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")  # set QDRANT_API_KEY env var before running

def migrate():
    logger.info("Initializing ChromaDB connection...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_col = chroma_client.get_collection(name=CHROMA_COLLECTION)
    count = chroma_col.count()
    logger.info(f"Connected to Chroma collection '{CHROMA_COLLECTION}' with {count} documents.")

    logger.info("Initializing Qdrant connection...")
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    
    col_name = "medical_knowledge_v2"
    
    # We need to know the vector size. We can fetch 1 item from chroma to check.
    sample = chroma_col.get(limit=1, include=["embeddings"])
    if sample.get("embeddings") is None or len(sample.get("embeddings")) == 0:
        logger.error("No embeddings found in Chroma!")
        return
    
    vector_size = len(sample["embeddings"][0])
    logger.info(f"Vector size detected: {vector_size}")

    if not qdrant.collection_exists(col_name):
        logger.info(f"Creating Qdrant collection: {col_name}")
        qdrant.create_collection(
            collection_name=col_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    batch_size = 500
    logger.info(f"Starting migration to Qdrant Cloud in batches of {batch_size}...")
    
    for offset in tqdm(range(0, count, batch_size), desc="Migrating"):
        try:
            batch = chroma_col.get(limit=batch_size, offset=offset, include=["embeddings", "documents", "metadatas"])
            
            ids = batch["ids"]
            embeddings = batch.get("embeddings", [])
            documents = batch.get("documents", [])
            metadatas = batch.get("metadatas", [])
            
            points = []
            for i in range(len(ids)):
                # Handle possible missing metadata keys safely
                meta = metadatas[i] if metadatas and i < len(metadatas) and metadatas[i] else {}
                meta["page_content"] = documents[i]  # Store doc inside payload
                
                # We need numeric UUID-like strictly or string uuid. Chroma uses string hash IDs.
                # Qdrant accepts UUID strings. We can hash chroma ID to proper UUID
                import uuid
                import hashlib
                hash_id = hashlib.md5(ids[i].encode()).hexdigest()
                qdrant_id = str(uuid.UUID(hash_id))
                
                meta["chroma_id"] = ids[i] # keeping reference to old id
                
                points.append(
                    PointStruct(
                        id=qdrant_id,
                        vector=embeddings[i],
                        payload=meta
                    )
                )
            
            qdrant.upsert(
                collection_name=col_name,
                points=points
            )
            
        except Exception as e:
            logger.error(f"Failed at offset {offset}: {e}")
            break

    logger.info("Migration strictly completed!")

if __name__ == "__main__":
    migrate()
