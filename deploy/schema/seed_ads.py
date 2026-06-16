#!/usr/bin/env python3
"""
Seed Qdrant vector DB with initial ad creatives.

Reads deploy/schema/seed_ads.json, generates embeddings via
sentence-transformers, and upserts to Qdrant collection 'ad_vectors'.
"""

import json
import sys
from pathlib import Path


def _load_embedding_model():
    """Lazy-load the sentence-transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"  WARNING: Cannot load sentence-transformers: {e}")
        print("  Using random 384-dim vectors as fallback.")
        return None


def _random_vector() -> list[float]:
    import random
    return [random.random() for _ in range(384)]


def main() -> int:
    seed_file = Path(__file__).resolve().parent / "schema" / "seed_ads.json"
    if not seed_file.exists():
        seed_file = Path(__file__).resolve().parent.parent / "schema" / "seed_ads.json"
    if not seed_file.exists():
        print(f"ERROR: seed_ads.json not found (tried: {seed_file})")
        return 1

    with open(seed_file) as f:
        ads = json.load(f)

    print(f"Loading {len(ads)} seed ads from {seed_file}")

    model = _load_embedding_model()

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels
    except ImportError:
        print("ERROR: qdrant-client not installed. pip install qdrant-client")
        return 1

    client = QdrantClient("localhost", port=6333)
    collection_name = "ad_vectors"
    vector_size = 384

    # Recreate collection
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=vector_size,
            distance=qmodels.Distance.COSINE,
        ),
    )
    print(f"  Created collection '{collection_name}' ({vector_size}-dim, cosine)")

    points = []
    for ad in ads:
        title = ad.get("title", "")
        desc = ad.get("description", "")
        text = f"{title}. {desc}"

        if model is not None:
            vector = model.encode(text).tolist()
        else:
            vector = _random_vector()

        payload = {
            "ad_id": ad["id"],
            "campaign_id": ad["campaign_id"],
            "title": title,
            "description": desc,
            "category": ad.get("category", ""),
            "tags": ad.get("tags", []),
            "bid_price": ad.get("bid_price", 1.0),
        }

        points.append(
            qmodels.PointStruct(
                id=ad["id"],
                vector=vector,
                payload=payload,
            )
        )

    client.upsert(collection_name=collection_name, points=points)
    print(f"  Upserted {len(points)} vectors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
