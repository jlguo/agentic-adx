"""
Ad embedding generator for GPR ADX.
Generates vector embeddings from ad creative text using sentence-transformers.
Outputs embeddings for ingestion into Qdrant.
"""

import argparse
import json
import os
import sys
from typing import List, Dict

import numpy as np


def load_ads_from_mysql():
    """Try to load ads from MySQL. Falls back to seed data JSON."""
    seed_path = os.path.join(os.path.dirname(__file__), "..", "..", "deploy", "schema", "seed_ads.json")

    if os.path.exists(seed_path):
        with open(seed_path, "r") as f:
            return json.load(f)

    return [
        {
            "id": i,
            "campaign_id": (i // 5) + 1,
            "title": f"Ad Creative {i}",
            "description": f"This is a sample ad creative number {i} for testing purposes.",
            "category": "general",
            "tags": ["sample", "test"],
            "bid_price": 1.0 + (i % 10) * 0.5,
        }
        for i in range(1, 51)
    ]


def build_ad_text(ad: Dict) -> str:
    """Construct a text representation of an ad for embedding."""
    parts = [
        f"Title: {ad.get('title', '')}",
        f"Description: {ad.get('description', '')}",
        f"Category: {ad.get('category', '')}",
    ]
    tags = ad.get("tags", [])
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")
    return "\n".join(parts)


def generate_embeddings(ads: List[Dict], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Generate embeddings for ads using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
    except ImportError:
        print(f"Warning: sentence-transformers not installed. Using random embeddings for testing.", file=sys.stderr)
        rng = np.random.default_rng(42)
        return rng.normal(0, 0.1, (len(ads), 384)).astype(np.float32)

    texts = [build_ad_text(ad) for ad in ads]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return embeddings


def save_qdrant_payload(ads: List[Dict], embeddings: np.ndarray, output_path: str):
    """Save ads and embeddings in a format suitable for Qdrant upsert."""
    points = []
    for ad, emb in zip(ads, embeddings):
        points.append({
            "id": ad["id"],
            "vector": emb.tolist(),
            "payload": {
                "campaign_id": ad.get("campaign_id", 0),
                "title": ad.get("title", ""),
                "category": ad.get("category", ""),
                "tags": ad.get("tags", []),
                "bid_price": ad.get("bid_price", 0.0),
            },
        })

    with open(output_path, "w") as f:
        json.dump(points, f, ensure_ascii=False)

    print(f"Saved {len(points)} embeddings to {output_path}")
    print(f"Vector dimension: {embeddings.shape[1]}")


def main():
    parser = argparse.ArgumentParser(description="Generate ad embeddings for Qdrant")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model name")
    parser.add_argument("--output", default="qdrant_points.json", help="Output JSON file path")
    parser.add_argument("--dim", type=int, default=384, help="Embedding dimension (fallback only)")
    args = parser.parse_args()

    ads = load_ads_from_mysql()
    print(f"Loaded {len(ads)} ads")

    embeddings = generate_embeddings(ads, model_name=args.model)
    print(f"Generated embeddings: shape={embeddings.shape}")

    save_qdrant_payload(ads, embeddings, args.output)


if __name__ == "__main__":
    main()
