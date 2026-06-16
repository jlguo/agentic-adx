"""
Vector updater — polls MySQL for new/updated creatives, generates embeddings,
and upserts to Qdrant.
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
import datetime
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("vector_updater")

_RUNNING = True


def _shutdown_handler(signum, frame):
    global _RUNNING
    logger.info("Received signal %s, shutting down gracefully...", signum)
    _RUNNING = False


signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)


def build_ad_text(ad: Dict) -> str:
    parts = [
        f"Title: {ad.get('title', '')}",
        f"Description: {ad.get('description', '')}",
        f"Category: {ad.get('category', '')}",
    ]
    tags = ad.get("tags", [])
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except (json.JSONDecodeError, TypeError):
            tags = [tags]
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")
    return "\n".join(parts)


def generate_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        emb = model.encode([text], normalize_embeddings=True)
        return emb[0]
    except ImportError:
        logger.warning("sentence-transformers not installed, using random fallback")
        rng = np.random.default_rng(42)
        vec = rng.normal(0, 0.1, 384).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


def load_state(state_path: str) -> str:
    """Return the last-processed timestamp string, or epoch 0."""
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            data = json.load(f)
            return data.get("last_updated_at", "1970-01-01 00:00:00")
    return "1970-01-01 00:00:00"


def save_state(state_path: str, timestamp: str):
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "w") as f:
        json.dump({"last_updated_at": timestamp}, f)


def connect_mysql(host: str, user: str, password: str, database: str = "adx"):
    import pymysql
    try:
        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            charset="utf8mb4",
            connect_timeout=5,
        )
        logger.info("Connected to MySQL at %s", host)
        return conn
    except pymysql.err.OperationalError as e:
        logger.error("MySQL connection failed: %s", e)
        return None


_CURSOR_CLASS = None


def _get_dict_cursor():
    global _CURSOR_CLASS
    if _CURSOR_CLASS is None:
        try:
            import pymysql.cursors
            _CURSOR_CLASS = pymysql.cursors.DictCursor
        except ImportError:
            _CURSOR_CLASS = None
    return _CURSOR_CLASS


def query_new_creatives(conn, last_ts: str) -> List[Dict]:
    sql = """
        SELECT id, campaign_id, title, description, category, tags
        FROM creatives
        WHERE updated_at > %s
        ORDER BY updated_at ASC
    """
    cursor_cls = _get_dict_cursor()
    if cursor_cls is not None:
        with conn.cursor(cursor_cls) as cursor:
            cursor.execute(sql, (last_ts,))
            return cursor.fetchall()
    else:
        with conn.cursor() as cursor:
            cursor.execute(sql, (last_ts,))
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


def get_new_max_timestamp(rows: List[Dict]) -> str:
    """Return the current timestamp to use as next checkpoint.

    Since the query is ordered by updated_at ASC, the last row has the
    newest timestamp. We use datetime.now() as a safe upper bound.
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def upsert_to_qdrant(qdrant_url: str, collection: str, points: List[Dict]):
    if not points:
        return True

    import urllib.request
    import urllib.error

    url = f"{qdrant_url}/collections/{collection}/points"
    payload = json.dumps({"points": points}).encode("utf-8")

    req = urllib.request.Request(url, data=payload, method="PUT")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode()
            if resp.status != 200:
                logger.error("Qdrant upsert failed: HTTP %d %s", resp.status, body)
                return False
            logger.info("Upserted %d points to Qdrant", len(points))
            return True
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        logger.error("Qdrant HTTP error: %d %s", e.code, body)
        return False
    except urllib.error.URLError as e:
        logger.error("Qdrant connection error: %s", e.reason)
        return False


def process_creatives(rows: List[Dict], qdrant_url: str, collection: str) -> bool:
    if not rows:
        return True

    points = []
    for row in rows:
        text = build_ad_text(row)
        emb = generate_embedding(text)
        points.append({
            "id": row["id"],
            "vector": emb.tolist(),
            "payload": {
                "campaign_id": row.get("campaign_id", 0),
                "title": row.get("title", ""),
                "category": row.get("category", ""),
                "tags": row.get("tags", []),
                "bid_price": 0.0,
            },
        })

    return upsert_to_qdrant(qdrant_url, collection, points)


def main():
    parser = argparse.ArgumentParser(
        description="Poll MySQL for new creatives and upsert embeddings to Qdrant"
    )
    parser.add_argument("--mysql-host", default="localhost", help="MySQL host")
    parser.add_argument("--mysql-user", default="root", help="MySQL user")
    parser.add_argument("--mysql-pass", default="", help="MySQL password")
    parser.add_argument("--mysql-db", default="adx", help="MySQL database name")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant REST API base URL")
    parser.add_argument("--collection", default="ad_vectors", help="Qdrant collection name")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between polls (default: 60)")
    parser.add_argument("--state-file", default="data/.vector_updater_state.json",
                        help="Path to state file (default: data/.vector_updater_state.json)")
    args = parser.parse_args()

    state_path = os.path.normpath(args.state_file)

    while _RUNNING:
        conn = connect_mysql(args.mysql_host, args.mysql_user, args.mysql_pass, args.mysql_db)
        if conn is None:
            logger.info("Retrying in %d seconds...", args.poll_interval)
            time.sleep(args.poll_interval)
            continue

        try:
            last_ts = load_state(state_path)
            logger.info("Querying for creatives updated since %s", last_ts)
            rows = query_new_creatives(conn, last_ts)

            if rows:
                logger.info("Found %d new/updated creatives", len(rows))
                if process_creatives(rows, args.qdrant_url, args.collection):
                    new_ts = get_new_max_timestamp(rows)
                    save_state(state_path, new_ts)
                    logger.info("State updated to %s", new_ts)
                else:
                    logger.warning("Upsert failed, state NOT advanced")
            else:
                logger.info("No new creatives found")
        finally:
            conn.close()

        time.sleep(args.poll_interval)

    logger.info("Vector updater stopped.")


if __name__ == "__main__":
    main()
