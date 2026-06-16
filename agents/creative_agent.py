"""
Creative Generation Agent for GPR ADX.

LangChain-based agent that generates ad creatives, performs two-layer
compliance verification, and persists approved creatives to MySQL + Qdrant.

IMPORTANT — This agent operates on the AI Agent Control side-path (Layer 5).
It is NEVER invoked in the synchronous RTB hot path (Layer 2).
"""

import argparse
import json
import logging
import os
import random
import struct
import time
import urllib.request
from typing import Any, Dict, List, Optional

import numpy as np

from agents.compliance import check_compliance, llm_compliance_check

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default connection strings (overridable via environment variables)
# ---------------------------------------------------------------------------
_DB_HOST = os.getenv("ADX_MYSQL_HOST", "localhost")
_DB_PORT = int(os.getenv("ADX_MYSQL_PORT", "3306"))
_DB_USER = os.getenv("ADX_MYSQL_USER", "adx")
_DB_PASS = os.getenv("ADX_MYSQL_PASS", "adx_pass")
_DB_NAME = os.getenv("ADX_MYSQL_DB", "adx")
_QDRANT_URL = os.getenv("ADX_QDRANT_URL", "http://localhost:6333")
_QDRANT_COLLECTION = os.getenv("ADX_QDRANT_COLLECTION", "ad_vectors")
_VLLM_ENDPOINT = os.getenv("ADX_VLLM_ENDPOINT", "http://localhost:8000/v1")
_EMBEDDING_DIM = 384


_embedding_model: Optional[Any] = None


def _gen_embedding(text: str) -> List[float]:
    """Return a 384-dim embedding for *text*.

    Uses ``sentence-transformers`` when available; falls back to a
    reproducible random vector for offline / test environments.
    """
    global _embedding_model

    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _embedding_model = False  # sentinel — use fallback

    if _embedding_model is False:
        rng = np.random.default_rng(abs(hash(text)) % (2 ** 31))
        return rng.normal(0, 0.1, _EMBEDDING_DIM).astype(np.float32).tolist()

    vec = _embedding_model.encode([text], normalize_embeddings=True)[0]  # type: ignore[union-attr]
    return vec.tolist()


class CreativeAgent:
    """LangChain-based creative generation agent.

    Generates ad creatives via an LLM, validates them through a
    two-layer compliance pipeline, and stores approved creatives
    in MySQL + Qdrant.

    Parameters
    ----------
    llm_endpoint: OpenAI-compatible API base URL (defaults to vLLM).
    """

    def __init__(self, llm_endpoint: str = _VLLM_ENDPOINT):
        self._llm_endpoint = llm_endpoint.rstrip("/")
        self._llm = None  # lazy init

    # -- lazy LLM accessor --------------------------------------------------
    @property
    def _lc_llm(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI  # type: ignore

            self._llm = ChatOpenAI(
                model="gpr-adx",
                openai_api_key="adx-gpr-serve",
                openai_api_base=self._llm_endpoint,
                temperature=0.8,
                max_tokens=2048,
            )
        return self._llm

    # -- public API ---------------------------------------------------------

    def generate_creatives(
        self,
        account_id: int,
        campaign_id: int,
        count: int = 5,
        industry: str = "general",
    ) -> List[Dict[str, Any]]:
        """Generate *count* ad creatives for the given campaign.

        Each creative passes through four stages:
        1. LLM generation via ChatPromptTemplate
        2. Layer-1 rule-based compliance scan
        3. Layer-2 LLM semantic compliance re-check
        4. Persistence: MySQL INSERT + Qdrant upsert

        Returns a list of creative dicts (including inserted IDs).
        """
        creatives_raw = self._call_llm_generate(count, industry)
        results = []
        rejected_count = 0
        max_rejected = count * 3  # avoid infinite retry loops

        for creative in creatives_raw:
            if len(results) >= count:
                break
            if rejected_count >= max_rejected:
                logger.warning(
                    "Too many rejected creatives (%d). Stopping early.", rejected_count
                )
                break

            title = creative.get("title", "").strip()
            description = creative.get("description", "").strip()

            # Stage 1 — rule-based compliance
            rule_passed, rule_violations = check_compliance(title, description)

            # Stage 2 — LLM semantic compliance re-check
            llm_passed, llm_violations = self._llm_compliance_check(title, description)

            all_violations = rule_violations + llm_violations
            if not rule_passed or not llm_passed:
                logger.info(
                    "Creative rejected — violations: %s",
                    ", ".join(all_violations),
                )
                rejected_count += 1
                continue

            # Stage 3 — persistence
            creative_data = {
                "campaign_id": campaign_id,
                "title": title,
                "description": description,
                "image_url": creative.get("image_url", ""),
                "landing_url": creative.get("landing_url", ""),
                "category": creative.get("category", industry),
                "tags": json.dumps(creative.get("tags", [])),
                "status": "active",
            }

            try:
                creative_id = self._insert_to_mysql(creative_data)
                creative_data["id"] = creative_id
            except Exception as exc:
                logger.error("MySQL insert failed: %s", exc)
                continue

            # Stage 4 — Qdrant upsert
            try:
                vec_text = f"{title}\n{description}\n{creative_data['category']}"
                embedding = _gen_embedding(vec_text)
                self._upsert_to_qdrant(creative_data, embedding)
            except Exception as exc:
                logger.error("Qdrant upsert failed: %s", exc)
                # Non-fatal — creative is already in MySQL.

            results.append(creative_data)

        return results

    # -- internal helpers ---------------------------------------------------

    def _call_llm_generate(self, count: int, industry: str) -> List[Dict]:
        from langchain_core.prompts import ChatPromptTemplate  # type: ignore

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "You are a professional ad copywriter. "
                    "Generate {count} ad creatives for the {industry} industry.\n"
                    "Output ONLY a JSON array. Each element must have these fields:\n"
                    '  title: string (max 150 chars),\n'
                    '  description: string (max 500 chars),\n'
                    '  category: string,\n'
                    '  tags: array of strings,\n'
                    '  image_url: string (can be empty ""),\n'
                    '  landing_url: string (can be empty "").\n'
                    "Do NOT include any text outside the JSON array."
                ),
            ),
            ("human", "{count} {industry}"),
        ])

        chain = prompt | self._lc_llm
        response = chain.invoke({"count": str(count), "industry": industry})

        raw = response.content if hasattr(response, "content") else str(response)
        return self._parse_json_response(raw, count, industry)

    def _parse_json_response(
        self, raw: str, count: int, industry: str
    ) -> List[Dict]:
        raw = raw.strip()
        # LLMs sometimes wrap JSON in ``` fences.
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON: %.200s...", raw)
            data = []

        if not isinstance(data, list):
            data = [data]

        # Pad with random fallbacks if LLM returned fewer than requested.
        while len(data) < count:
            data.append(self._fallback_creative(industry))
        return data

    def _fallback_creative(self, industry: str) -> Dict[str, Any]:
        templates = [
            {
                "title": f"Premium {industry.title()} Solutions",
                "description": f"Discover our {industry} offerings tailored to your needs. Quality service and competitive pricing.",
            },
            {
                "title": f"Your Trusted {industry.title()} Partner",
                "description": f"We deliver excellence in {industry} with years of proven expertise and innovation.",
            },
            {
                "title": f"Smart {industry.title()} for Modern Business",
                "description": f"Upgrade your {industry} strategy with our cutting-edge platform. Fast, reliable, scalable.",
            },
        ]
        base = random.choice(templates)
        return {
            "title": base["title"],
            "description": base["description"],
            "category": industry,
            "tags": [industry, "professional"],
            "image_url": "",
            "landing_url": "",
        }

    def _llm_compliance_check(self, title: str, description: str) -> tuple:
        """Second-layer compliance: LLM semantic review."""
        return llm_compliance_check(title, description, self._lc_llm)

    # -- persistence --------------------------------------------------------

    def _insert_to_mysql(self, data: Dict[str, Any]) -> int:
        import pymysql

        conn = pymysql.connect(
            host=_DB_HOST,
            port=_DB_PORT,
            user=_DB_USER,
            password=_DB_PASS,
            database=_DB_NAME,
            charset="utf8mb4",
            connect_timeout=5,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO creatives
                        (campaign_id, title, description, image_url,
                         landing_url, category, tags, status)
                    VALUES
                        (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        data["campaign_id"],
                        data["title"],
                        data["description"],
                        data.get("image_url", ""),
                        data.get("landing_url", ""),
                        data.get("category", ""),
                        data.get("tags", "[]"),
                        data.get("status", "active"),
                    ),
                )
                conn.commit()
                return cur.lastrowid or 0
        finally:
            conn.close()

    def _upsert_to_qdrant(self, data: Dict[str, Any], embedding: List[float]):
        point_id = int(data.get("id", int(time.time() * 1000)))
        payload = {
            "campaign_id": data["campaign_id"],
            "title": data["title"],
            "category": data.get("category", ""),
            "tags": data.get("tags", "[]"),
        }

        body = json.dumps({
            "points": [
                {
                    "id": point_id,
                    "vector": embedding,
                    "payload": payload,
                }
            ]
        }).encode("utf-8")

        url = f"{_QDRANT_URL}/collections/{_QDRANT_COLLECTION}/points"
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        urllib.request.urlopen(req, timeout=10)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPR ADX Creative Agent")
    parser.add_argument(
        "--account-id", type=int, required=True,
        help="Advertiser account ID",
    )
    parser.add_argument(
        "--campaign-id", type=int, required=True,
        help="Campaign ID to associate creatives with",
    )
    parser.add_argument(
        "--count", type=int, default=5,
        help="Number of creatives to generate (default: 5)",
    )
    parser.add_argument(
        "--industry", default="general",
        help="Industry vertical (default: general)",
    )
    parser.add_argument(
        "--llm-endpoint", default=_VLLM_ENDPOINT,
        help=f"vLLM / OpenAI-compatible endpoint (default: {_VLLM_ENDPOINT})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate creative text but skip MySQL + Qdrant persistence",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    agent = CreativeAgent(llm_endpoint=args.llm_endpoint)

    if args.dry_run:
        original_insert = agent._insert_to_mysql
        original_upsert = agent._upsert_to_qdrant
        agent._insert_to_mysql = lambda d: (  # type: ignore[method-assign]
            logger.info("DRY-RUN: would insert title=%r", d.get("title")),
            struct.unpack("!I", os.urandom(4))[0],
        )[1]
        agent._upsert_to_qdrant = lambda d, e: logger.info(  # type: ignore[method-assign]
            "DRY-RUN: would upsert id=%d", d.get("id", 0)
        )
        try:
            creatives = agent.generate_creatives(
                args.account_id, args.campaign_id, args.count, args.industry
            )
        finally:
            agent._insert_to_mysql = original_insert
            agent._upsert_to_qdrant = original_upsert
    else:
        creatives = agent.generate_creatives(
            args.account_id, args.campaign_id, args.count, args.industry
        )

    if creatives:
        print(json.dumps(creatives, ensure_ascii=False, indent=2))
    else:
        print("No creatives generated.")


if __name__ == "__main__":
    main()
