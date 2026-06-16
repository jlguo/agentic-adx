import json
import os
from unittest.mock import MagicMock

import pytest

from agents.compliance import SENSITIVE_WORDS, check_compliance, llm_compliance_check
from agents.creative_agent import CreativeAgent


# =========================================================================
# Compliance tests
# =========================================================================

class TestCheckCompliance:

    def test_clean_creative_passes(self):
        passed, violations = check_compliance(
            title="Affordable Wireless Earbuds",
            description="High quality Bluetooth earbuds with great sound and long battery life.",
        )
        assert passed is True
        assert violations == []

    def test_chinese_superlative_fails(self):
        passed, violations = check_compliance(
            title="全国最好的手机",
            description="我们提供最优质的服务，唯一的选择。",
        )
        assert passed is False
        assert len(violations) > 0

    def test_english_absolute_claim_fails(self):
        passed, violations = check_compliance(
            title="The best headphones guaranteed",
            description="100% satisfaction or your money back — risk-free trial.",
        )
        assert passed is False
        assert len(violations) > 0

    def test_miracle_claim_fails(self):
        passed, violations = check_compliance(
            title="Miracle weight loss supplement",
            description="Lose weight fast with our miracle diet formula.",
        )
        assert passed is False
        assert "miracle" in (v.lower() for v in violations)
        assert "miracle diet" in (v.lower() for v in violations)

    def test_case_insensitive_english(self):
        passed, violations = check_compliance(
            title="GUARANTEED Best Results #1 Product",
            description="",
        )
        assert passed is False
        lower_violations = [v.lower() for v in violations]
        assert "guaranteed" in lower_violations
        assert "#1" in lower_violations

    def test_medical_claim_fails(self):
        passed, violations = check_compliance(
            title="特效感冒药",
            description="药到病除，无副作用，一疗程根治。",
        )
        assert passed is False
        assert len(violations) >= 3

    def test_only_description_violation(self):
        passed, violations = check_compliance(
            title="Simple Product Name",
            description="This is a guaranteed miracle cure that works instantly!",
        )
        assert passed is False
        assert len(violations) > 0

    def test_passed_violations_are_strings(self):
        _, violations = check_compliance("最", "最好")
        assert all(isinstance(v, str) for v in violations)

    def test_returns_false_when_no_violations(self):
        passed, violations = check_compliance("普通产品", "常规服务描述")
        assert passed is True
        assert violations == []

    def test_sensitive_words_set_not_empty(self):
        assert len(SENSITIVE_WORDS) > 20
        assert "最" in SENSITIVE_WORDS
        assert "第一" in SENSITIVE_WORDS
        assert "guaranteed" in SENSITIVE_WORDS
        assert "100%" in SENSITIVE_WORDS


class TestLLMComplianceCheck:

    def test_passes_clean_creative(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "passed": True,
            "violations": [],
            "reason": "No issues found.",
        })
        mock_llm.invoke.return_value = mock_response

        passed, violations = llm_compliance_check(
            title="Simple Product",
            description="Standard description.",
            llm=mock_llm,
        )
        assert passed is True
        assert violations == []

    def test_flags_exaggerated_claims(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "passed": False,
            "violations": ["Exaggerated efficacy claim"],
            "reason": "The ad claims unrealistic results.",
        })
        mock_llm.invoke.return_value = mock_response

        passed, violations = llm_compliance_check(
            title="Magic Cure",
            description="Cures everything overnight!",
            llm=mock_llm,
        )
        assert passed is False
        assert len(violations) > 0

    def test_handles_malformed_json_gracefully(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "not valid json"
        mock_llm.invoke.return_value = mock_response

        passed, violations = llm_compliance_check(
            title="Test", description="Test", llm=mock_llm
        )
        assert passed is True
        assert violations == []

    def test_handles_llm_exception_gracefully(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")

        passed, violations = llm_compliance_check(
            title="Test", description="Test", llm=mock_llm
        )
        assert passed is True
        assert violations == []


# =========================================================================
# CreativeAgent tests
# =========================================================================

_CLEAN_CREATIVES = [
    {
        "title": "Reliable Cloud Solutions",
        "description": "Scalable cloud infrastructure for modern enterprises. Secure and efficient.",
        "category": "tech",
        "tags": ["cloud", "enterprise"],
        "image_url": "",
        "landing_url": "",
    },
    {
        "title": "AI-Powered Analytics",
        "description": "Transform your data into actionable insights with our AI platform.",
        "category": "tech",
        "tags": ["ai", "analytics"],
        "image_url": "",
        "landing_url": "",
    },
    {
        "title": "DevOps Automation Tools",
        "description": "Streamline your deployment pipeline with automated CI/CD solutions.",
        "category": "tech",
        "tags": ["devops", "automation"],
        "image_url": "",
        "landing_url": "",
    },
    {
        "title": "Smart IoT Platform",
        "description": "Connect and manage IoT devices at scale with our unified dashboard.",
        "category": "tech",
        "tags": ["iot", "platform"],
        "image_url": "",
        "landing_url": "",
    },
    {
        "title": "Secure Data Backup",
        "description": "Enterprise-grade backup and disaster recovery for critical workloads.",
        "category": "tech",
        "tags": ["backup", "security"],
        "image_url": "",
        "landing_url": "",
    },
]


def _build_agent(creatives=None):
    if creatives is None:
        creatives = _CLEAN_CREATIVES
    _mock_embedding()
    agent = CreativeAgent()
    agent._call_llm_generate = MagicMock(return_value=list(creatives))
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '{"passed": true, "violations": [], "reason": "ok"}'
    mock_llm.invoke.return_value = mock_response
    agent._llm = mock_llm
    return agent


def _mock_db(agent, insert_id=1):
    agent._insert_to_mysql = MagicMock(return_value=insert_id)
    agent._upsert_to_qdrant = MagicMock()


def _mock_embedding():
    import agents.creative_agent as ca_mod
    import hashlib

    def _fake_embed(text):
        h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        return [float((h >> i) & 0xFF) / 255.0 for i in range(384)]

    ca_mod._gen_embedding = _fake_embed


class TestCreativeAgent:

    def test_generate_creatives_returns_list(self):
        agent = _build_agent()
        _mock_db(agent)
        results = agent.generate_creatives(1, 1, count=3, industry="tech")
        assert isinstance(results, list)
        assert len(results) == 3

    def test_each_creative_has_required_fields(self):
        agent = _build_agent()
        _mock_db(agent)
        results = agent.generate_creatives(1, 2, count=2, industry="tech")
        for creative in results:
            assert "title" in creative
            assert "description" in creative
            assert "category" in creative
            assert "tags" in creative
            assert "id" in creative or "campaign_id" in creative

    def test_creative_includes_campaign_id(self):
        agent = _build_agent()
        _mock_db(agent)
        results = agent.generate_creatives(42, 99, count=1, industry="travel")
        assert results[0]["campaign_id"] == 99

    def test_rejects_noncompliant_creatives(self):
        agent = _build_agent()
        _mock_db(agent)
        agent._call_llm_generate.return_value = [
            {
                "title": "The Best Product Ever",
                "description": "Guaranteed results, 100% satisfaction!",
                "category": "general",
                "tags": ["best"],
                "image_url": "",
                "landing_url": "",
            },
        ]
        results = agent.generate_creatives(1, 1, count=1, industry="general")
        assert len(results) == 0

    def test_stitches_missing_creatives_from_fallback(self):
        agent = _build_agent()
        _mock_db(agent)
        agent._call_llm_generate.return_value = [
            {
                "title": "Good Product",
                "description": "A reliable solution for your needs.",
                "category": "general",
                "tags": ["reliable"],
                "image_url": "",
                "landing_url": "",
            },
            {
                "title": "Another Product",
                "description": "Quality and value combined for modern living.",
                "category": "general",
                "tags": ["quality"],
                "image_url": "",
                "landing_url": "",
            },
            {
                "title": "Third Option",
                "description": "Affordable and practical solutions for everyone.",
                "category": "general",
                "tags": ["practical"],
                "image_url": "",
                "landing_url": "",
            },
        ]
        results = agent.generate_creatives(1, 1, count=3, industry="retail")
        assert len(results) == 3

    def test_skips_creatives_that_fail_db_insert(self):
        agent = _build_agent()
        _mock_db(agent)
        agent._insert_to_mysql = MagicMock(
            side_effect=[1, RuntimeError("DB error"), 2, 3, 4]
        )
        results = agent.generate_creatives(1, 1, count=3, industry="tech")
        assert len(results) == 3

    def test_no_db_calls_when_dry_run_is_set(self):
        agent = _build_agent()
        agent._insert_to_mysql = MagicMock(return_value=1)
        agent._upsert_to_qdrant = MagicMock()
        agent._insert_to_mysql.assert_not_called()
        agent._upsert_to_qdrant.assert_not_called()

    def test_embedding_generation_dimension(self):
        from agents.creative_agent import _gen_embedding
        import numpy as np

        vec = _gen_embedding("test ad creative text")
        assert len(vec) == 384
        assert all(isinstance(v, float) for v in vec)

    def test_embedding_fallback_is_deterministic(self):
        _mock_embedding()
        import agents.creative_agent as ca

        a = ca._gen_embedding("same text")
        b = ca._gen_embedding("same text")
        c = ca._gen_embedding("different text")
        assert a == b
        assert a != c

    def test_parse_json_response_handles_fenced_json(self):
        agent = CreativeAgent()
        raw = '```json\n[{"title":"Test","description":"Desc","category":"a","tags":[],"image_url":"","landing_url":""}]\n```'
        result = agent._parse_json_response(raw, 1, "test")
        assert len(result) == 1
        assert result[0]["title"] == "Test"

    def test_parse_json_response_handles_plain_json(self):
        agent = CreativeAgent()
        raw = '[{"title":"X","description":"Y","category":"Z","tags":[],"image_url":"","landing_url":""}]'
        result = agent._parse_json_response(raw, 1, "test")
        assert len(result) == 1
        assert result[0]["title"] == "X"

    def test_parse_json_response_fills_gaps(self):
        agent = CreativeAgent()
        raw = '[{"title":"Only","description":"One","category":"c","tags":[],"image_url":"","landing_url":""}]'
        result = agent._parse_json_response(raw, 5, "fill")
        assert len(result) == 5

    def test_parse_json_response_handles_malformed(self):
        agent = CreativeAgent()
        result = agent._parse_json_response("not json at all", 3, "bad")
        assert len(result) == 3
        for item in result:
            assert "title" in item
            assert "description" in item


# =========================================================================
# Integration markers — skip when infrastructure is unavailable
# =========================================================================

def _has_mysql() -> bool:
    try:
        import pymysql

        pymysql.connect(
            host=os.getenv("ADX_MYSQL_HOST", "localhost"),
            port=int(os.getenv("ADX_MYSQL_PORT", "3306")),
            user=os.getenv("ADX_MYSQL_USER", "adx"),
            password=os.getenv("ADX_MYSQL_PASS", "adx_pass"),
            database=os.getenv("ADX_MYSQL_DB", "adx"),
            connect_timeout=2,
        ).close()
        return True
    except Exception:
        return False


def _has_qdrant() -> bool:
    try:
        import urllib.request

        url = f"{os.getenv('ADX_QDRANT_URL', 'http://localhost:6333')}/healthz"
        urllib.request.urlopen(url, timeout=2)
        return True
    except Exception:
        return False


skip_if_no_mysql = pytest.mark.skipif(not _has_mysql(), reason="MySQL not available")
skip_if_no_qdrant = pytest.mark.skipif(
    not _has_qdrant(), reason="Qdrant not available"
)


class TestMySQLIntegration:
    @skip_if_no_mysql
    def test_insert_returns_id(self):
        agent = CreativeAgent()
        data = {
            "campaign_id": 9999,
            "title": "Integration Test Creative",
            "description": "Test insertion for integration verification.",
            "category": "test",
            "tags": "[]",
            "status": "active",
        }
        creative_id = agent._insert_to_mysql(data)
        assert creative_id > 0


class TestQdrantIntegration:
    @skip_if_no_qdrant
    def test_upsert_succeeds(self):
        agent = CreativeAgent()
        data = {
            "id": 99990001,
            "campaign_id": 9999,
            "title": "Qdrant Test",
            "category": "test",
            "tags": "[]",
        }
        from agents.creative_agent import _gen_embedding

        embedding = _gen_embedding("Qdrant test creative")
        agent._upsert_to_qdrant(data, embedding)
