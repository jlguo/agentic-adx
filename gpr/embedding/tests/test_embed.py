import json
import os
import tempfile

import numpy as np
import pytest

from gpr.embedding.embed import (
    build_ad_text,
    generate_embeddings,
    load_ads_from_mysql,
    save_qdrant_payload,
)


class TestBuildAdText:
    def test_full_ad(self):
        ad = {
            "title": "Wireless Earbuds",
            "description": "Premium audio quality",
            "category": "electronics",
            "tags": ["audio", "wireless"],
        }
        text = build_ad_text(ad)
        assert "Title: Wireless Earbuds" in text
        assert "Description: Premium audio quality" in text
        assert "Category: electronics" in text
        assert "Tags: audio, wireless" in text

    def test_ad_without_tags(self):
        ad = {
            "title": "Test",
            "description": "Desc",
            "category": "general",
        }
        text = build_ad_text(ad)
        assert text is not None
        assert len(text) > 0

    def test_minimal_ad(self):
        ad = {}
        text = build_ad_text(ad)
        assert "Title:" in text
        assert "Description:" in text


class TestLoadAdsFromMySQL:
    def test_returns_list(self):
        ads = load_ads_from_mysql()
        assert isinstance(ads, list)
        assert len(ads) > 0

    def test_each_ad_has_required_fields(self):
        ads = load_ads_from_mysql()
        for ad in ads:
            assert "id" in ad
            assert "title" in ad
            assert "bid_price" in ad

    def test_returns_seed_data_when_no_file(self):
        ads = load_ads_from_mysql()
        # seed generates 50 ads by default
        assert len(ads) >= 50


class TestGenerateEmbeddings:
    def test_shape_fallback(self):
        ads = load_ads_from_mysql()[:10]
        embeddings = generate_embeddings(ads)
        assert embeddings.shape == (10, 384)
        assert embeddings.dtype == np.float32

    def test_all_embeddings_different(self):
        ads = load_ads_from_mysql()[:20]
        embeddings = generate_embeddings(ads)
        norms = np.linalg.norm(embeddings, axis=1)
        assert len(np.unique(norms.round(2))) > 1


class TestSaveQdrantPayload:
    def test_saves_valid_json(self):
        ads = load_ads_from_mysql()[:5]
        embeddings = generate_embeddings(ads)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.close()
            try:
                save_qdrant_payload(ads, embeddings, f.name)
                with open(f.name) as fh:
                    data = json.load(fh)
                assert len(data) == 5
                for point in data:
                    assert "id" in point
                    assert "vector" in point
                    assert "payload" in point
                    assert len(point["vector"]) == 384
            finally:
                os.unlink(f.name)

    def test_payload_fields(self):
        ads = load_ads_from_mysql()[:3]
        embeddings = generate_embeddings(ads)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.close()
            try:
                save_qdrant_payload(ads, embeddings, f.name)
                with open(f.name) as fh:
                    data = json.load(fh)
                for point in data:
                    assert "bid_price" in point["payload"]
                    assert "title" in point["payload"]
                    assert "category" in point["payload"]
            finally:
                os.unlink(f.name)
