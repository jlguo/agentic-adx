import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from data.flink.training_trigger import (
    archive_data_file,
    count_lines,
    trigger_training,
)
from data.flink.vector_updater import (
    build_ad_text,
    generate_embedding,
    load_state,
    save_state,
    query_new_creatives,
    upsert_to_qdrant,
    process_creatives,
    get_new_max_timestamp,
)


# ---------------------------------------------------------------------------
# training_trigger tests
# ---------------------------------------------------------------------------

class TestCountLines:
    def test_returns_zero_for_missing_file(self):
        assert count_lines("/nonexistent/path.tsv") == 0

    def test_counts_non_empty_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("line1\t0.1\t0.02\t1.5\n")
            f.write("\n")
            f.write("line2\t0.2\t0.03\t2.0\n")
            f.write("\n")
            f.write("line3\t0.15\t0.01\t1.8\n")
            f.flush()
            path = f.name
        try:
            assert count_lines(path) == 3
        finally:
            os.unlink(path)

    def test_empty_file_returns_zero(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("")
            f.flush()
            path = f.name
        try:
            assert count_lines(path) == 0
        finally:
            os.unlink(path)


class TestArchiveDataFile:
    def test_renames_file_with_timestamp_suffix(self):
        archived = None
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("data\n")
            f.flush()
            path = f.name
        try:
            archived = archive_data_file(path)
            assert not os.path.exists(path)
            assert os.path.exists(archived)
            assert os.path.basename(archived).startswith(os.path.basename(path))
            with open(archived) as fh:
                assert fh.read().strip() == "data"
        finally:
            for p in [path, archived]:
                if os.path.exists(p):
                    os.unlink(p)


class TestTriggerTraining:
    def test_calls_pretrain_with_correct_args(self):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            exit_code = trigger_training("/tmp/test_data.tsv", epochs=3, batch_size=16)
            assert exit_code == 0
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert call_args[-6:] == [
                "--data", "/tmp/test_data.tsv",
                "--epochs", "3",
                "--batch-size", "16",
            ]

    def test_returns_nonzero_on_failure(self):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            exit_code = trigger_training("/tmp/test_data.tsv", epochs=2, batch_size=8)
            assert exit_code == 1


# ---------------------------------------------------------------------------
# vector_updater tests
# ---------------------------------------------------------------------------

class TestBuildAdText:
    def test_full_ad_with_tags_as_list(self):
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

    def test_tags_as_json_string(self):
        ad = {
            "title": "Test",
            "description": "Desc",
            "category": "general",
            "tags": '["tag1", "tag2"]',
        }
        text = build_ad_text(ad)
        assert "Tags: tag1, tag2" in text

    def test_no_tags_field(self):
        ad = {
            "title": "Shoes",
            "description": "Running shoes",
            "category": "sports",
        }
        text = build_ad_text(ad)
        assert "Tags:" not in text

    def test_empty_tags_list(self):
        ad = {
            "title": "Item",
            "description": "Desc",
            "category": "misc",
            "tags": [],
        }
        text = build_ad_text(ad)
        assert len(text) > 0


class TestGenerateEmbedding:
    def test_returns_384d_vector(self):
        vec = generate_embedding("test ad text")
        assert len(vec) == 384
        assert vec.dtype == np.float32

    def test_normalized_output(self):
        vec = generate_embedding("some creative text")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5


class TestStatePersistence:
    def test_load_state_returns_default_for_missing_file(self):
        ts = load_state("/nonexistent/state.json")
        assert ts == "1970-01-01 00:00:00"

    def test_save_and_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.close()
            path = f.name
        try:
            save_state(path, "2025-06-10 12:00:00")
            ts = load_state(path)
            assert ts == "2025-06-10 12:00:00"
        finally:
            os.unlink(path)

    def test_save_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = os.path.join(tmp, "sub", "state.json")
            save_state(state_path, "2025-06-10 12:00:00")
            assert os.path.exists(state_path)


class TestGetNewMaxTimestamp:
    def test_returns_valid_timestamp_format(self):
        ts = get_new_max_timestamp([])
        parts = ts.split(" ")
        assert len(parts) == 2


class TestQueryNewCreatives:
    def test_uses_last_timestamp_in_sql(self):
        mock_conn = mock.MagicMock()
        mock_cursor = mock.MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        query_new_creatives(mock_conn, "2025-06-10 12:00:00")

        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0]
        assert "WHERE updated_at >" in call_args[0]
        assert call_args[1] == ("2025-06-10 12:00:00",)

    def test_returns_list_of_dicts(self):
        expected = [
            {"id": 1, "campaign_id": 10, "title": "Ad 1",
             "description": "Desc 1", "category": "cat1", "tags": ["t1"]},
            {"id": 2, "campaign_id": 20, "title": "Ad 2",
             "description": "Desc 2", "category": "cat2", "tags": ["t2"]},
        ]

        mock_conn = mock.MagicMock()
        mock_cursor = mock.MagicMock()
        mock_cursor.fetchall.return_value = expected
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        rows = query_new_creatives(mock_conn, "1970-01-01 00:00:00")
        assert rows == expected
        assert len(rows) == 2


class TestUpsertToQdrant:
    def test_sends_correct_payload_format(self):
        points = [{
            "id": 1,
            "vector": [0.1, 0.2, 0.3],
            "payload": {
                "campaign_id": 10,
                "title": "Test Ad",
                "category": "tech",
                "tags": ["cool"],
                "bid_price": 2.5,
            },
        }]

        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = mock.MagicMock()
            mock_resp.status = 200
            mock_resp.read.return_value = b'{"result":{"operation_id":1,"status":"completed"}}'
            mock_urlopen.return_value.__enter__.return_value = mock_resp

            result = upsert_to_qdrant(
                "http://localhost:6333", "ad_vectors", points
            )
            assert result is True

            req = mock_urlopen.call_args[0][0]
            assert req.method == "PUT"
            assert "collections/ad_vectors/points" in req.full_url

            body = json.loads(req.data)
            assert len(body["points"]) == 1
            assert body["points"][0]["id"] == 1
            assert body["points"][0]["vector"] == [0.1, 0.2, 0.3]
            assert body["points"][0]["payload"]["campaign_id"] == 10

    def test_handles_http_error(self):
        import urllib.error
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.HTTPError(
                "http://localhost:6333", 500, "Internal Error",
                {}, mock.MagicMock()
            )

            result = upsert_to_qdrant(
                "http://localhost:6333", "ad_vectors",
                [{"id": 1, "vector": [0.1], "payload": {}}]
            )
            assert result is False

    def test_handles_connection_error(self):
        import urllib.error
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

            result = upsert_to_qdrant(
                "http://localhost:6333", "ad_vectors",
                [{"id": 1, "vector": [0.1], "payload": {}}]
            )
            assert result is False

    def test_empty_points_returns_early(self):
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            result = upsert_to_qdrant("http://localhost:6333", "ad_vectors", [])
            assert result is True
            mock_urlopen.assert_not_called()


class TestProcessCreatives:
    def test_empty_rows_returns_true(self):
        result = process_creatives([], "http://localhost:6333", "ad_vectors")
        assert result is True

    def test_processes_creative_and_upserts(self):
        rows = [{
            "id": 42,
            "campaign_id": 7,
            "title": "Summer Sale",
            "description": "Big discounts",
            "category": "retail",
            "tags": ["sale", "summer"],
        }]

        fake_vec = np.zeros(384, dtype=np.float32)
        with mock.patch("data.flink.vector_updater.generate_embedding", return_value=fake_vec), \
             mock.patch("data.flink.vector_updater.upsert_to_qdrant", return_value=True) as mock_upsert:

            result = process_creatives(rows, "http://localhost:6333", "ad_vectors")
            assert result is True

            upsert_args, _ = mock_upsert.call_args
            qdrant_url, collection, upl_points = upsert_args
            assert len(upl_points) == 1
            assert upl_points[0]["id"] == 42
            assert upl_points[0]["payload"]["campaign_id"] == 7
            assert upl_points[0]["payload"]["title"] == "Summer Sale"
            assert upl_points[0]["payload"]["category"] == "retail"
            assert len(upl_points[0]["vector"]) == 384


# ---------------------------------------------------------------------------
# Integration: training_trigger flow (mock subprocess)
# ---------------------------------------------------------------------------

class TestTrainingTriggerFlow:
    def test_archive_then_train_sequence(self):
        call_order = []
        archived = None

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            for i in range(10):
                f.write(f"sample_{i}\t0.1\t0.02\t1.0\n")
            f.flush()
            orig_path = f.name

        real_rename = os.rename
        def _track_rename(src, dst):
            call_order.append(("archive", src, dst))
            real_rename(src, dst)

        try:
            with mock.patch("subprocess.run") as mock_run, \
                 mock.patch("os.rename", side_effect=_track_rename):
                mock_run.return_value.returncode = 0

                archived = archive_data_file(orig_path)
                call_order.append(("archive_returned", archived))
                trigger_training(archived, epochs=3, batch_size=16)

            assert call_order[0][0] == "archive"
            assert call_order[1][0] == "archive_returned"
            assert os.path.exists(archived)
            assert not os.path.exists(orig_path)
        finally:
            for p in [orig_path, archived]:
                if os.path.exists(p):
                    os.unlink(p)


# ---------------------------------------------------------------------------
# Integration: vector_updater end-to-end (all mocked)
# ---------------------------------------------------------------------------

class TestVectorUpdaterE2E:
    def test_full_pipeline_mocked(self):
        rows = [
            {"id": 1, "campaign_id": 10, "title": "Ad A",
             "description": "Desc A", "category": "c1", "tags": ["t1"]},
            {"id": 2, "campaign_id": 20, "title": "Ad B",
             "description": "Desc B", "category": "c2", "tags": ["t2"]},
        ]

        fake_vec = np.zeros(384, dtype=np.float32)
        with mock.patch("data.flink.vector_updater.generate_embedding", return_value=fake_vec), \
             mock.patch("data.flink.vector_updater.upsert_to_qdrant", return_value=True) as mock_upsert, \
             mock.patch("data.flink.vector_updater.connect_mysql"), \
             mock.patch("data.flink.vector_updater.query_new_creatives", return_value=rows), \
             mock.patch("data.flink.vector_updater.load_state", return_value="2025-06-01 00:00:00"), \
             mock.patch("data.flink.vector_updater.save_state") as mock_save, \
             mock.patch("data.flink.vector_updater.get_new_max_timestamp", return_value="2025-06-10 12:00:00"):

            process_creatives(rows, "http://localhost:6333", "ad_vectors")
            mock_save("data/.vector_updater_state.json", "2025-06-10 12:00:00")

            upsert_args, _ = mock_upsert.call_args
            upl_points = upsert_args[2]
            assert len(upl_points) == 2
            assert upl_points[0]["id"] == 1
            assert upl_points[1]["id"] == 2
            assert all(len(p["vector"]) == 384 for p in upl_points)
