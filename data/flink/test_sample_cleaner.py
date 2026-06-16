import os
import tempfile
import time
from pathlib import Path

import pytest

from data.flink.sample_cleaner import SampleCleaner, format_sample


class TestFormatSample:
    def test_basic_format(self):
        line = format_sample(ad_id="ad_001", ecpm=1500.0)
        assert line == "User browsing on site. Ad: ad_001.\t0.0\t0.0\t1500.000000"

    def test_impression_only_ctr0_cvr0(self):
        line = format_sample(ad_id="ad_001", clicked=False, converted=False, ecpm=1.5)
        parts = line.split("\t")
        assert float(parts[1]) == 0.0
        assert float(parts[2]) == 0.0

    def test_impression_plus_click_ctr1_cvr0(self):
        line = format_sample(ad_id="ad_001", clicked=True, converted=False, ecpm=2.0)
        parts = line.split("\t")
        assert float(parts[1]) == 1.0
        assert float(parts[2]) == 0.0

    def test_impression_plus_click_plus_conversion_ctr1_cvr1(self):
        line = format_sample(ad_id="ad_001", clicked=True, converted=True, ecpm=3.0)
        parts = line.split("\t")
        assert float(parts[1]) == 1.0
        assert float(parts[2]) == 1.0

    def test_conversion_without_click_edge_case(self):
        line = format_sample(ad_id="ad_001", clicked=False, converted=True, ecpm=4.0)
        parts = line.split("\t")
        assert float(parts[1]) == 0.0
        assert float(parts[2]) == 1.0

    def test_four_tab_separated_fields(self):
        line = format_sample(ad_id="ad_001", ecpm=1500.0)
        assert len(line.split("\t")) == 4

    def test_custom_domain(self):
        line = format_sample(ad_id="ad_42", domain="example.com", ecpm=500.0)
        assert line.startswith("User browsing on example.com. Ad: ad_42.\t")


class TestSampleCleanerJoinLogic:
    def test_impression_only_yields_ctr0_cvr0(self, tmp_path):
        output = tmp_path / "samples.tsv"
        cleaner = SampleCleaner(output_file=str(output))

        cleaner._on_impression("bid_1", {"ad_id": "ad_A", "ecpm": 1500.0})
        cleaner._flush_sample("bid_1")

        assert len(cleaner.buffer) == 1
        parts = cleaner.buffer[0].split("\t")
        ad_text, ctr_text, cvr_text, _ = parts
        assert float(ctr_text) == 0.0
        assert float(cvr_text) == 0.0
        assert "ad_A" in ad_text

    def test_impression_plus_click_yields_ctr1_cvr0(self, tmp_path):
        output = tmp_path / "samples.tsv"
        cleaner = SampleCleaner(output_file=str(output))

        cleaner._on_impression("bid_1", {"ad_id": "ad_B", "ecpm": 2000.0})
        cleaner._on_click("bid_1")
        cleaner._flush_sample("bid_1")

        assert len(cleaner.buffer) == 1
        _, ctr_text, cvr_text, _ = cleaner.buffer[0].split("\t")
        assert float(ctr_text) == 1.0
        assert float(cvr_text) == 0.0

    def test_impression_click_conversion_yields_ctr1_cvr1(self, tmp_path):
        output = tmp_path / "samples.tsv"
        cleaner = SampleCleaner(output_file=str(output))

        cleaner._on_impression("bid_2", {"ad_id": "ad_C", "ecpm": 2500.0})
        cleaner._on_click("bid_2")
        cleaner._on_conversion("bid_2")

        assert len(cleaner.buffer) == 1
        _, ctr_text, cvr_text, _ = cleaner.buffer[0].split("\t")
        assert float(ctr_text) == 1.0
        assert float(cvr_text) == 1.0

    def test_click_without_prior_impression_is_ignored(self, tmp_path):
        output = tmp_path / "samples.tsv"
        cleaner = SampleCleaner(output_file=str(output))
        cleaner._on_click("unknown_bid")
        assert len(cleaner.pending) == 0
        assert len(cleaner.buffer) == 0

    def test_conversion_flushes_immediately(self, tmp_path):
        output = tmp_path / "samples.tsv"
        cleaner = SampleCleaner(output_file=str(output))

        cleaner._on_impression("bid_3", {"ad_id": "ad_D", "ecpm": 3000.0})
        cleaner._on_conversion("bid_3")

        assert len(cleaner.buffer) == 1
        assert len(cleaner.pending) == 0

    def test_state_cleared_after_flush(self, tmp_path):
        output = tmp_path / "samples.tsv"
        cleaner = SampleCleaner(output_file=str(output))

        cleaner._on_impression("bid_4", {"ad_id": "ad_E", "ecpm": 100.0})
        assert "bid_4" in cleaner.pending

        cleaner._flush_sample("bid_4")
        assert "bid_4" not in cleaner.pending


class TestSampleCleanerTTL:
    def test_stale_samples_are_flushed(self, tmp_path, monkeypatch):
        output = tmp_path / "samples.tsv"
        cleaner = SampleCleaner(output_file=str(output))

        cleaner._on_impression("old_bid", {"ad_id": "ad_old", "ecpm": 50.0})
        cleaner.pending["old_bid"]["received_at"] = 0

        cleaner._expire_stale_samples()

        assert len(cleaner.buffer) == 1
        assert "old_bid" not in cleaner.pending

    def test_fresh_samples_not_flushed(self, tmp_path):
        output = tmp_path / "samples.tsv"
        cleaner = SampleCleaner(output_file=str(output))

        cleaner._on_impression("fresh_bid", {"ad_id": "ad_fresh", "ecpm": 100.0})

        cleaner._expire_stale_samples()

        assert len(cleaner.buffer) == 0
        assert "fresh_bid" in cleaner.pending


class TestSampleCleanerBuffer:
    def test_buffer_flushes_at_100_samples(self, tmp_path):
        output = tmp_path / "samples.tsv"
        cleaner = SampleCleaner(output_file=str(output))

        for i in range(105):
            cleaner._on_impression(f"bid_{i}", {"ad_id": f"ad_{i}", "ecpm": float(i)})
            cleaner._flush_sample(f"bid_{i}")

        assert output.exists()
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 100
        assert len(cleaner.buffer) == 5

    def test_shutdown_flushes_remaining(self, tmp_path):
        output = tmp_path / "samples.tsv"
        cleaner = SampleCleaner(output_file=str(output))

        cleaner._on_impression("bid_x", {"ad_id": "ad_x", "ecpm": 999.0})

        cleaner._shutdown()

        assert output.exists()
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_output_is_append_mode(self, tmp_path):
        output = tmp_path / "samples.tsv"
        output.write_text("existing line\n")

        cleaner = SampleCleaner(output_file=str(output))
        cleaner._on_impression("bid_y", {"ad_id": "ad_y", "ecpm": 1.0})
        cleaner._flush_sample("bid_y")
        cleaner._flush_buffer()

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "existing line"
