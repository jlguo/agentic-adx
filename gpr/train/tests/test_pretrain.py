import numpy as np
import pytest
import torch

from gpr.train.pretrain import CTRDataset, collate_batch


class TestCTRDdataset:
    def test_creates_without_file(self):
        dataset = CTRDataset("", max_samples=100)
        assert len(dataset) == 100

    def test_each_sample_has_required_keys(self):
        dataset = CTRDataset("", max_samples=50)
        for item in dataset:
            assert "prompt" in item
            assert "ctr" in item
            assert "cvr" in item
            assert "ecpm" in item

    def test_ctr_values_are_probabilities(self):
        dataset = CTRDataset("", max_samples=100)
        ctr_values = [item["ctr"] for item in dataset]
        assert all(0.0 <= v <= 1.0 for v in ctr_values)

    def test_cvr_values_are_probabilities(self):
        dataset = CTRDataset("", max_samples=100)
        cvr_values = [item["cvr"] for item in dataset]
        assert all(0.0 <= v <= 1.0 for v in cvr_values)

    def test_ecpm_values_are_positive(self):
        dataset = CTRDataset("", max_samples=100)
        ecpm_values = [item["ecpm"] for item in dataset]
        assert all(v > 0 for v in ecpm_values)

    def test_variable_lengths(self):
        dataset = CTRDataset("", max_samples=1000)
        prompts = [item["prompt"] for item in dataset]
        # synthetic fallback produces variable-length prompts via rng.random() < threshold
        unique_lengths = set(len(p) for p in prompts)
        assert len(unique_lengths) >= 1


class TestCollateBatch:
    @pytest.fixture
    def simple_tokenizer(self):
        class _Tok:
            def __call__(self, texts, padding, truncation, max_length, return_tensors):
                batch_size = len(texts)
                max_len = min(max(len(t.split()) for t in texts), max_length)
                return {
                    "input_ids": torch.randint(0, 1000, (batch_size, max_len)),
                    "attention_mask": torch.ones(batch_size, max_len, dtype=torch.long),
                }

        return _Tok()

    def test_output_shapes(self, simple_tokenizer):
        batch = [
            {"prompt": "ad text one", "ctr": 0.1, "cvr": 0.02, "ecpm": 2.5},
            {"prompt": "ad text two", "ctr": 0.15, "cvr": 0.03, "ecpm": 3.0},
            {"prompt": "ad text three longer", "ctr": 0.12, "cvr": 0.01, "ecpm": 1.8},
        ]
        encoded, ctr, cvr, ecpm = collate_batch(batch, simple_tokenizer)
        assert encoded["input_ids"].shape[0] == 3
        assert encoded["attention_mask"].shape[0] == 3
        assert ctr.shape == (3,)
        assert cvr.shape == (3,)
        assert ecpm.shape == (3,)
        assert ctr.dtype == torch.float32

    def test_single_item_batch(self, simple_tokenizer):
        batch = [{"prompt": "one ad", "ctr": 0.2, "cvr": 0.05, "ecpm": 4.0}]
        encoded, ctr, cvr, ecpm = collate_batch(batch, simple_tokenizer)
        assert encoded["input_ids"].shape[0] == 1
        assert ctr.item() == pytest.approx(0.2)
        assert cvr.item() == pytest.approx(0.05)
        assert ecpm.item() == pytest.approx(4.0)
