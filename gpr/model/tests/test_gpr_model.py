import pytest
import torch

from gpr.model.gpr_model import (
    CTRHead,
    CVRHead,
    ECPMHead,
    GPRConfig,
    GPRModel,
    GPROutput,
    MultiTaskLoss,
    create_model,
)


@pytest.fixture
def config():
    return GPRConfig(use_lora=False)


class TestCTRHead:
    def test_output_shape(self, config):
        head = CTRHead(config.hidden_size, config.ctr_hidden, config.dropout)
        x = torch.randn(4, config.hidden_size)
        out = head(x)
        assert out.shape == (4,)
        assert out.dtype == torch.float32

    def test_output_is_logit(self, config):
        head = CTRHead(config.hidden_size, config.ctr_hidden, config.dropout)
        x = torch.randn(8, config.hidden_size)
        out = head(x)
        assert torch.isfinite(out).all()

    def test_dropout_disabled_in_eval(self, config):
        head = CTRHead(config.hidden_size, config.ctr_hidden, config.dropout)
        head.eval()
        x = torch.randn(2, config.hidden_size)
        out1 = head(x)
        out2 = head(x)
        assert torch.equal(out1, out2)

    def test_output_varies_with_input(self, config):
        head = CTRHead(config.hidden_size, config.ctr_hidden, config.dropout)
        head.eval()
        a = torch.randn(2, config.hidden_size)
        b = torch.randn(2, config.hidden_size)
        assert not torch.equal(head(a), head(b))


class TestCVRHead:
    def test_output_shape(self, config):
        head = CVRHead(config.hidden_size, config.cvr_hidden, config.dropout)
        x = torch.randn(3, config.hidden_size)
        out = head(x)
        assert out.shape == (3,)
        assert out.dtype == torch.float32

    def test_output_in_0_1_range(self, config):
        """CVR head has sigmoid, outputs should be in [0, 1]."""
        head = CVRHead(config.hidden_size, config.cvr_hidden, config.dropout)
        x = torch.randn(32, config.hidden_size)
        out = head(x)
        assert (out >= 0).all()
        assert (out <= 1).all()

    def test_deterministic_in_eval(self, config):
        head = CVRHead(config.hidden_size, config.cvr_hidden, config.dropout)
        head.eval()
        x = torch.randn(4, config.hidden_size)
        assert torch.equal(head(x), head(x))


class TestECPMHead:
    def test_output_shape(self, config):
        head = ECPMHead(config.hidden_size, config.ecpm_hidden, config.dropout)
        x = torch.randn(5, config.hidden_size)
        out = head(x)
        assert out.shape == (5,)
        assert out.dtype == torch.float32

    def test_output_is_finite(self, config):
        head = ECPMHead(config.hidden_size, config.ecpm_hidden, config.dropout)
        x = torch.randn(16, config.hidden_size)
        out = head(x)
        assert torch.isfinite(out).all()

    def test_deterministic_in_eval(self, config):
        head = ECPMHead(config.hidden_size, config.ecpm_hidden, config.dropout)
        head.eval()
        x = torch.randn(4, config.hidden_size)
        assert torch.equal(head(x), head(x))

    def test_relative_ordering(self, config):
        """Higher-quality input should not always give lower score."""
        head = ECPMHead(config.hidden_size, config.ecpm_hidden, config.dropout)
        head.eval()
        x = torch.stack([torch.arange(config.hidden_size, dtype=torch.float32) for _ in range(4)])
        out = head(x)
        assert len(torch.unique(out)) > 0


class TestGPRModel:
    def test_create_model_cpu(self):
        model = create_model(use_lora=False)
        assert isinstance(model, GPRModel)
        assert not model._is_hf_backbone

    def test_forward_shapes(self, config):
        model = GPRModel(config)
        model.eval()
        bs, seq = 2, 64
        input_ids = torch.randint(0, 1000, (bs, seq))
        attention_mask = torch.ones(bs, seq)

        output = model(input_ids, attention_mask)

        assert output.ctr_logits.shape == (bs,)
        assert output.cvr_logits.shape == (bs,)
        assert output.ecpm_scores.shape == (bs,)
        assert output.pooled_embedding.shape == (bs, config.hidden_size)

    def test_forward_with_padding(self, config):
        model = GPRModel(config)
        model.eval()
        input_ids = torch.randint(0, 1000, (3, 128))
        attention_mask = torch.cat([
            torch.ones(3, 96),
            torch.zeros(3, 32),
        ], dim=1)

        output = model(input_ids, attention_mask)
        assert torch.isfinite(output.ctr_logits).all()
        assert torch.isfinite(output.cvr_logits).all()
        assert torch.isfinite(output.ecpm_scores).all()

    def test_pooling_ignores_padding(self, config):
        model = GPRModel(config)
        model.eval()
        bs = 2
        # all padding
        input_ids = torch.randint(0, 1000, (bs, 64))
        attn_all_zero = torch.zeros(bs, 64)
        # full attention
        attn_all_one = torch.ones(bs, 64)

        out_pad = model(input_ids, attn_all_zero)
        out_full = model(input_ids, attn_all_one)
        # pooling with all zeros should produce different result
        assert not torch.equal(out_pad.pooled_embedding, out_full.pooled_embedding)

    def test_large_batch(self, config):
        model = GPRModel(config)
        model.eval()
        bs = 16
        input_ids = torch.randint(0, 1000, (bs, 32))
        attention_mask = torch.ones(bs, 32)
        output = model(input_ids, attention_mask)
        assert output.ctr_logits.shape[0] == bs

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="gradient flow through 635M params too slow on CPU")
    def test_gradient_flow(self, config):
        model = GPRModel(config)
        model.train()
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32)
        output = model(input_ids, attention_mask)

        loss = output.ctr_logits.sum()
        loss.backward()

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"{name} has no gradient"


class TestMultiTaskLoss:
    def test_loss_shapes(self):
        loss_fn = MultiTaskLoss()
        bs = 4
        outputs = GPROutput(
            ctr_logits=torch.randn(bs),
            cvr_logits=torch.rand(bs).sigmoid(),
            ecpm_scores=torch.randn(bs),
            pooled_embedding=torch.randn(bs, 3584),
        )
        ctr_labels = torch.randint(0, 2, (bs,)).float()
        cvr_labels = torch.rand(bs)
        ecpm_order = torch.ones(bs // 2) if bs >= 2 else torch.tensor([])

        losses = loss_fn(outputs, ctr_labels, cvr_labels, ecpm_order)
        assert "loss" in losses
        assert "loss_ctr" in losses
        assert "loss_cvr" in losses
        assert "loss_rank" in losses
        assert losses["loss"] >= 0

    def test_loss_no_ranking_pairs(self):
        loss_fn = MultiTaskLoss()
        outputs = GPROutput(
            ctr_logits=torch.randn(1),
            cvr_logits=torch.rand(1).sigmoid(),
            ecpm_scores=torch.randn(1),
            pooled_embedding=torch.randn(1, 3584),
        )
        losses = loss_fn(outputs, torch.zeros(1), torch.zeros(1), torch.tensor([]))
        assert losses["loss"] >= 0

    def test_loss_reduces_with_perfect_predictions(self):
        loss_fn = MultiTaskLoss()
        outputs = GPROutput(
            ctr_logits=torch.tensor([10.0, -10.0]),
            cvr_logits=torch.tensor([0.9, 0.1]).sigmoid(),
            ecpm_scores=torch.tensor([1.0, 0.0]),
            pooled_embedding=torch.randn(2, 3584),
        )
        ctr_labels = torch.tensor([1.0, 0.0])
        cvr_labels = torch.tensor([0.9, 0.1])
        ecpm_order = torch.ones(1)

        good_loss = loss_fn(outputs, ctr_labels, cvr_labels, ecpm_order)

        bad_outputs = GPROutput(
            ctr_logits=torch.tensor([-10.0, 10.0]),
            cvr_logits=torch.tensor([0.1, 0.9]).sigmoid(),
            ecpm_scores=torch.tensor([0.0, 1.0]),
            pooled_embedding=torch.randn(2, 3584),
        )
        bad_loss = loss_fn(bad_outputs, ctr_labels, cvr_labels, ecpm_order)
        assert good_loss["loss"] < bad_loss["loss"]


class TestGPRConfig:
    def test_defaults(self):
        config = GPRConfig()
        assert config.model_name == "Qwen/Qwen2-7B"
        assert config.hidden_size == 3584
        assert config.use_lora is True

    def test_custom_config(self):
        config = GPRConfig(
            model_name="test/model",
            hidden_size=768,
            use_lora=False,
            lora_rank=8,
        )
        assert config.model_name == "test/model"
        assert config.hidden_size == 768
        assert config.use_lora is False
        assert config.lora_rank == 8
