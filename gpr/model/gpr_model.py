"""
GPR (Generative Pre-trained Recommendation) Model Architecture.
Qwen2-7B backbone with custom multi-task prediction heads for CTR/CVR/eCPM.

Based on the architecture spec in draft/product-idea.md Section 4.3.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPRConfig:
    model_name: str = "Qwen/Qwen2-7B"
    hidden_size: int = 3584
    ctr_hidden: int = 512
    cvr_hidden: int = 512
    ecpm_hidden: int = 512
    dropout: float = 0.1
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass
class GPROutput:
    ctr_logits: torch.Tensor
    cvr_logits: torch.Tensor
    ecpm_scores: torch.Tensor
    pooled_embedding: Optional[torch.Tensor] = None


class CTRHead(nn.Module):
    """CTR prediction: binary classification (click/no-click)."""

    def __init__(self, hidden_size: int, ctr_hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ctr_hidden)
        self.fc2 = nn.Linear(ctr_hidden, ctr_hidden // 2)
        self.fc3 = nn.Linear(ctr_hidden // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.gelu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x).squeeze(-1)


class CVRHead(nn.Module):
    """CVR prediction: regression (conversion probability)."""

    def __init__(self, hidden_size: int, cvr_hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, cvr_hidden)
        self.fc2 = nn.Linear(cvr_hidden, cvr_hidden // 2)
        self.fc3 = nn.Linear(cvr_hidden // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.gelu(self.fc2(x))
        x = self.dropout(x)
        return self.sigmoid(self.fc3(x)).squeeze(-1)


class ECPMHead(nn.Module):
    """eCPM scoring: ranking score for auction ordering."""

    def __init__(self, hidden_size: int, ecpm_hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ecpm_hidden)
        self.fc2 = nn.Linear(ecpm_hidden, ecpm_hidden // 2)
        self.fc3 = nn.Linear(ecpm_hidden // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.gelu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x).squeeze(-1)


class GPRModel(nn.Module):
    """
    GPR Unified Scoring Model.

    Architecture:
        Qwen2-7B (or compatible) backbone
        → Global mean pooling over last hidden state
        → [CTR Head | CVR Head | eCPM Head]
        → Structured output: [ctr, cvr, ecpm]

    This is a discriminative model — no token generation occurs.
    Only a single forward pass with structured output.
    """

    def __init__(self, config: GPRConfig):
        super().__init__()
        self.config = config

        try:
            from transformers import AutoModel, AutoConfig
            hf_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)
            self.backbone = AutoModel.from_pretrained(
                config.model_name,
                config=hf_config,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            self._is_hf_backbone = True
        except (ImportError, OSError) as e:
            print(f"Warning: Cannot load {config.model_name}: {e}")
            print("Using random initialized transformer for development.")
            self._is_hf_backbone = False
            self.backbone_embed = nn.Embedding(config.hidden_size, config.hidden_size)
            self.backbone = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=16,
                    dim_feedforward=config.hidden_size * 4,
                    batch_first=True,
                ),
                num_layers=4,
            )

        if config.use_lora:
            self._apply_lora()

        self.ctr_head = CTRHead(config.hidden_size, config.ctr_hidden, config.dropout)
        self.cvr_head = CVRHead(config.hidden_size, config.cvr_hidden, config.dropout)
        self.ecpm_head = ECPMHead(config.hidden_size, config.ecpm_hidden, config.dropout)

    def _apply_lora(self):
        try:
            from peft import LoraConfig, get_peft_model
            peft_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            )
            self.backbone = get_peft_model(self.backbone, peft_config)
        except ImportError:
            print("Warning: peft not installed. Skipping LoRA configuration.")

    def pool_embedding(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Global mean pooling over non-padding tokens."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> GPROutput:
        if self._is_hf_backbone:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.last_hidden_state
        else:
            # CPU fallback: TransformerEncoder takes src/src_key_padding_mask
            src = self.backbone_embed(input_ids)
            src_key_padding_mask = (attention_mask == 0)
            hidden_states = self.backbone(
                src=src,
                src_key_padding_mask=src_key_padding_mask,
            )

        pooled = self.pool_embedding(hidden_states, attention_mask)

        ctr_logits = self.ctr_head(pooled)
        cvr_logits = self.cvr_head(pooled)
        ecpm_scores = self.ecpm_head(pooled)

        return GPROutput(
            ctr_logits=ctr_logits,
            cvr_logits=cvr_logits,
            ecpm_scores=ecpm_scores,
            pooled_embedding=pooled,
        )


class MultiTaskLoss(nn.Module):
    """
    Combined loss: L_total = α·L_ctr + β·L_cvr + γ·L_rank

    - CTR: BCE loss (click prediction)
    - CVR: Smooth L1 loss (conversion probability)
    - eCPM: Margin ranking loss (correct ad ordering)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.margin = nn.MarginRankingLoss(margin=0.1)

    def forward(
        self,
        outputs: GPROutput,
        ctr_labels: torch.Tensor,
        cvr_labels: torch.Tensor,
        ecpm_order: torch.Tensor,
    ) -> dict:
        loss_ctr = self.bce(outputs.ctr_logits, ctr_labels)
        loss_cvr = self.smooth_l1(outputs.cvr_logits, cvr_labels)

        n = outputs.ecpm_scores.size(0)
        half = n // 2
        better = outputs.ecpm_scores[:half]
        worse = outputs.ecpm_scores[half:]
        target = torch.ones_like(better)
        loss_rank = self.margin(better, worse, target) if half > 0 else torch.tensor(0.0)

        total = self.alpha * loss_ctr + self.beta * loss_cvr + self.gamma * loss_rank

        return {
            "loss": total,
            "loss_ctr": loss_ctr.item(),
            "loss_cvr": loss_cvr.item(),
            "loss_rank": loss_rank.item() if isinstance(loss_rank, torch.Tensor) else loss_rank,
        }


def create_model(
    model_name: str = "Qwen/Qwen2-7B",
    use_lora: bool = True,
    lora_rank: int = 16,
) -> GPRModel:
    config = GPRConfig(
        model_name=model_name,
        use_lora=use_lora,
        lora_rank=lora_rank,
    )
    return GPRModel(config)


if __name__ == "__main__":
    model = create_model(use_lora=False)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created successfully")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
