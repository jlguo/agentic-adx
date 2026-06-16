"""
GPR model pre-training on Criteo/Avazu public ad datasets.
Transforms tabular data into text prompts and fine-tunes with LoRA.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CTRDataset(Dataset):
    def __init__(self, data_path: str, max_samples: int = 10000):
        samples = []
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                for i, line in enumerate(f):
                    if i >= max_samples:
                        break
                    samples.append(self._parse_line(line))
        else:
            rng = np.random.default_rng(42)
            for _ in range(max_samples):
                samples.append({
                    "prompt": f"User browsing electronics site. Ad: Wireless Earbuds - Premium audio.",
                    "ctr": float(rng.random() < 0.15),
                    "cvr": float(rng.random() < 0.03),
                    "ecpm": float(rng.uniform(0.1, 5.0)),
                })

        self.samples = samples

    @staticmethod
    def _parse_line(line: str) -> dict:
        parts = line.strip().split("\t")
        return {
            "prompt": f"User features: click history, device mobile. Ad: {parts[0] if len(parts) > 0 else 'sample'}.",
            "ctr": float(parts[1]) if len(parts) > 1 else 0.0,
            "cvr": float(parts[2]) if len(parts) > 2 else 0.0,
            "ecpm": float(parts[3]) if len(parts) > 3 else 1.0,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_batch(batch, tokenizer, max_length=512):
    texts = [item["prompt"] for item in batch]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    ctr = torch.tensor([item["ctr"] for item in batch], dtype=torch.float32)
    cvr = torch.tensor([item["cvr"] for item in batch], dtype=torch.float32)
    ecpm = torch.tensor([item["ecpm"] for item in batch], dtype=torch.float32)

    return encoded, ctr, cvr, ecpm


def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (encoded, ctr, cvr, ecpm) in enumerate(dataloader):
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        ctr = ctr.to(device)
        cvr = cvr.to(device)
        ecpm = ecpm.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        n = ecpm.size(0)
        half = n // 2
        ecpm_order = torch.ones(half, device=device) if half > 0 else torch.tensor([])

        losses = loss_fn(outputs, ctr, cvr, ecpm_order)
        loss = losses["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 100 == 0:
            print(f"  batch {batch_idx}: loss={loss.item():.4f} "
                  f"ctr={losses['loss_ctr']:.4f} cvr={losses['loss_cvr']:.4f}")

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Pre-train GPR model")
    parser.add_argument("--data", default="", help="Path to training data (tab-separated)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=5000, help="Max training samples")
    parser.add_argument("--output", default="gpr_pretrained.pt", help="Output model path")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
    from gpr_model import create_model, MultiTaskLoss

    print("Creating GPR model...")
    model = create_model(use_lora=True, lora_rank=16)
    model = model.to(device)

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except (ImportError, OSError) as e:
        print(f"Warning: Cannot load tokenizer: {e}")
        print("Using simple whitespace tokenizer for development.")
        from collections import defaultdict
        class SimpleTokenizer:
            def __init__(self):
                self.vocab = defaultdict(lambda: len(self.vocab))
                self.pad_token = "[PAD]"
                self.eos_token = "[EOS]"
                self.vocab[self.pad_token] = 0
                for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-":
                    self.vocab[c] = len(self.vocab)
            def __call__(self, texts, padding=True, truncation=True, max_length=512, return_tensors="pt"):
                input_ids = torch.zeros(len(texts), max_length, dtype=torch.long)
                attention_mask = torch.zeros(len(texts), max_length, dtype=torch.long)
                for i, text in enumerate(texts):
                    for j, ch in enumerate(text[:max_length]):
                        input_ids[i, j] = self.vocab[ch]
                        attention_mask[i, j] = 1
                return type('obj', (object,), {'input_ids': input_ids, 'attention_mask': attention_mask})()
        tokenizer = SimpleTokenizer()

    dataset = CTRDataset(args.data, max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=lambda b: collate_batch(b, tokenizer))

    loss_fn = MultiTaskLoss(alpha=1.0, beta=1.0, gamma=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Training on {len(dataset)} samples, {args.epochs} epochs")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        avg_loss = train_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f"  avg_loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), args.output)
    print(f"\nModel saved to {args.output}")


if __name__ == "__main__":
    main()
