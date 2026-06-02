"""
Train the baseline Reasoning Core transformer (no memory, no search).
Used as the control system in experiments.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sbi.core.transformer import ReasoningCore
from sbi.core.config import ReasoningConfig
from data.babi.dataset import BabiDataset, build_tokenizer
from data.babi.loader import load_babi_tasks


def get_lr(step: int, warmup_steps: int, lr: float) -> float:
    if step < warmup_steps:
        return lr * step / warmup_steps
    return lr


def evaluate(model: ReasoningCore, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for input_ids, target_ids in loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=0,  # PAD_ID
            )
            total_loss += loss.item()
            n_batches += 1

    model.train()
    return total_loss / max(1, n_batches)


def answer_accuracy(model: ReasoningCore, loader: DataLoader, device: torch.device) -> float:
    """Exact-match accuracy on answer tokens only."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, target_ids in loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)
            preds = logits.argmax(dim=-1)

            # Only count positions where target is not PAD (i.e. answer positions)
            mask = target_ids != 0
            correct += ((preds == target_ids) & mask).sum().item()
            total += mask.sum().item()

    model.train()
    return correct / max(1, total)


def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build datasets — tokenizer is built from train split
    dc = cfg["data"]
    print("Loading bAbI tasks:", dc["task_ids"])

    train_dataset = BabiDataset(
        task_ids=dc["task_ids"],
        split="train",
        max_seq_len=dc["max_seq_len"],
        size_per_task=dc.get("size_per_task"),
        data_dir=dc.get("data_dir"),
        allow_synthetic_fallback=dc.get("allow_synthetic_fallback", False),
    )
    eval_dataset = BabiDataset(
        task_ids=dc["task_ids"],
        split="test",
        tokenizer=train_dataset.tokenizer,   # share vocab
        max_seq_len=dc["max_seq_len"],
        data_dir=dc.get("data_dir"),
        allow_synthetic_fallback=dc.get("allow_synthetic_fallback", False),
    )

    vocab_size = train_dataset.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train examples: {len(train_dataset)}, Eval examples: {len(eval_dataset)}")

    mc = cfg["model"]
    reasoning_cfg = ReasoningConfig(
        vocab_size=vocab_size,
        max_seq_len=mc["max_seq_len"],
        n_layers=mc["n_layers"],
        n_heads=mc["n_heads"],
        d_model=mc["d_model"],
        d_ff=mc["d_ff"],
        dropout=mc["dropout"],
        hidden_dim=mc["d_model"],
    )
    model = ReasoningCore(reasoning_cfg).to(device)
    print(f"Parameters: {model.num_parameters():,}")

    tc = cfg["training"]
    train_loader = DataLoader(train_dataset, batch_size=tc["batch_size"], shuffle=True, num_workers=2)
    eval_loader = DataLoader(eval_dataset, batch_size=tc["batch_size"], num_workers=2)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"]
    )

    os.makedirs("experiments/checkpoints", exist_ok=True)
    os.makedirs("experiments/results", exist_ok=True)

    step = 0
    best_eval_loss = float("inf")

    model.train()
    pbar = tqdm(total=tc["max_steps"])

    while step < tc["max_steps"]:
        for input_ids, target_ids in train_loader:
            if step >= tc["max_steps"]:
                break

            lr = get_lr(step, tc["warmup_steps"], tc["learning_rate"])
            for g in optimizer.param_groups:
                g["lr"] = lr

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=0,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            optimizer.step()

            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")
            pbar.update(1)

            if step % tc["eval_every"] == 0:
                eval_loss = evaluate(model, eval_loader, device)
                acc = answer_accuracy(model, eval_loader, device)
                print(f"\n[step {step}] eval_loss={eval_loss:.4f}  answer_acc={acc:.4f}")
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    torch.save(
                        {
                            "step": step,
                            "model_state": model.state_dict(),
                            "eval_loss": eval_loss,
                            "answer_acc": acc,
                            "vocab_size": vocab_size,
                        },
                        "experiments/checkpoints/baseline_best.pt",
                    )

            if step % tc["save_every"] == 0:
                torch.save(
                    {"step": step, "model_state": model.state_dict(), "vocab_size": vocab_size},
                    f"experiments/checkpoints/baseline_step{step}.pt",
                )

    pbar.close()
    print(f"Training done. Best eval loss: {best_eval_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    train(args.config)
