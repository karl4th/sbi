"""
Compare baseline vs SBI on bAbI test set.
Reports per-task accuracy and overall metrics.
Produces a summary table and bar chart.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from sbi.core.transformer import ReasoningCore
from sbi.core.config import ReasoningConfig
from sbi.system import SBISystem
from sbi.core.config import SBIConfig, MemoryConfig
from data.babi.dataset import BabiDataset


def answer_accuracy(logits, target_ids):
    preds = logits.argmax(dim=-1)
    mask = target_ids != 0
    correct = ((preds == target_ids) & mask).sum().item()
    total = mask.sum().item()
    return correct, total


def evaluate_on_tasks(model, task_ids, tokenizer, device, is_sbi=False):
    results = {}
    for task_id in task_ids:
        ds = BabiDataset(
            task_ids=[task_id], split="test", tokenizer=tokenizer, max_seq_len=256
        )
        loader = DataLoader(ds, batch_size=64)
        correct_total = 0
        token_total = 0
        loss_total = 0.0
        n = 0

        model.eval()
        with torch.no_grad():
            for input_ids, target_ids in loader:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)

                if is_sbi:
                    logits, _ = model(input_ids)
                else:
                    logits = model(input_ids)

                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                    ignore_index=0,
                )
                c, t = answer_accuracy(logits, target_ids)
                correct_total += c
                token_total += t
                loss_total += loss.item()
                n += 1

        results[task_id] = {
            "accuracy": correct_total / max(1, token_total),
            "loss": loss_total / max(1, n),
        }
    return results


def main(baseline_ckpt: str, sbi_ckpt: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_ids = [1, 2, 3]

    # Load tokenizer from a train dataset
    train_ds = BabiDataset(task_ids=task_ids, split="train", max_seq_len=256)
    tokenizer = train_ds.tokenizer
    vocab_size = tokenizer.vocab_size

    # Load baseline
    baseline_cfg = ReasoningConfig(vocab_size=vocab_size, hidden_dim=768)
    baseline = ReasoningCore(baseline_cfg).to(device)
    baseline.load_state_dict(torch.load(baseline_ckpt, map_location=device)["model_state"])

    # Load SBI
    sbi_config = SBIConfig(
        reasoning=ReasoningConfig(vocab_size=vocab_size, hidden_dim=768),
        memory=MemoryConfig(),
    )
    sbi = SBISystem(sbi_config).to(device)
    sbi.load_state_dict(torch.load(sbi_ckpt, map_location=device)["model_state"])

    baseline_results = evaluate_on_tasks(baseline, task_ids, tokenizer, device, is_sbi=False)
    sbi_results = evaluate_on_tasks(sbi, task_ids, tokenizer, device, is_sbi=True)

    # Print table
    print("\n── Per-Task Accuracy ──────────────────────────────────────")
    print(f"{'Task':<12} {'Baseline':>12} {'SBI':>12} {'Delta':>12}")
    print("─" * 52)
    for task_id in task_ids:
        task_name = f"bAbI-{task_id}"
        b_acc = baseline_results[task_id]["accuracy"]
        s_acc = sbi_results[task_id]["accuracy"]
        delta = s_acc - b_acc
        sign = "+" if delta >= 0 else ""
        print(f"{task_name:<12} {b_acc:>12.4f} {s_acc:>12.4f} {sign}{delta:>11.4f}")
    print("─" * 52)

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(task_ids))
    width = 0.35
    colors = {"baseline": "#4C72B0", "sbi": "#DD8452"}
    task_labels = [f"Task {t}" for t in task_ids]

    for ax, metric, title, higher_is_better in [
        (axes[0], "accuracy", "Answer Accuracy ↑", True),
        (axes[1], "loss", "Eval Loss ↓", False),
    ]:
        b_vals = [baseline_results[t][metric] for t in task_ids]
        s_vals = [sbi_results[t][metric] for t in task_ids]

        bars_b = ax.bar(x - width / 2, b_vals, width, label="Baseline", color=colors["baseline"])
        bars_s = ax.bar(x + width / 2, s_vals, width, label="SBI", color=colors["sbi"])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(task_labels)
        ax.legend()

        for bar, val in list(zip(bars_b, b_vals)) + list(zip(bars_s, s_vals)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                fontsize=8,
            )

    plt.suptitle("Baseline vs SBI — bAbI Tasks 1, 2, 3", fontsize=13)
    plt.tight_layout()
    os.makedirs("experiments/results", exist_ok=True)
    plt.savefig("experiments/results/comparison.png", dpi=150)
    print("\nPlot saved to experiments/results/comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="experiments/checkpoints/baseline_best.pt")
    parser.add_argument("--sbi", default="experiments/checkpoints/sbi_best.pt")
    args = parser.parse_args()
    main(args.baseline, args.sbi)
