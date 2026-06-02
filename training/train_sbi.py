"""
Train the SBI system: Reasoning Core + State Fingerprint + Memory.
Compared against the baseline to validate H1 and H2.
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

from sbi.system import SBISystem
from sbi.core.config import ReasoningConfig, MemoryConfig, SBIConfig
from data.babi.dataset import BabiDataset


def get_lr(step: int, warmup_steps: int, lr: float) -> float:
    if step < warmup_steps:
        return lr * step / warmup_steps
    return lr


def evaluate(system: SBISystem, loader: DataLoader, device: torch.device) -> dict:
    system.eval()
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total_answer_tokens = 0
    retrieval_hits = 0

    with torch.no_grad():
        for input_ids, target_ids in loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits, fp_numpy = system(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=0,
            )
            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=-1)
            mask = target_ids != 0
            correct += ((preds == target_ids) & mask).sum().item()
            total_answer_tokens += mask.sum().item()

            # Check memory retrieval activity
            entries = system.retrieve(fp_numpy)
            if entries:
                retrieval_hits += 1

    system.train()
    return {
        "eval_loss": total_loss / max(1, n_batches),
        "answer_acc": correct / max(1, total_answer_tokens),
        "retrieval_rate": retrieval_hits / max(1, n_batches),
        **system.memory_stats(),
    }


def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dc = cfg["data"]
    print("Loading bAbI tasks:", dc["task_ids"])

    train_dataset = BabiDataset(
        task_ids=dc["task_ids"],
        split="train",
        max_seq_len=dc["max_seq_len"],
        size_per_task=dc.get("size_per_task"),
    )
    eval_dataset = BabiDataset(
        task_ids=dc["task_ids"],
        split="test",
        tokenizer=train_dataset.tokenizer,
        max_seq_len=dc["max_seq_len"],
    )

    vocab_size = train_dataset.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    mc = cfg["model"]
    mm = cfg["memory"]

    sbi_config = SBIConfig(
        reasoning=ReasoningConfig(
            vocab_size=vocab_size,
            max_seq_len=mc["max_seq_len"],
            n_layers=mc["n_layers"],
            n_heads=mc["n_heads"],
            d_model=mc["d_model"],
            d_ff=mc["d_ff"],
            dropout=mc["dropout"],
            hidden_dim=mc["d_model"],
        ),
        memory=MemoryConfig(
            fingerprint_dim=mm["fingerprint_dim"],
            max_memory_size=mm["max_memory_size"],
            hebbian_lr=mm["hebbian_lr"],
            decay_rate=mm["decay_rate"],
            min_confidence=mm["min_confidence"],
            compression_threshold=mm["compression_threshold"],
            top_k=mm["top_k"],
        ),
    )

    system = SBISystem(sbi_config).to(device)
    print(f"Parameters: {system.num_parameters():,}")

    tc = cfg["training"]
    train_loader = DataLoader(train_dataset, batch_size=tc["batch_size"], shuffle=True, num_workers=2)
    eval_loader = DataLoader(eval_dataset, batch_size=tc["batch_size"], num_workers=2)

    optimizer = torch.optim.AdamW(
        system.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"]
    )

    os.makedirs("experiments/checkpoints", exist_ok=True)
    os.makedirs("experiments/results", exist_ok=True)

    step = 0
    best_eval_loss = float("inf")

    system.train()
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

            logits, fp_numpy = system(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=0,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(system.parameters(), tc["grad_clip"])
            optimizer.step()

            # Write experience: confidence = exp(-loss) as a proxy
            confidence = float(torch.exp(-loss).item())
            if confidence >= sbi_config.memory.min_confidence:
                system.remember(
                    fingerprint=fp_numpy,
                    action=f"step_{step}",
                    outcome="correct" if confidence > 0.8 else "partial",
                    confidence=confidence,
                )

            system.step_housekeeping()

            step += 1
            stats = system.memory_stats()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                mem=stats["memory_size"],
                edges=stats["graph_edges"],
            )
            pbar.update(1)

            if step % tc["eval_every"] == 0:
                metrics = evaluate(system, eval_loader, device)
                print(
                    f"\n[step {step}] "
                    f"loss={metrics['eval_loss']:.4f}  "
                    f"acc={metrics['answer_acc']:.4f}  "
                    f"retrieval={metrics['retrieval_rate']:.2%}  "
                    f"mem={metrics['memory_size']}  "
                    f"edges={metrics['graph_edges']}"
                )
                if metrics["eval_loss"] < best_eval_loss:
                    best_eval_loss = metrics["eval_loss"]
                    torch.save(
                        {
                            "step": step,
                            "model_state": system.state_dict(),
                            "vocab_size": vocab_size,
                            **metrics,
                        },
                        "experiments/checkpoints/sbi_best.pt",
                    )

            if step % tc["save_every"] == 0:
                torch.save(
                    {"step": step, "model_state": system.state_dict(), "vocab_size": vocab_size},
                    f"experiments/checkpoints/sbi_step{step}.pt",
                )

    pbar.close()
    print(f"Training done. Best eval loss: {best_eval_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sbi_small.yaml")
    args = parser.parse_args()
    train(args.config)
