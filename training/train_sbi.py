"""
Train the SBI system with two-pass memory injection.

Key difference from baseline:
  - Pass 1 (no_grad): get query fingerprint from current input
  - Retrieve semantically relevant memories (fingerprint + answer token)
  - Pass 2 (grad): forward with injected memory context → loss → backprop
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
from data.babi.tokenizer import PAD_ID


def get_lr(step: int, warmup_steps: int, lr: float) -> float:
    if step < warmup_steps:
        return lr * step / warmup_steps
    return lr


def extract_answer_tokens(target_ids: torch.Tensor) -> list[int]:
    """Extract the first non-PAD target token for each batch item."""
    answer_tokens = []
    for row in target_ids:
        nonzero = (row != PAD_ID).nonzero(as_tuple=True)[0]
        answer_tokens.append(row[nonzero[0]].item() if len(nonzero) > 0 else -1)
    return answer_tokens


def evaluate(system: SBISystem, loader: DataLoader, device: torch.device) -> dict:
    system.eval()
    tokenizer = getattr(loader.dataset, "tokenizer", None)
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total_answer_tokens = 0
    retrieval_hits = 0
    retrieval_queries = 0
    memory_answer_correct = 0
    memory_answer_total = 0
    similarity_sum = 0.0
    wrong_examples = []

    with torch.no_grad():
        for input_ids, target_ids in loader:
            input_ids  = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits, query_fp = system(input_ids)

            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=PAD_ID,
            )
            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=-1)
            mask  = target_ids != PAD_ID
            correct             += ((preds == target_ids) & mask).sum().item()
            total_answer_tokens += mask.sum().item()

            if system.episodic_memory.size() > 0:
                true_answers = extract_answer_tokens(target_ids.cpu())
                memory_answers, similarities = system.retrieve_answer_tokens(query_fp)
                for mem_answer, true_answer, similarity in zip(
                    memory_answers, true_answers, similarities
                ):
                    retrieval_queries += 1
                    if mem_answer < 0:
                        continue
                    retrieval_hits += 1
                    memory_answer_total += int(true_answer >= 0)
                    memory_answer_correct += int(mem_answer == true_answer)
                    similarity_sum += similarity
                    if (
                        mem_answer != true_answer
                        and true_answer >= 0
                        and len(wrong_examples) < 5
                    ):
                        if tokenizer is None:
                            wrong_examples.append((mem_answer, true_answer, similarity))
                        else:
                            wrong_examples.append(
                                (
                                    tokenizer.id2word.get(mem_answer, "<unk>"),
                                    tokenizer.id2word.get(true_answer, "<unk>"),
                                    round(similarity, 4),
                                )
                            )

    system.train()
    memory_answer_acc = memory_answer_correct / max(1, memory_answer_total)
    avg_similarity = similarity_sum / max(1, retrieval_hits)
    return {
        "eval_loss":      total_loss / max(1, n_batches),
        "answer_acc":     correct / max(1, total_answer_tokens),
        "retrieval_rate": retrieval_hits / max(1, retrieval_queries),
        "memory_answer_acc": memory_answer_acc,
        "memory_avg_similarity": avg_similarity,
        "memory_wrong_examples": wrong_examples,
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
        data_dir=dc.get("data_dir"),
        allow_synthetic_fallback=dc.get("allow_synthetic_fallback", False),
    )
    eval_dataset = BabiDataset(
        task_ids=dc["task_ids"],
        split="test",
        tokenizer=train_dataset.tokenizer,
        max_seq_len=dc["max_seq_len"],
        data_dir=dc.get("data_dir"),
        allow_synthetic_fallback=dc.get("allow_synthetic_fallback", False),
    )

    vocab_size = train_dataset.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    mc = cfg["model"]
    mm = cfg["memory"]
    tc = cfg["training"]

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
            graph_top_k=mm.get("graph_top_k", 3),
            memory_logit_bias=mm.get("memory_logit_bias", 0.0),
            answer_marker_token_id=mm.get("answer_marker_token_id", 4),
            use_learned_fingerprint=mm.get("use_learned_fingerprint", True),
        ),
    )

    system = SBISystem(sbi_config).to(device)
    print(f"Parameters: {system.num_parameters():,}")

    # Optional baseline initialization for A/B experiments.
    baseline_ckpt = tc.get("baseline_checkpoint", "experiments/checkpoints/baseline_best.pt")
    if tc.get("init_from_baseline", True) and os.path.exists(baseline_ckpt):
        ckpt = torch.load(baseline_ckpt, map_location=device)
        baseline_state = ckpt["model_state"]
        # Load only the reasoning core weights (keys without "reasoning_core." prefix)
        core_state = {
            k.replace("reasoning_core.", ""): v
            for k, v in baseline_state.items()
        }
        missing, unexpected = system.reasoning_core.load_state_dict(core_state, strict=False)
        print(f"Loaded baseline into reasoning core. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    elif tc.get("init_from_baseline", True):
        print(f"WARNING: baseline checkpoint not found at {baseline_ckpt}. Training from scratch.")
    else:
        print("Training SBI from scratch; baseline initialization disabled.")

    train_loader = DataLoader(train_dataset, batch_size=tc["batch_size"], shuffle=True, num_workers=2)
    eval_loader  = DataLoader(eval_dataset,  batch_size=tc["batch_size"], num_workers=2)

    optimizer = torch.optim.AdamW(
        system.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"]
    )

    os.makedirs("experiments/checkpoints", exist_ok=True)
    os.makedirs("experiments/results", exist_ok=True)
    experiment_name = cfg.get("logging", {}).get("experiment_name", "sbi")
    log_every = cfg.get("logging", {}).get("log_every", 50)

    step = 0
    best_eval_loss = float("inf")
    last_eval_memory_answer_acc = 0.0
    last_eval_memory_avg_similarity = 0.0
    last_train_memory_answer_acc = 0.0
    last_train_memory_avg_similarity = 0.0

    system.train()
    pbar = tqdm(total=tc["max_steps"])

    while step < tc["max_steps"]:
        for input_ids, target_ids in train_loader:
            if step >= tc["max_steps"]:
                break

            lr = get_lr(step, tc["warmup_steps"], tc["learning_rate"])
            for g in optimizer.param_groups:
                g["lr"] = lr

            input_ids  = input_ids.to(device)
            target_ids = target_ids.to(device)
            answer_tokens = extract_answer_tokens(target_ids.cpu())

            # Two-pass forward (pass 1 inside system.forward is no_grad)
            logits, query_fp_np = system(input_ids)

            if system.episodic_memory.size() > 0 and step % log_every == 0:
                memory_answers, similarities = system.retrieve_answer_tokens(query_fp_np)
                hits = [
                    (mem_answer, true_answer, similarity)
                    for mem_answer, true_answer, similarity in zip(
                        memory_answers, answer_tokens, similarities
                    )
                    if mem_answer >= 0 and true_answer >= 0
                ]
                if hits:
                    last_train_memory_answer_acc = sum(
                        int(mem_answer == true_answer)
                        for mem_answer, true_answer, _similarity in hits
                    ) / len(hits)
                    last_train_memory_avg_similarity = sum(
                        similarity for _mem_answer, _true_answer, similarity in hits
                    ) / len(hits)

            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=PAD_ID,
            )
            if not torch.isfinite(loss):
                stats = system.memory_stats()
                raise RuntimeError(
                    "Non-finite SBI loss detected. "
                    f"step={step} loss={loss.item()} "
                    f"memory_size={stats['memory_size']} "
                    f"graph_edges={stats['graph_edges']} "
                    f"memory_logit_bias={system.config.memory.memory_logit_bias}"
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(system.parameters(), tc["grad_clip"])
            optimizer.step()

            # Store supervised experiences for every batch item. The answer token
            # comes from the dataset, so it should not be gated by current model loss.
            system.remember_batch(
                query_fp_np=query_fp_np,
                answer_tokens=answer_tokens,
            )

            system.step_housekeeping()

            step += 1
            stats = system.memory_stats()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                mem=stats["memory_size"],
                edges=stats["graph_edges"],
                train_mem=f"{last_train_memory_answer_acc:.2%}",
                eval_mem=f"{last_eval_memory_answer_acc:.2%}",
                sim=f"{last_train_memory_avg_similarity:.2f}",
            )
            pbar.update(1)

            if step % tc["eval_every"] == 0:
                metrics = evaluate(system, eval_loader, device)
                last_eval_memory_answer_acc = metrics["memory_answer_acc"]
                last_eval_memory_avg_similarity = metrics["memory_avg_similarity"]
                print(
                    f"\n[step {step}] "
                    f"loss={metrics['eval_loss']:.4f}  "
                    f"acc={metrics['answer_acc']:.4f}  "
                    f"retrieval={metrics['retrieval_rate']:.2%}  "
                    f"mem_ans_acc={metrics['memory_answer_acc']:.2%}  "
                    f"mem_sim={metrics['memory_avg_similarity']:.3f}  "
                    f"mem={metrics['memory_size']}  "
                    f"edges={metrics['graph_edges']}"
                )
                if metrics["memory_wrong_examples"]:
                    print("  wrong memory examples (pred,true,sim):", metrics["memory_wrong_examples"])
                if metrics["eval_loss"] < best_eval_loss:
                    best_eval_loss = metrics["eval_loss"]
                    torch.save(
                        {
                            "step": step,
                            "model_state": system.state_dict(),
                            "vocab_size": vocab_size,
                            **metrics,
                        },
                        f"experiments/checkpoints/{experiment_name}_best.pt",
                    )

            if step % tc["save_every"] == 0:
                torch.save(
                    {"step": step, "model_state": system.state_dict(), "vocab_size": vocab_size},
                    f"experiments/checkpoints/{experiment_name}_step{step}.pt",
                )

    pbar.close()
    print(f"Training done. Best eval loss: {best_eval_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sbi_small.yaml")
    args = parser.parse_args()
    train(args.config)
