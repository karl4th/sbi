# Search-Based Intelligence (SBI)

> *Intelligence is defined not by the amount of knowledge inside a model, but by the ability to effectively search, evaluate, and use accumulated experience.*

## Overview

SBI is a research project exploring an alternative AI architecture where knowledge is separated from reasoning. Instead of storing everything inside model weights (like standard LLMs), the system is split into three independent components:

| Component | Responsibility | Update Method |
|-----------|---------------|---------------|
| **Reasoning Core** | How to think | Gradient descent |
| **Memory System** | What to remember | Hebbian updates |
| **Search System** | How to navigate memory | Similarity + graph search |

## Core Hypotheses

| ID | Hypothesis |
|----|-----------|
| H1 | A small model (100M) with memory matches a larger model without memory |
| H2 | Experience is more efficiently stored in memory than in parameters |
| H3 | Search over reasoning states outperforms search over raw text |
| H4 | Hebbian memory reduces clutter without backpropagation |

## Architecture

```
Input Tokens
     │
     ▼
┌─────────────────────┐
│   Reasoning Core    │  ← 100M GPT-style Transformer
│   (Transformer)     │    trained with gradient descent
└──────────┬──────────┘
           │ hidden state
           ▼
┌─────────────────────┐
│  State Fingerprint  │  ← Projects 768-dim hidden state
│      Layer          │    to 128-dim fingerprint
└──────────┬──────────┘
           │ fingerprint
           ▼
┌─────────────────────┐
│    Search Layer     │  ← FAISS similarity search
│                     │    + Hebbian graph expansion
└──────────┬──────────┘
           │ relevant experiences
           ▼
┌─────────────────────┐
│   Episodic Memory   │  ← Stores (state, action, outcome)
│  + Hebbian Graph    │    updated via Hebbian rule
└─────────────────────┘
```

## Project Structure

```
sbi/
├── src/sbi/
│   ├── core/
│   │   ├── transformer.py     # Reasoning Core (GPT-style)
│   │   └── config.py          # Configuration dataclasses
│   ├── memory/
│   │   ├── episodic.py        # Episodic memory with FAISS index
│   │   ├── hebbian.py         # Hebbian memory graph
│   │   └── compression.py     # Meta-state compression
│   ├── search/
│   │   ├── fingerprint.py     # State Fingerprint Layer
│   │   └── search_layer.py    # Full search procedure
│   └── system.py              # SBISystem — main entry point
├── data/tasks/
│   └── generator.py           # Synthetic reasoning task generator
├── training/
│   ├── train_baseline.py      # Train transformer without memory
│   ├── train_sbi.py           # Train full SBI system
│   └── evaluate.py            # Compare baseline vs SBI
├── configs/
│   ├── baseline.yaml          # Baseline config (~100M params)
│   └── sbi_small.yaml         # SBI config (~100M params + memory)
├── notebooks/
│   └── 01_sbi_colab.ipynb     # Full experiment notebook for Colab
└── experiments/
    ├── checkpoints/           # Saved model weights
    └── results/               # Evaluation plots and metrics
```

## Quick Start

### Local

```bash
git clone https://github.com/karl4th/sbi.git
cd sbi
pip install -r requirements.txt

# Train baseline
python training/train_baseline.py --config configs/baseline.yaml

# Train SBI
python training/train_sbi.py --config configs/sbi_small.yaml

# Compare
python training/evaluate.py
```

### Google Colab

Open `notebooks/01_sbi_colab.ipynb` in Colab with GPU (A100 recommended).

## Experiment Design

**Control:** 100M Transformer, no memory, no search

**Experimental:** Same 100M Transformer + State Fingerprint + Episodic Memory + Hebbian Graph + Search Layer

**Tasks:** Synthetic reasoning (no world knowledge required)
- Logic chains — multi-step deductive reasoning
- Analogies — structural pattern completion
- Planning sequences — reach goal through minimal operations

**Metrics:**
- Task accuracy (primary)
- Eval loss
- Memory retrieval rate
- Memory growth and compression ratio

## Success / Failure Criteria

**Project succeeds if:**
1. SBI outperforms baseline on reasoning tasks
2. Performance improves as memory accumulates
3. Memory stays bounded through compression
4. Similar problems are solved faster over time

**Project fails if:**
1. Memory retrieval rate stays near zero
2. Retrieved memories do not improve outcomes
3. Random retrieval performs equally well
4. Memory grows unboundedly

## Dependencies

- PyTorch ≥ 2.0
- FAISS (CPU or GPU)
- NetworkX
- scikit-learn (for Meta-State compression)

## Research Context

This project is part of an ongoing research initiative at **Manifestro** (Central Asia) exploring alternative neural architectures. The long-term vision:

```
Current AI:    Knowledge → Parameters
Target:        Reasoning → Parameters
               Experience → Memory
               Intelligence → Search
```

---

*Version 0.1 — Proof of Concept*
