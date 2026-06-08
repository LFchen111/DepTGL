# DepTGL

This repository provides the implementation of **DepTGL**, a distributed training framework for memory-based Temporal Graph Neural Networks (M-TGNNs).

DepTGL supports distributed temporal memory management, dependency-aware event replay, adaptive cache synchronization, and load-aware auxiliary replay pruning.

## Repository Structure

```text
.
├── distributed/                  # Distributed memory, partitioning, and dependency utilities
├── evaluation/                   # Evaluation code
├── model/                        # TGN model
├── modules/                      # Memory, message, aggregation, and embedding modules
└── train_self_supervised.py      # Main training script
```

## Requirements

The code is based on Python and PyTorch.

```text
Python >= 3.8
PyTorch >= 1.12
NumPy
Pandas
scikit-learn
CUDA-enabled GPU environment
```

Install basic dependencies:

```bash
pip install numpy pandas scikit-learn torch
```

Please install a PyTorch version compatible with your CUDA environment.

## Data

Datasets should be placed under `./data/` in the TGN-style processed format. For example:

```text
data/
├── ml_wikipedia.csv
├── ml_wikipedia.npy
└── ml_wikipedia_node.npy
```

## Quick Start

Single-GPU training:

```bash
python train_self_supervised.py \
  --data wikipedia \
  --use_memory \
  --bs 80 \
  --n_epoch 10 \
  --n_layer 2 \
  --n_degree 5 \
  --lr 0.0001 \
  --memory_dim 172 \
  --message_dim 172 \
  --time_dim 172
```

## Multi-node Training

The following example uses **3 machines**, with **1 GPU process per machine**.

Set a different `NODE_RANK` on each machine:

```bash
# Machine 0
NODE_RANK=0

# Machine 1
NODE_RANK=1

# Machine 2
NODE_RANK=2
```

Launch command:

```bash
torchrun \
  --nnodes=3 \
  --nproc_per_node=1 \
  --node_rank=${NODE_RANK} \
  --master_addr=<MASTER_ADDR> \
  --master_port=29514 \
  train_self_supervised.py \
  --use_memory \
  --distributed \
  --world_size 3 \
  --backend nccl \
  --start_gpu 0 \
  --data reddit \
  --bs 500 \
  --n_epoch 10 \
  --n_layer 2 \
  --n_degree 10 \
  --lr 0.0001 \
  --memory_dim 172 \
  --message_dim 172 \
  --time_dim 172 \
  --sync_every 3 \
  --grad_ema_threshold 1.6 \
  --time_ema_threshold 0.5
```

Replace `<MASTER_ADDR>` with the IP address of the master machine.

## Main Arguments

Basic training arguments:

```text
--data                 Dataset name
--bs                   Batch size
--n_epoch              Number of epochs
--n_degree             Number of temporal neighbors
--n_layer              Number of TGNN layers
--lr                   Learning rate
--use_memory           Enable memory-based training
--memory_dim           Memory dimension
--message_dim          Message dimension
--time_dim             Time encoding dimension
```

Distributed arguments:

```text
--distributed          Enable distributed training
--world_size           Number of distributed processes
--backend              Distributed backend: gloo or nccl
--start_gpu            Starting GPU index
--partition_strategy   Node partition strategy: hash or range
--sync_every           Cache synchronization interval
```

Runtime control arguments:

```text
--grad_ema_threshold   Threshold for adaptive cache synchronization
--time_ema_threshold   Threshold for load-aware replay pruning
```

## Ablation and Variants

Useful switches:

```text
--disable_dynamic_sync       Disable adaptive cache synchronization
--disable_load_aware_drop    Disable load-aware replay pruning
--disable_prefetch           Disable dependency prefetching / replay
--random_ablation            Use random synchronization skipping and random pruning
```

## Outputs

The script writes outputs to:

```text
results/              # Pickled result files
saved_models/         # Saved model weights
saved_checkpoints/    # Intermediate checkpoints
log/                  # Runtime logs
```
