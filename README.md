# CAGRO for Qwen2.5-Omni

This repository implements **Confidence-Aware Group-Relative Optimization
(CAGRO)** for structured multimodal reasoning with Qwen2.5-Omni-7B-Thinker.
The canonical response protocol is:

```text
<context>observable multimodal evidence</context>
<think>reasoning grounded in that evidence</think>
<answer>final answer</answer>
```

## What is implemented

- Four bounded base signals only: format, answer, context, and reasoning.
- Serial Gate I task qualification and Gate II group-relative answer-support
  qualification, including the singleton fail-closed rule.
- Frozen RL-start reference-policy scoring over answer-body tokens only.
- Independent within-group normalization of all four streams.
- Post-normalization weights `(0.2, 0.7, 0.2, 0.2)` and semantic-span routing.
- Cold-start SFT masking that trains assistant completion tokens only.
- Dependency-free unit tests for the paper-critical parsing and gating logic.

The implementation-to-paper mapping and known reproduction limits are recorded
in [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md).

## Installation

Use Linux with CUDA for multimodal training. Install a CUDA-compatible PyTorch
build first, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Install the PyTorch build appropriate for the host CUDA version.
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

## Tests

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

## Paper-profile stage-2 training

The canonical launcher defaults to `G=4`, `epsilon=0.2`, learning rate
`5e-7`, gradient accumulation `8`, maximum completion length `1024`, and
DeepSpeed ZeRO-2. It requires the actual stage-1 checkpoint and fixed evaluator
endpoint to be supplied explicitly:

```bash
export STAGE1=/path/to/cold-start-sft-checkpoint
export API=https://your-fixed-evaluator-endpoint/v1
export API_KEY=...
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash run_scripts/run_grpo_qwenomni_stage2.sh 1 4
```

Large annotation JSON files and media are intentionally git-ignored. Before
training, place the licensed annotation file at the path referenced by
`data_config/stage2.yaml` (or provide another YAML through `DATASET_CONFIG`).

The evaluator service must keep its model/version fixed. CAGRO requests greedy
decoding (`temperature=0`) and treats timeouts, malformed outputs, and
non-finite scores as zero evidence.


