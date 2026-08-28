#!/usr/bin/env bash

ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-4}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16669
ARG_RANK=0
MAX_STEPS="${MAX_STEPS:-300}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
SAVE_STEPS="${SAVE_STEPS:-50}"
LOG_COMPLETIONS="${LOG_COMPLETIONS:-false}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-false}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
API_REWARD_MAX_WORKERS="${API_REWARD_MAX_WORKERS:-4}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
GRPO_GENERATION_CHUNK_SIZE="${GRPO_GENERATION_CHUNK_SIZE:-2}"
GRPO_MULTIMODAL_SCORE_CHUNK_SIZE="${GRPO_MULTIMODAL_SCORE_CHUNK_SIZE:-1}"
GRPO_REFERENCE_SCORE_CHUNK_SIZE="${GRPO_REFERENCE_SCORE_CHUNK_SIZE:-2}"
GRPO_POLICY_SCORE_CHUNK_SIZE="${GRPO_POLICY_SCORE_CHUNK_SIZE:-2}"
GRPO_LOGPROB_TOKEN_CHUNK_SIZE="${GRPO_LOGPROB_TOKEN_CHUNK_SIZE:-64}"
QUIET_TRAIN_LOGS="${QUIET_TRAIN_LOGS:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
REPORT_TO="${REPORT_TO:-none}"
REWARD_FUNCS="${REWARD_FUNCS:-format accuracy context reasoning}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-0.2 0.7 0.2 0.2}"
BONUS_COEFFICIENT="${BONUS_COEFFICIENT:-0.2}"
BETA="${BETA:-0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.3}"
read -r -a REWARD_FUNC_ARGS <<< "$REWARD_FUNCS"
read -r -a REWARD_WEIGHT_ARGS <<< "$REWARD_WEIGHTS"
if [[ ${#REWARD_FUNC_ARGS[@]} -ne ${#REWARD_WEIGHT_ARGS[@]} ]]; then
  echo "ERROR: REWARD_FUNCS and REWARD_WEIGHTS must contain the same number of entries." >&2
  exit 2
fi

WORLD_SIZE=$ARG_WORLD_SIZE
NPROC_PER_NODE=$ARG_NPROC_PER_NODE
MASTER_ADDR=$ARG_MASTER_ADDR
MASTER_PORT=$ARG_MASTER_PORT
RANK=$ARG_RANK


RUN_NAME="${RUN_NAME:-cagro_stage2_paper_profile}"
DATASET_CONFIG="${DATASET_CONFIG:-data_config/stage2.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
OUTPUT_DIR="${OUTPUT_ROOT%/}/${RUN_NAME}"
mkdir -p "$OUTPUT_DIR"
export LOG_PATH="$OUTPUT_DIR/debug_log.txt"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
# Refuse to guess which accelerators are available on a shared host.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES must be set to GPUs already confirmed free." >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

if [[ "$REPORT_TO" == "wandb" ]]; then
  export WANDB_PROJECT="${WANDB_PROJECT:-cagro}"
  export WANDB_DIR="${WANDB_DIR:-$OUTPUT_DIR/wandb}"
  export WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-false}"
  mkdir -p "$WANDB_DIR"
  if ! python -c 'import wandb' >/dev/null 2>&1; then
    echo "ERROR: REPORT_TO=wandb requires wandb in the active zjb environment." >&2
    exit 2
  fi
  if [[ -z "${WANDB_API_KEY:-}" ]] && ! grep -q 'machine api.wandb.ai' "$HOME/.netrc" 2>/dev/null; then
    echo "ERROR: W&B is not authenticated. Run wandb login or export WANDB_API_KEY." >&2
    exit 2
  fi
fi

export USE_API_REWARD="${USE_API_REWARD:-1}"
export API_REWARD_MAX_WORKERS
export GRPO_GENERATION_CHUNK_SIZE
export GRPO_MULTIMODAL_SCORE_CHUNK_SIZE
export GRPO_REFERENCE_SCORE_CHUNK_SIZE
export GRPO_POLICY_SCORE_CHUNK_SIZE
export GRPO_LOGPROB_TOKEN_CHUNK_SIZE
if [[ "$QUIET_TRAIN_LOGS" == "1" ]]; then
  # Keep errors and scalar metrics, suppress repetitive library warnings.
  export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
  export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
fi

filter_training_log() {
  if [[ "$QUIET_TRAIN_LOGS" == "1" ]]; then
    sed -u -E '/^(Invalidate trace cache @ step|qwen-vl-utils using decord to read video\.|Unused or unrecognized kwargs:)/d'
  else
    cat
  fi
}

API_REWARD_REQUESTED=0
for reward_name in "${REWARD_FUNC_ARGS[@]}"; do
  if [[ "$reward_name" == "context" || "$reward_name" == "reasoning" ]]; then
    API_REWARD_REQUESTED=1
  fi
done
if [[ "$API_REWARD_REQUESTED" == "0" ]]; then
  export USE_API_REWARD=0
fi
if [[ "$API_REWARD_REQUESTED" == "1" && "$USE_API_REWARD" != "1" ]]; then
  echo "ERROR: context/reasoning rewards were requested while USE_API_REWARD is disabled." >&2
  exit 2
fi
if [[ "$USE_API_REWARD" == "1" ]]; then
  # Context and reasoning rewards are API-backed. Credentials are loaded only
  # when those rewards are explicitly requested, so a no-API run cannot make
  # accidental paid requests.
  API_ENV_FILE="${API_ENV_FILE:-$HOME/.config/humanomni/api.env}"
  if [[ -f "$API_ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$API_ENV_FILE"
    set +a
  fi
  API_BEARER_KEY="${API_KEY:-${OPENAI_API_KEY:-${DASHSCOPE_API_KEY:-}}}"
  if [[ -z "${API:-}" || -z "$API_BEARER_KEY" || "$API_BEARER_KEY" == "0" ]]; then
    echo "ERROR: API rewards require a valid endpoint and bearer key." >&2
    exit 2
  fi
fi

if [[ -z "${STAGE1:-}" ]]; then
  echo "ERROR: STAGE1 must point to the frozen RL-start/SFT checkpoint." >&2
  exit 2
fi

cp "$0" "$OUTPUT_DIR/"

export NCCL_SOCKET_TIMEOUT=3600
export NCCL_DEBUG=WARN
export DS_SKIP_CUDA_CHECK=1

torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/open_r1/grpo_qwenomni.py \
    --deepspeed ${DEEPSPEED_CONFIG:-run_scripts/zero2.json} \
    --output_dir "$OUTPUT_DIR" \
    --model_name_or_path "$STAGE1" \
    --dataset_name "$DATASET_CONFIG" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --num_generations "$NUM_GENERATIONS" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --freeze_vision_modules true \
    --logging_steps 1 \
    --learning_rate 5e-7 \
    --bf16 true \
    --fp16 false \
    --torch_dtype bfloat16 \
    --dataloader_num_workers 0 \
    --dataloader_pin_memory false \
    --dataloader_drop_last true \
    --data_seed 42 \
    --report_to "$REPORT_TO" \
    --scale_rewards true \
    --reward_funcs "${REWARD_FUNC_ARGS[@]}" \
    --reward_weights "${REWARD_WEIGHT_ARGS[@]}" \
    --ema_ref_model false \
    --use_cagro true \
    --beta "$BETA" \
    --epsilon 0.2 \
    --bonus_coefficient "$BONUS_COEFFICIENT" \
    --support_upper_bound 0.8 \
    --support_tolerance 0.05 \
    --answer_validity_threshold 0.5 \
    --context_validity_threshold 0.3 \
    --reasoning_validity_threshold 0.4 \
    --max_steps "$MAX_STEPS" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --temperature 0.7 \
    --top_p 0.9 \
    --use_audio_in_video true \
    --gradient_checkpointing true \
    --log_completions "$LOG_COMPLETIONS" \
    --attn_implementation "$ATTN_IMPLEMENTATION" \
    --max_pixels 100352 \
    --min_pixels 3136 \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --run_name "$RUN_NAME" \
    --save_steps "$SAVE_STEPS" \
    --save_only_model "$SAVE_ONLY_MODEL" \
    --save_total_limit 2 \
    2>&1 | filter_training_log | tee "$OUTPUT_DIR/train.log"

exit ${PIPESTATUS[0]}
