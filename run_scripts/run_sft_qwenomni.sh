ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-2}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16666
ARG_RANK=0


export VIDEO_READER_BACKEND=decord          # 优先使用 decord（更稳定）
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# 强制 decord + 限制并发
export NUM_THREADS=4

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

STAGE1="/data0/data/HumanOmniV2-main-smoke/src/open-r1-multimodal/output/qwenomni-sft-3b-text-smoke"

RUN_NAME="qwenomni-sft"
export CUDA_VISIBLE_DEVICES=0,1
export LOG_PATH="./debug_log_$RUN_NAME.txt"

export PYTHONPATH="$PYTHONPATH:./src"
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200
# 解决 librosa 警告问题
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/



mkdir -p output/$RUN_NAME/
cp $0  output/$RUN_NAME

torchrun  --nproc_per_node=$NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/open_r1/sft.py \
    --deepspeed run_scripts/zero3_offload.json \
    --output_dir output/$RUN_NAME \
    --trust_remote_code true \
    --model_name_or_path /data1/pretrained/Qwen2___5-Omni-7B-Thinker \
     --dataset_name data_config/stage1.yaml \
    --freeze_vision_modules true \
    --use_audio_in_video true \
    --per_device_train_batch_size 1\
    --gradient_accumulation_steps 8\
    --max_steps -1 \
    --num_train_epochs 3 \
    --ignore_data_skip true \
    --dataloader_num_workers 0 \
    --disable_tqdm false \
    --logging_steps 1 \
    --dataloader_drop_last true \
    --learning_rate 1.0e-5 \
    --bf16 true \
    --data_seed 42 \
    --report_to none \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --ddp_find_unused_parameters true \
    --optim adamw_torch  \
    --attn_implementation eager \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --log_level info \
    --save_only_model true 2>&1 | tee output/$RUN_NAME/train.log

