#!/bin/bash

# TODO: change some params something related to images
# this --limit-mm-per-prompt image=2 \
# or limit_mm_per_prompt={"image": 2}

export VLLM_USE_V1=1
# Model and environment settings
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL="/home/vladimir_albrekht/projects/gemma_3_avt_zfg/models/gemma-3-12b-it"

export WHISPER_MODEL_PATH="$MODEL"
MODEL_SERVED_NAME="kita" 
PORT=6655
HOST="0.0.0.0"
SEED=0

# vLLM configuration parameters
GPU_MEMORY_UTILIZATION=0.90 # 80 is fine
TENSOR_PARALLEL_SIZE=4 # changable
DTYPE="bfloat16"
MAX_NUM_BATCHED_TOKENS=32768 # 32768 vs 4096
MAX_MODEL_LEN=4096
KV_CACHE_DTYPE="auto"
BLOCK_SIZE=32 
SWAP_SPACE=0
MAX_NUM_SEQS=5

# Params
# This probably --chat-template param will be necessary later when Batyr will provide new model.
# --chat-template "/scratch/vladimir_albrekht/projects/21_july_vllm_intergration/vllm/1_vladimir_utils/utils/assets/chat_template.tmpl" \

# ? 
# --disable-mm-preprocessor-cache \

# not helpder:
# --enforce-eager \

CMD="vllm serve $MODEL \
  --tokenizer "$MODEL" \
  --host $HOST \
  --port $PORT \
  --served-model-name $MODEL_SERVED_NAME \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
  --max-model-len $MAX_MODEL_LEN \
  --trust-remote-code \
  --dtype $DTYPE \
  --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
  --swap-space $SWAP_SPACE \
  --block-size $BLOCK_SIZE \
  --kv-cache-dtype $KV_CACHE_DTYPE \
  --max-num-seqs $MAX_NUM_SEQS \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --seed $SEED"

# Execute the command
eval $CMD 2>&1 | grep -v -E "'WhisperTokenizer'|'Qwen2Tokenizer'|unexpected tokenization|'WhisperTokenizerFast'"
