#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /graph/suyongxin/miniconda3/etc/profile.d/conda.sh
conda activate graphcot

# Keep local embedding and retrieval on physical GPU 0 only.
# The Qwen3-8B service is external and is not affected by this setting.
export CUDA_VISIBLE_DEVICES=0

# Reuse the existing Qwen3-8B vLLM service; do not start another server.
MODEL_PATH=/graph/suyongxin/GraphOS/cache/modelscope/models/Qwen--Qwen3-8B/snapshots/master
MODEL_NAME=Qwen3-8B
VLLM_HOST=127.0.0.1
VLLM_PORT=8001
VLLM_API_URL=http://$VLLM_HOST:$VLLM_PORT/v1
LOG_DIR=/graph/suyongxin/baselines/Graph-Counselor/log
mkdir -p "$LOG_DIR"

if ! curl -sf $VLLM_API_URL/models >/dev/null; then
    echo "Existing Qwen3-8B vLLM is unavailable at $VLLM_API_URL"
    exit 1
fi
echo "Using existing Qwen3-8B vLLM at $VLLM_API_URL"

OPENAI_KEY="not-real"
QIANFAN_AK="not-real"
QIANFAN_SK="not-real"

max_steps=10      # max iteration
max_reflect=2     # max reflection times
llm_way=vllm      # [vllm, transformer] use vllm or transformer to generate response
judge_correct=llm #llm groundtruth   # use llm or groundtruth to judge
 
GPT_version=$MODEL_PATH
EVAL_GPT_version=$MODEL_NAME
SIMPLE_MODEL_NAME=$MODEL_NAME
REFLECT_version=$MODEL_NAME
reflexion_strategy=Reflexion #None Last_attempt_and_Reflexion Last_attempt Reflexion
prompt=multiple              #base short_multiple multiple
compound_strategy=plan_compound       #None compound plan_compound plan

for DATASET in dblp #amazon legal biomedical goodreads dblp maple
do
    if [ "$DATASET" != "maple" ]; then
        DATA_PATH=../../data/processed_data/$DATASET
        SAVE_FILE=../results/$SIMPLE_MODEL_NAME/$compound_strategy/$judge_correct-$reflexion_strategy/$prompt/$max_reflect-$DATASET/results.jsonl
        SAVE_FILE_NOREFLECT=../results/$SIMPLE_MODEL_NAME/$compound_strategy/$judge_correct-$reflexion_strategy/$prompt/$max_reflect-$DATASET/result_noreflect.jsonl

        python ../code/run.py --dataset $DATASET \
                    --path $DATA_PATH \
                    --save_file $SAVE_FILE \
                    --save_file_first $SAVE_FILE_NOREFLECT\
                    --llm_version $GPT_version \
                    --reflect_version $REFLECT_version \
                    --openai_api_key "$OPENAI_KEY" \
                    --qianfan_ak "$QIANFAN_AK" \
                    --qianfan_sk "$QIANFAN_SK" \
                    --max_steps $max_steps \
                    --reflexion_strategy $reflexion_strategy\
                    --max_reflect $max_reflect\
                    --llm_way $llm_way\
                    --compound_strategy $compound_strategy\
                    --eval_llm_version $EVAL_GPT_version\
                    --api_url $VLLM_API_URL/completions\
                    --api_url2 $VLLM_API_URL/completions\
                    --judge_correct $judge_correct\
                    --reflect_prompt $prompt\
                    --api_url3 $VLLM_API_URL/completions > "$LOG_DIR/Qwen3-8B-$DATASET-plan_reflect_llm_multiple.log"
    else
        for SUBDATASET in Biology Chemistry Materials_Science Medicine Physics #Biology Chemistry Materials_Science Medicine Physics
        do
            DATA_PATH=../../data/processed_data/maple/$SUBDATASET
            SAVE_FILE=../results/$SIMPLE_MODEL_NAME/$compound_strategy/$judge_correct-$reflexion_strategy/$prompt/$max_reflect-maple-$SUBDATASET/results.jsonl
            SAVE_FILE_NOREFLECT=../results/$SIMPLE_MODEL_NAME/$compound_strategy/$judge_correct-$reflexion_strategy/$prompt/$max_reflect-maple-$SUBDATASET/result_noreflect.jsonl

            python ../code/run.py --dataset $DATASET \
                    --path $DATA_PATH \
                    --save_file $SAVE_FILE \
                    --save_file_first $SAVE_FILE_NOREFLECT\
                    --llm_version $GPT_version \
                    --reflect_version $REFLECT_version \
                    --openai_api_key "$OPENAI_KEY" \
                    --qianfan_ak "$QIANFAN_AK" \
                    --qianfan_sk "$QIANFAN_SK" \
                    --max_steps $max_steps \
                    --reflexion_strategy $reflexion_strategy\
                    --max_reflect $max_reflect\
                    --llm_way $llm_way\
                    --compound_strategy $compound_strategy\
                    --eval_llm_version $EVAL_GPT_version\
                    --api_url $VLLM_API_URL/completions\
                    --api_url2 $VLLM_API_URL/completions\
                    --judge_correct $judge_correct\
                    --reflect_prompt $prompt\
                    --api_url3 $VLLM_API_URL/completions > "$LOG_DIR/Qwen3-8B-$DATASET-$maple-plan_reflect_llm_multiple.log"

        done
    fi
done
