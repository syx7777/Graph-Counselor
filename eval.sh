#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESULT_ROOT=${RESULT_ROOT:-$REPO_ROOT/Graph-Counselor/results}
PYTHON=${PYTHON:-/graph/suyongxin/miniconda3/envs/graphcot/bin/python}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-http://127.0.0.1:8001/v1}
JUDGE_MODEL=${JUDGE_MODEL:-qwen3-8b}
JUDGE_API_KEY=${JUDGE_API_KEY:-EMPTY}
EVAL_FILTER=${EVAL_FILTER:-}

# Metrics are vendored in this repository; avoid any remote Hugging Face lookup.
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

if [ ! -x "$PYTHON" ]; then
    echo "Python executable not found: $PYTHON" >&2
    exit 1
fi
if [ ! -d "$RESULT_ROOT" ]; then
    echo "Result root not found: $RESULT_ROOT" >&2
    exit 1
fi

mapfile -t RESULT_FILES < <(find "$RESULT_ROOT" -type f -name 'results.jsonl' -print | sort)
if [ ${#RESULT_FILES[@]} -eq 0 ]; then
    echo "No results.jsonl found under $RESULT_ROOT" >&2
    exit 1
fi

EVALUATED=0
for RESULT_FILE in "${RESULT_FILES[@]}"; do
    REL=${RESULT_FILE#"$RESULT_ROOT"/}
    if [ -n "$EVAL_FILTER" ] && [[ $REL != *"$EVAL_FILTER"* ]]; then
        continue
    fi
    RUN_DIR=$(basename "$(dirname "$RESULT_FILE")")
    DATASET=${RUN_DIR#2-}
    MODEL_LABEL=${REL%/results.jsonl}
    MODEL_LABEL=${MODEL_LABEL//\//-}
    echo "=== evaluating $REL with local $JUDGE_MODEL"
    "$PYTHON" "$REPO_ROOT/eval_Qwen.py"         --result_file "$RESULT_FILE"         --dataset "$DATASET"         --model "$MODEL_LABEL"         --openai_key "$JUDGE_API_KEY"         --api_url "$JUDGE_BASE_URL"         --judge_model "$JUDGE_MODEL"
    EVALUATED=$((EVALUATED + 1))
done

if [ "$EVALUATED" -eq 0 ]; then
    echo "EVAL_FILTER='$EVAL_FILTER' matched no result files" >&2
    exit 1
fi
echo "=== done: $EVALUATED result file(s) evaluated with local $JUDGE_MODEL"
