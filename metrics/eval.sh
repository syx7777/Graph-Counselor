#!/usr/bin/env bash
set -euo pipefail

# 自动发现 $RESULT_ROOT 下所有 results*.jsonl 并逐个评测。
# 原脚本把 12 组作者本机的 /shared/data3/bowenj4/... 路径硬编码在里面，在别的机器上
# 必然 FileNotFoundError；改成扫描实际产出，评测集合就永远和 run_*.sh 的输出对齐。
EVAL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$EVAL_ROOT/.." && pwd)
# 与 LLM/scripts, GPT/scripts, Graph-CoT/scripts 下各 run_*.sh 的 RESULT_ROOT 保持一致
RESULT_ROOT=${RESULT_ROOT:-$REPO_ROOT/results}
PYTHON=${PYTHON:-python}

# 评测指标脚本已预置到本地缓存；强制离线，避免内网机器反复尝试连接 HF/GitHub 而超时。
export HF_EVALUATE_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 所有裁判调用都走同机 vLLM 的 Qwen3-8B，不读取 .conf 或 Sampool。
JUDGE_MODEL=${JUDGE_MODEL:-qwen3-8b}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-http://127.0.0.1:8001/v1}
JUDGE_API_KEY=${JUDGE_API_KEY:-EMPTY}
JUDGE_ARGS=(--judge_model "$JUDGE_MODEL" --judge_base_url "$JUDGE_BASE_URL" --judge_api_key "$JUDGE_API_KEY")

# 只评测路径中包含该片段的结果，例如 EVAL_FILTER=Graph-CoT 或 EVAL_FILTER=dblp。
EVAL_FILTER=${EVAL_FILTER:-}

if [ ! -d "$RESULT_ROOT" ]; then
    echo "RESULT_ROOT not found: $RESULT_ROOT" >&2
    echo "先用 LLM/scripts, GPT/scripts, Graph-CoT/scripts 下的 run_*.sh 生成结果。" >&2
    exit 1
fi

RESULT_FILES=()
while IFS= read -r f; do
    RESULT_FILES+=("$f")
done < <(find "$RESULT_ROOT" -type d -name '.ipynb_checkpoints' -prune -o -type f -name 'results*.jsonl' -print | sort)

if [ ${#RESULT_FILES[@]} -eq 0 ]; then
    echo "no results*.jsonl found under $RESULT_ROOT" >&2
    exit 1
fi

FAILED=()
EVALUATED=0

for RESULT_FILE in "${RESULT_FILES[@]}"; do
    REL=${RESULT_FILE#"$RESULT_ROOT"/}
    if [ -n "$EVAL_FILTER" ] && [[ $REL != *"$EVAL_FILTER"* ]]; then
        continue
    fi

    # 目录层级 <runner>/<模型>/<数据集> 直接拼成标签：Graph-CoT-gpt-4o-mini-dblp；
    # RAG 结果的文件名带 hop 后缀（results_0.jsonl），补成 ...-hop0 以便区分。
    FILE_NAME=${REL##*/}
    MODEL_LABEL=${REL%/*}
    MODEL_LABEL=${MODEL_LABEL//\//-}
    HOP=${FILE_NAME%.jsonl}
    HOP=${HOP#results}
    if [ -n "$HOP" ]; then
        MODEL_LABEL=$MODEL_LABEL-hop${HOP#_}
    fi

    echo "=== evaluating $REL"
    # 单个结果文件评测失败（脏数据 / 裁判模型报错）不应终止整批，记下来最后汇总。
    # "${JUDGE_ARGS[@]+"${JUDGE_ARGS[@]}"}"：空数组展开在 set -u 下会报 unbound variable（bash 4.3 以前）。
    if "$PYTHON" "$EVAL_ROOT/eval.py" --result_file "$RESULT_FILE" \
                        --model "$MODEL_LABEL" \
                        "${JUDGE_ARGS[@]+"${JUDGE_ARGS[@]}"}"; then
        EVALUATED=$((EVALUATED + 1))
    else
        FAILED+=("$REL")
    fi
done

if [ "$EVALUATED" -eq 0 ] && [ ${#FAILED[@]} -eq 0 ]; then
    echo "EVAL_FILTER='$EVAL_FILTER' matched nothing under $RESULT_ROOT" >&2
    exit 1
fi

echo "=== done: $EVALUATED evaluated, ${#FAILED[@]} failed"
if [ ${#FAILED[@]} -gt 0 ]; then
    printf 'failed: %s\n' "${FAILED[@]}" >&2
    exit 1
fi

# RESULT_ROOT="/graph/suyongxin/baselines/think-with-graph/results/ours-r19-full" \
# JUDGE_MODEL="qwen3-8b" \
# JUDGE_BASE_URL="http://127.0.0.1:8001/v1" \
# bash /graph/suyongxin/baselines/think-with-graph/eval/eval.sh