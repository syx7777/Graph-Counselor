import json
import logging
import evaluate
import argparse
from pathlib import Path
import sys
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# evaluate.load() 离线回退时会打印 "Using the latest cached version..." warning，
# 这是正常行为；屏蔽该模块的 warning 以下，保持评测输出干净。
logging.getLogger("evaluate.loading").setLevel(logging.ERROR)

parser = argparse.ArgumentParser("")
parser.add_argument("--result_file", type=str, default="None")
parser.add_argument("--model", type=str, default="None")
parser.add_argument("--judge_model", type=str, default="qwen3-8b",
                    help="本地 vLLM 服务暴露的裁判模型名。")
parser.add_argument("--judge_base_url", type=str, default="http://127.0.0.1:8001/v1",
                    help="本地 OpenAI-compatible 裁判服务的 /v1 地址。")
parser.add_argument("--judge_api_key", type=str, default="EMPTY",
                    help="本地 vLLM API key；默认 EMPTY。")
args = parser.parse_args()

EVAL_DIR = Path(__file__).resolve().parent


def load_local_metric(name):
    """从随仓库提交的官方 Hugging Face 指标脚本加载，完全不依赖网络或用户缓存。"""
    metric_dir = EVAL_DIR / name
    metric_script = metric_dir / f"{name}.py"
    if not metric_script.is_file():
        raise FileNotFoundError(
            f"缺少本地评测脚本：{metric_script}。"
            f"请将 Hugging Face evaluate-metric/{name} 的脚本放入 {metric_dir}/。"
        )
    return evaluate.load(str(metric_dir))

def compute_exact_match(predictions, references):
    em_metric = load_local_metric("exact_match")
    return em_metric.compute(predictions=predictions, references=references)

def compute_bleu(predictions, references):
    bleu_metric = load_local_metric("bleu")
    return bleu_metric.compute(predictions=predictions, references=references)

def compute_rouge(predictions, references):
    rouge_metric = load_local_metric("rouge")
    return rouge_metric.compute(predictions=predictions, references=references)

def local_judge_chat(base_url, model, messages):
    """调用本地 vLLM 的 OpenAI-compatible Chat Completions 接口。"""
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model, "messages": messages, "temperature": 0.01,
        "chat_template_kwargs": {"enable_thinking": False},
        "top_p": 1.0, "max_tokens": 32, "stream": False,
    }).encode("utf-8")
    request = urllib.request.Request(
        endpoint, data=payload, headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + args.judge_api_key,
        }, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.load(response)
    except urllib.error.URLError as error:
        raise RuntimeError(f"本地裁判模型请求失败 ({endpoint}): {error}") from error
    try:
        return body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"本地裁判模型返回格式异常: {body!r}") from error


def GPT4score(predictions, references, questions):
    judge_model = args.judge_model
    chat = lambda messages: local_judge_chat(args.judge_base_url, judge_model, messages)
    # eval_prompt = "Please help me judge if the ground truth is inside the model prediction with string match.\nModel prediction: {} \nGround truth: {}. \nPlease answer Yes or No."
    eval_prompt = "Question:{} \nModel prediction: {} \nGround truth: {}. \nPlease help me judge if the model prediction is correct or not given the question and ground truth answer. Please use one word (Yes or No) to answer. Do not explain."
    
    res = []
    unparsable = []
    for pred, ref, question in zip(predictions, references, questions):
        x = eval_prompt.format(question, pred, ref)
        GPT_score = chat([
            {"role": "system", "content": "You are a generative language model evaluator."},
            {"role": "user", "content": x},
        ])
        # 原实现要求裁判严格输出 'Yes'/'No'，否则 assert 失败后 embed() 掉进交互 shell，
        # 会把整批评测挂死。这里做归一化，仍无法判定的按 0 计并出声告警而不静默吞掉。
        verdict = (GPT_score or '').strip().strip('.').lower()
        if verdict.startswith('yes'):
            res.append(1)
        elif verdict.startswith('no'):
            res.append(0)
        else:
            print(f"[warn] unparsable judge output {GPT_score!r}, counted as incorrect")
            unparsable.append(GPT_score)
            res.append(0)
    if unparsable:
        print(f"[warn] {len(unparsable)}/{len(res)} judge outputs were unparsable")
    return sum(res) / len(res)

def read_json(file):
    results = []
    preds = []
    gts = []
    questions = []
    with open(file) as f:
        readin = f.readlines()
        for line in readin:
            tmp = json.loads(line)
            results.append(tmp)
            preds.append(tmp['model_answer'])
            gts.append(tmp['gt_answer'])
            questions.append(tmp['question'])
    return results, preds, gts, questions

results, preds, gts, questions = read_json(args.result_file)
preds = [pred if pred != None else '' for pred in preds]
em_score = compute_exact_match(preds, gts)
bleu_score = compute_bleu(preds, gts)
rouge_score = compute_rouge(preds, gts)
gpt4_score = GPT4score(preds, gts, questions)

print(f"{args.model} || EM: {em_score['exact_match']} | Bleu: {bleu_score['bleu']} | Rouge1: {rouge_score['rouge1']} | Rouge2: {rouge_score['rouge2']} | RougeL: {rouge_score['rougeL']} | RougeLSum: {rouge_score['rougeLsum']} | JudgeScore({args.judge_model}): {gpt4_score}")
