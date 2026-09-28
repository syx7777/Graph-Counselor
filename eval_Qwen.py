import json
import evaluate
import argparse
from openai import OpenAI

from IPython import embed
from transformers import pipeline, AutoTokenizer, AutoConfig
import torch

parser = argparse.ArgumentParser("")
parser.add_argument("--result_file", type=str, default="None")
parser.add_argument("--model", type=str, default="None")
parser.add_argument("--openai_key", type=str, default="None")
parser.add_argument("--api_url", type=str, default="http://127.0.0.1:8001/v1")
parser.add_argument("--dataset", type=str, default="None")
parser.add_argument("--judge_model", type=str, default="qwen3-8b")
args = parser.parse_args()

def compute_exact_match(predictions, references):
    em_metric = evaluate.load('metrics/exact_match')
    return em_metric.compute(predictions=predictions, references=references)

def compute_bleu(predictions, references):
    bleu_metric = evaluate.load('metrics/bleu')
    return bleu_metric.compute(predictions=predictions, references=references)

def compute_rouge(predictions, references):
    rouge_metric = evaluate.load('metrics/rouge')
    return rouge_metric.compute(predictions=predictions, references=references)

def GPT4score(predictions, references, questions):
    client = OpenAI(api_key=args.openai_key)
    eval_prompt = "Question:{} \nModel prediction: {} \nGround truth: {}. \nPlease help me judge if the model prediction is correct or not given the question and ground truth answer. Please use one word (Yes or No) to answer. Do not explain."
    
    res = []
    for pred, ref, question in zip(predictions, references, questions):
        x = eval_prompt.format(question, pred, ref)
        response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a generative language model evaluator."},
                    {"role": "user", "content": x},
                ],
                temperature=0.01,
                top_p=1.0,
            )
        GPT_score = response.choices[0].message.content
        try:
            assert GPT_score in ['Yes', 'No']
        except:
            embed()
        if GPT_score == 'Yes':
            res.append(1)
        else:
            res.append(0)
    return sum(res) / len(res)


def QWenscore(predictions, references, questions):
    """Judge with the already deployed local Qwen3-8B vLLM endpoint."""
    client = OpenAI(base_url=args.api_url.rstrip('/'), api_key=args.openai_key or "EMPTY")
    eval_prompt = "Question:{} \nModel prediction: {} \nGround truth: {}. \nPlease help me judge if the model prediction is correct or not given the question and ground truth answer. Please use one word (Yes or No) to answer. Do not explain."
    scores = []
    unparsable = 0
    for pred, ref, question in zip(predictions, references, questions):
        response = client.chat.completions.create(
            model=args.judge_model,
            messages=[
                {"role": "system", "content": "You are a generative language model evaluator."},
                {"role": "user", "content": eval_prompt.format(question, pred, ref)},
            ],
            temperature=0.01,
            top_p=1.0,
            max_tokens=32,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        verdict = (response.choices[0].message.content or "").strip().strip('.').lower()
        if verdict.startswith("yes"):
            scores.append(1)
        elif verdict.startswith("no"):
            scores.append(0)
        else:
            print(f"[warn] unparsable Qwen judge response: {verdict!r}; counted as incorrect")
            unparsable += 1
            scores.append(0)
    if unparsable:
        print(f"[warn] {unparsable}/{len(scores)} Qwen judge responses were unparsable")
    return sum(scores) / len(scores) if scores else 0.0


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
# gpt4_score = GPT4score(preds, gts, questions)
llm_score = QWenscore(preds, gts, questions)

print(f"JudgeScore({args.judge_model}): {llm_score}")
print(f"{args.model}-{args.dataset} || EM: {em_score['exact_match']} | Bleu: {bleu_score['bleu']} | Rouge1: {rouge_score['rouge1']} | Rouge2: {rouge_score['rouge2']} | RougeL: {rouge_score['rougeL']} | RougeLSum: {rouge_score['rougeLsum']}")
