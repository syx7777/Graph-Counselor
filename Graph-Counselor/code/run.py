import os
from tqdm import tqdm
import logging
import argparse
import jsonlines
import datetime
from GraphAgent import GraphAgent
from GraphReflectAgent import GraphReflectAgent
from GraphAgent_vllm import GraphAgent_vllm
from GraphReflectAgent_vllm import GraphReflectAgent_vllm
from GraphAgent_Plan_Reflect_vllm import GraphAgent_Plan_Reflect_vllm
from tools.retriever import NODE_TEXT_KEYS
from graph_prompts import graph_agent_prompt, graph_agent_prompt_zeroshot, reflect_graph_agent_prompt, reflect_graph_agent_prompt_zeroshot, graph_compound_and_plan_prompt, graph_compound_prompt
from graph_prompts import graph_reflect_prompt, graph_eval_prompt, graph_reflect_prompt_base, graph_reflect_prompt_short_multiple, reflect_graph_compound_and_plan_prompt, reflect_graph_compound_prompt
from graph_prompts import graph_compound_and_plan_reflect_prompt, graph_compound_and_plan_reflect_prompt_base, graph_compound_reflect_prompt, graph_compound_reflect_prompt_base, graph_compound_and_plan_reflect_prompt_short_multiple, graph_compound_reflect_prompt_short_multiple, graph_compound_and_plan_eval_prompt, graph_compound_eval_prompt
from IPython import embed

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

logging.basicConfig(level=logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

parser = argparse.ArgumentParser("")
parser.add_argument("--dataset", type=str, default="dblp")
parser.add_argument("--openai_api_key", type=str, default="xxx")
parser.add_argument("--qianfan_ak", type=str, default="xxx")
parser.add_argument("--qianfan_sk", type=str, default="xxx")
parser.add_argument("--path", type=str)
parser.add_argument("--save_file", type=str)
parser.add_argument("--save_file_first", type=str)
parser.add_argument("--embedder_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
parser.add_argument("--faiss_gpu", type=bool, default=False)
parser.add_argument("--embed_cache", type=bool, default=True)
parser.add_argument("--max_steps", type=int, default=15)
parser.add_argument("--zero_shot", type=bool, default=False)
parser.add_argument("--ref_dataset", type=str, default=None)

parser.add_argument("--llm_version", type=str, default="gpt-3.5-turbo")
parser.add_argument("--eval_llm_version", type=str, default="gpt-3.5-turbo")
parser.add_argument("--reflect_version", type=str, default="gpt-3.5-turbo")
parser.add_argument("--reflexion_strategy", type=str, default="None")
parser.add_argument("--max_reflect", type=str, default="1")
parser.add_argument("--llm_way", type=str, default="transformer")
parser.add_argument("--judge_correct", type=str, default="groundtruth")
parser.add_argument("--compound_strategy", type=str, default="None")
parser.add_argument("--api_url", type=str, default="http://localhost:8000/v1/completions")
parser.add_argument("--api_url2", type=str, default="http://localhost:8030/v1/completions")
parser.add_argument("--api_url3", type=str, default="http://localhost:8000/v1/completions")
parser.add_argument("--reflect_prompt", type=str, default="base")
args = parser.parse_args()

args.embed_cache_dir = args.path
args.graph_dir = os.path.join(args.path, "graph.json")
args.data_dir = os.path.join(args.path, "data.json")
# args.data_dir = os.path.join(args.path, "data_subset.json")
args.node_text_keys = NODE_TEXT_KEYS[args.dataset]
args.ref_dataset = args.dataset if not args.ref_dataset else args.ref_dataset

assert args.llm_version in ['gpt-3.5-turbo', 'gpt-4', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-16k', "/cpfs02/user/lidong1/model/Mixtral-8x7B-Instruct-v0.1", "/cpfs02/user/lidong1/model/Llama-2-13b-chat-hf", "/cpfs02/user/lidong1/model/Mistral-Nemo-Instruct-2407", '/nas/shared/ma4agi/model/Mistral-Nemo-Instruct-2407', '/cpfs02/user/lidong1/model/gemma-2-9b-it', '/nas/shared/ma4agi/model/gemma-2-9b-it', '/cpfs02/user/lidong1/model/Qwen2.5-7B-Instruct', '/cpfs02/user/lidong1/model/Meta-Llama-3.1-70B-Instruct', '/cpfs02/user/lidong1/model/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4', '/cpfs02/user/lidong1/model/Qwen2.5-72B-Instruct', '/cpfs02/user/lidong1/model/Qwen2.5-72B-Instruct-GPTQ-Int4', '/graph/suyongxin/GraphOS/cache/modelscope/models/Qwen--Qwen3-8B/snapshots/master', 'ERNIE-Speed-8K', 'ERNIE-Speed-128K', 'ERNIE-Lite-8K', 'ERNIE-Tiny-8K']
assert args.eval_llm_version in ['gpt-3.5-turbo', 'gpt-4', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-16k', "Mixtral-8x7B-Instruct-v0.1", "Llama-2-13b-chat-hf", "Mistral-Nemo-Instruct-2407", 'gemma-2-9b-it', 'Qwen2.5-7B-Instruct', 'Meta-Llama-3.1-70B-Instruct', 'Qwen2.5-72B-Instruct', 'Qwen3-8B', 'ERNIE-Speed-8K', 'ERNIE-Speed-128K', 'ERNIE-Lite-8K', 'ERNIE-Tiny-8K']
assert args.reflect_version in ['gpt-3.5-turbo', 'gpt-4', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-16k', "Mixtral-8x7B-Instruct-v0.1", "Llama-2-13b-chat-hf", "Mistral-Nemo-Instruct-2407", 'gemma-2-9b-it', 'Qwen2.5-7B-Instruct', 'Meta-Llama-3.1-70B-Instruct', 'Qwen2.5-72B-Instruct', 'Qwen3-8B', 'ERNIE-Speed-8K', 'ERNIE-Speed-128K', 'ERNIE-Lite-8K', 'ERNIE-Tiny-8K']
assert args.reflexion_strategy in ["None","Last_attempt","Reflexion","Last_attempt_and_Reflexion"]
assert args.llm_way in ["vllm","transformer"]
assert args.judge_correct in ["llm","groundtruth"]
assert args.reflect_prompt in ["base","multiple","short_multiple"]
assert args.compound_strategy in ["None", "compound", "plan_compound", "plan"]


def remove_fewshot(prompt: str) -> str:
    # prefix = prompt.split('Here are some examples:')[0]
    # suffix = prompt.split('(END OF EXAMPLES)')[1]
    prefix = prompt[-1].content.split('Here are some examples:')[0]
    suffix = prompt[-1].content.split('(END OF EXAMPLES)')[1]
    return prefix.strip('\n').strip() + '\n' +  suffix.strip('\n').strip()

def main():

    os.environ["OPENAI_API_KEY"] = args.openai_api_key
    os.environ["QIANFAN_AK"] = args.qianfan_ak
    os.environ["QIANFAN_SK"] = args.qianfan_sk
    with open(args.data_dir, 'r') as f:
        contents = []
        for item in jsonlines.Reader(f):
            contents.append(item)

    ####################################################################
    # contents = [contents[80]]
    ####################################################################

    output_file_path = args.save_file
    output_file_path_first = args.save_file_first

    parent_folder = os.path.dirname(output_file_path)
    os.makedirs(parent_folder, exist_ok=True)

    if not os.path.exists('{}/logs'.format(parent_folder)):
        os.makedirs('{}/logs'.format(parent_folder))
    logs_dir = '{}/logs'.format(parent_folder)
    
    if args.reflexion_strategy == "None" and args.compound_strategy == "None":
        agent_prompt = graph_agent_prompt if not args.zero_shot else graph_agent_prompt_zeroshot 
        if args.llm_way == "transformer":
            agent = GraphAgent(args, agent_prompt)
        else:
            agent = GraphAgent_vllm(args, agent_prompt)
    elif args.compound_strategy == "None":
        agent_prompt = reflect_graph_agent_prompt if not args.zero_shot else reflect_graph_agent_prompt_zeroshot
        reflect_prompt = graph_reflect_prompt
        reflect_prompt_base = graph_reflect_prompt_base
        reflect_prompt_short_multiple = graph_reflect_prompt_short_multiple
        eval_prompt = graph_eval_prompt
        if args.llm_way == "transformer":
            agent = GraphReflectAgent(args, agent_prompt,reflect_prompt, eval_prompt, reflect_prompt_base, reflect_prompt_base)
        else:
            agent = GraphReflectAgent_vllm(args, agent_prompt,reflect_prompt, eval_prompt, reflect_prompt_base, reflect_prompt_short_multiple)
    elif args.reflexion_strategy == "None":
        agent_prompt = graph_compound_and_plan_prompt if args.compound_strategy == "plan_compound" else graph_compound_prompt
        if args.llm_way == "transformer":
            agent = GraphAgent(args, agent_prompt)
        elif args.llm_way == "vllm":
            if args.compound_strategy == "plan_compound":
                agent = GraphAgent_vllm_plan(args, agent_prompt)
            else:
                agent = GraphAgent_vllm(args, agent_prompt)
    else:
        agent_prompt = reflect_graph_compound_and_plan_prompt if args.compound_strategy in ["plan_compound", "plan"] else reflect_graph_compound_prompt
        reflect_prompt = graph_compound_and_plan_reflect_prompt if args.compound_strategy in ["plan_compound", "plan"] else graph_compound_reflect_prompt
        reflect_prompt_base = graph_compound_and_plan_reflect_prompt_base if args.compound_strategy in ["plan_compound", "plan"] else graph_compound_reflect_prompt_base
        reflect_prompt_short_multiple = graph_compound_and_plan_reflect_prompt_short_multiple if args.compound_strategy in ["plan_compound", "plan"] else graph_compound_reflect_prompt_short_multiple
        eval_prompt = graph_compound_and_plan_eval_prompt if args.compound_strategy in ["plan_compound", "plan"] else graph_compound_eval_prompt
        if args.llm_way == "vllm":
            agent = GraphAgent_Plan_Reflect_vllm(args, agent_prompt,reflect_prompt, eval_prompt, reflect_prompt_base, reflect_prompt_short_multiple)

    unanswered_questions = []
    correct_logs = []
    halted_logs = []
    incorrect_logs = []
    generated_text = []
    generated_text_first = [] 
    correct_round1_logs = []
    correct_round2_logs = []
    correct_round3_logs = []
    correct_round4_logs = []
    correct_round5_logs = []
    for i in tqdm(range(len(contents))):
        agent.run(contents[i]['question'], contents[i]['answer'])
        print(f'Ground Truth Answer: {agent.key}')
        print('---------')
        log = f"Question: {contents[i]['question']}\n"
        log += remove_fewshot(agent._build_agent_prompt()) + f'\nCorrect answer: {agent.key}\n\n' if not args.zero_shot else agent._build_agent_prompt()[-1].content + f'\nCorrect answer: {agent.key}\n\n'
        with open(os.path.join(logs_dir, contents[i]['qid']+'.txt'), 'w') as f:
            f.write(log)

        ## summarize the logs
        if agent.is_correct():
            correct_logs.append(log)
            if args.reflexion_strategy != "None":
                round_to_logs = {
                    "1": correct_round1_logs,
                    "2": correct_round2_logs,
                    "3": correct_round3_logs,
                    "4": correct_round4_logs,
                    "5": correct_round5_logs,
                }
                round_n = agent.caculate_round_n()  
                if round_n in round_to_logs:
                    round_to_logs[round_n].append(log)
        elif agent.is_halted():
            halted_logs.append(log)
        elif agent.is_finished() and not agent.is_correct():
            incorrect_logs.append(log)
        else:
            raise ValueError('Something went wrong!')

        generated_text.append({"question": contents[i]["question"], "model_answer": agent.answer if isinstance(agent.answer, (str, int, float, list, dict, bool, type(None))) else None, "gt_answer": contents[i]['answer']})
        generated_text_first.append({"question": contents[i]["question"], "model_answer": agent.answer_first if isinstance(agent.answer_first, (str, int, float, list, dict, bool, type(None))) else None, "gt_answer": contents[i]['answer']})

    with jsonlines.open(output_file_path, 'w') as writer:
        for row in generated_text:
            writer.write(row)

    with jsonlines.open(output_file_path_first, 'w') as writer:
        for row in generated_text_first:
            writer.write(row)

    print(f'Finished Trial {len(contents)}, Correct: {len(correct_logs)}, Correct_round_1:{len(correct_round1_logs)}, Correct_round_2:{len(correct_round2_logs)}, Correct_round_3:{len(correct_round3_logs)}, Correct_round_4:{len(correct_round4_logs)}, Correct_round_5:{len(correct_round5_logs)}, Incorrect: {len(incorrect_logs)}, Halted: {len(halted_logs)}')
    print('Unanswered questions {}: {}'.format(len(unanswered_questions), unanswered_questions))

if __name__ == '__main__':
    main()