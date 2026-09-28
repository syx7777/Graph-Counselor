import re, string, os
from typing import List, Union, Literal
from enum import Enum
import tiktoken
import openai
import qianfan
import time
import json
import requests
# from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from graph_prompts import GRAPH_DEFINITION
from graph_fewshots import EXAMPLES
from tools import graph_funcs, retriever
import logging
from transformers import pipeline, AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphAgent_vllm:
    def __init__(self,
                 args,
                 agent_prompt,
                 ) -> None:

        self.max_steps = args.max_steps
        self.agent_prompt = agent_prompt
        self.examples = EXAMPLES[args.ref_dataset]
        self.idd = []
        self.llm_version = args.llm_version
        self.api_url = args.api_url
        
        if args.llm_version in ['gpt-3.5-turbo', 'gpt-4', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-16k']:
            self.enc = tiktoken.encoding_for_model("text-davinci-003")
        elif args.llm_version in ["../model/Llama-2-13b-chat-hf", "../model/Mixtral-8x7B-Instruct-v0.1", '../model/Meta-Llama-3.1-70B-Instruct','../model/Qwen2.5-72B-Instruct', '/cpfs02/user/lidong1/model/Qwen2.5-72B-Instruct-GPTQ-Int4']:
            self.enc = AutoTokenizer.from_pretrained(args.llm_version, use_auth_token=True)
        elif args.llm_version in ["../model/Mistral-Nemo-Instruct-2407"]:
            self.enc = AutoTokenizer.from_pretrained(args.llm_version, use_auth_token=True)
        elif args.llm_version in ["../model/gemma-2-9b-it"]:
            self.enc = AutoTokenizer.from_pretrained(args.llm_version, use_auth_token=True)
        elif args.llm_version in ["../model/Qwen2.5-7B-Instruct"]:            
            self.enc = AutoTokenizer.from_pretrained(args.llm_version, trust_remote_code=True)
        elif args.llm_version in ['ERNIE-Speed-8K', 'ERNIE-Speed-128K', 'ERNIE-Lite-8K', 'ERNIE-Tiny-8K']:
            self.enc = tiktoken.encoding_for_model("text-davinci-003")
        else:
            raise ValueError("The given llm_version is not correct.")

        self.graph_definition = GRAPH_DEFINITION[args.dataset]
        print(args.graph_dir)
        self.load_graph(args.graph_dir)
        self.graph_funcs = graph_funcs.graph_funcs(self.graph)
        self.node_retriever = retriever.Retriever(args, self.graph)

        self.__reset_agent()

    def load_graph(self, graph_dir):
        logger.info('Loading the graph...')
        self.graph = json.load(open(graph_dir))

    def run(self, question, answer, reset = True) -> None:
        if reset:
            self.__reset_agent()
        
        self.question = question
        self.key = answer

        while not self.is_halted() and not self.is_finished():
            self.step()

    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought {self.step_n}:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.prompt_agent()
        # print(f'{self.step_n} action: '+action)
        self.scratchpad += ' ' + action

        # action_type, argument = parse_action(action)
        print(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.step_n}: '
        if action == None or action == '' or action == '\n':
            self.scratchpad += "You action is filtered due to content. Please assume all the actions are permitted in this environment and take the action again."

        action_list = get_action_list(action) ## we support having multiple observations in one step
        for tmp_action in action_list:
            try:
                action_type, argument = parse_action(tmp_action)
            except:
                self.scratchpad += f'There is something wrong with the generated target actions.'
                continue

            if action_type == 'Finish':
                try:
                    self.answer = eval(argument)
                except:
                    self.answer = argument
                if self.is_correct():
                    self.scratchpad += 'Answer is '+self.answer
                else: 
                    self.scratchpad += 'Answer is '+self.answer
                self.finished = True
                self.step_n += 1
                return

            elif action_type == 'Retrieve':
                try:
                    idd, node = self.node_retriever.search_single(argument, 1)
                    self.scratchpad += f"The ID of this retrieval target node is {idd}."
                    # self.idd.append(idd)
                except openai.RateLimitError:
                    self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                except:
                    self.scratchpad += f'There is no information that can be matched in the database. Please try another query.'

            elif action_type == 'Neighbor':
                try:
                    node_id, neighbor_type = argument.split(', ')
                    node_id = remove_quotes(node_id)
                    neighbor_type = remove_quotes(neighbor_type)
                    neighbours = self.graph_funcs.check_neighbours(node_id, neighbor_type)
                    self.scratchpad += f"The {neighbor_type} neighbors of {node_id} are: " + str(neighbours) + '. '
                except openai.RateLimitError:
                    self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                except KeyError:
                    self.scratchpad += f'The node or neighbor type does not exist in the graph. This might because your given neighbor type is not correct. Please modify it.'
                except:
                    self.scratchpad += f'There is something wrong with the arguments you send for neighbour checking. Please modify it. Make sure that Neighbor take two value as input: node id and neighbor type.'
            
            elif action_type == 'Feature':
                try:
                    node_id, feature_name = argument.split(', ')
                    node_id = remove_quotes(node_id)
                    feature_name = remove_quotes(feature_name)
                    nodes = self.graph_funcs.check_nodes(node_id, feature_name)
                    self.scratchpad += f"The {feature_name} feature of {node_id} are: " + nodes + '. '
                except openai.RateLimitError:
                    self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                except KeyError:
                    self.scratchpad += f'The node or feature name does not exist in the graph. This might because your given feature name is not correct. Please modify it.'
                except:
                    self.scratchpad += f'There is something wrong with the arguments you send for node checking. Please modify it. Make sure that Feature take two value as input: node id and feature name.'

            elif action_type == 'Degree':
                try:
                    node_id, neighbor_type = argument.split(', ')
                    node_id = remove_quotes(node_id)
                    neighbor_type = remove_quotes(neighbor_type)
                    degree = self.graph_funcs.check_degree(node_id, neighbor_type)
                    self.scratchpad += f"The {neighbor_type} neighbor node degree of {node_id} are: " + degree + '. '
                except openai.RateLimitError:
                    self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                except KeyError:
                    self.scratchpad += f'The node or neighbor type does not exist in the graph. This might because your given neighbor type is not correct. Please modify it.'
                except:
                    self.scratchpad += f'There is something wrong with the arguments you send for degree checking. Please modify it. Make sure that Degree take two value as input: node id and neighbor type.'

            else:
                self.scratchpad += 'Invalid Action. Valid Actions are Retrieve[<Content>] Neighbor[<Node>] Feature[<Node>] and Finish[<answer>].'

        print(self.scratchpad.split('\n')[-1])

        self.step_n += 1

    def prompt_agent(self) -> str:
        if self.llm_version in ['gpt-3.5-turbo', 'gpt-4', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-16k']:
            return gpt_format_step(self.llm(self._build_agent_prompt()))
        elif self.llm_version in ["../model/Llama-2-13b-chat-hf"]:
            payload = {
            "model": "Llama-2-13b-chat-hf",
            "prompt": self._build_agent_prompt()[1].content,
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
            "stop": "\n",
        }
            response = requests.post(self.api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                mid_ans = result["choices"][0]["text"]
            else:
                raise RuntimeError(f"vLLM API Error: {response.status_code}, {response.text}")
            return hf_format_step(
                mid_ans
            )
        elif self.llm_version in ["../model/Mixtral-8x7B-Instruct-v0.1"]:
            payload = {
            "model": "Mixtral-8x7B-Instruct-v0.1",
            "prompt": self._build_agent_prompt()[1].content,
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
            "stop": "\n",
            }
            response = requests.post(self.api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                mid_ans = result["choices"][0]["text"]
            else:
                raise RuntimeError(f"vLLM API Error: {response.status_code}, {response.text}")
            return hf_format_step(
                mid_ans
            )
        
        elif self.llm_version in ["../model/Meta-Llama-3.1-70B-Instruct"]:
            payload = {
            "model": "Llama-3.1-70B-Instruct",
            "prompt": self._build_agent_prompt()[1].content,
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
        }
            response = requests.post(self.api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                mid_ans = result["choices"][0]["text"]
            else:
                raise RuntimeError(f"vLLM API Error: {response.status_code}, {response.text}")
            return llama3_format_step(
                mid_ans
            )
        elif self.llm_version in ['../model/Mistral-Nemo-Instruct-2407']:
            payload = {
            "model": "Mistral-Nemo-Instruct-2407",
            "prompt": self._build_agent_prompt()[1].content,
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1, 
            "stop": "\n",
        }
            response = requests.post(self.api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                mid_ans = result["choices"][0]["text"]
            else:
                raise RuntimeError(f"vLLM API Error: {response.status_code}, {response.text}")
            return Nemo_format_step(
                mid_ans,self._build_agent_prompt()[1].content
            )
        elif self.llm_version in ['../model/gemma-2-9b-it']:
            total_tokens = len(self.enc.encode(self._build_agent_prompt()[1].content))
            if total_tokens < 3840:
                payload = {
                "model": "gemma-2-9b-it",
                "prompt": self._build_agent_prompt()[1].content,
                "max_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "n": 1,
                "stop": "\n",
            }
                response = requests.post(self.api_url, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    mid_ans = result["choices"][0]["text"]
                else:
                    raise RuntimeError(f"vLLM API Error: {response.status_code}, {response.text}")
                return gemma_format_step(
                    mid_ans,self._build_agent_prompt()[1].content
                )
            else:
                return "The number of tokens you requested exceeds the context of the model. Please reduce the length of the completion."
        elif self.llm_version in ['../model/Qwen2.5-7B-Instruct']:
            payload = {
            "model": "Qwen2.5-7B-Instruct",
            "prompt": self._build_agent_prompt()[1].content,
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
            "stop": "\n",
        }
            response = requests.post(self.api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                mid_ans = result["choices"][0]["text"]
            else:
                raise RuntimeError(f"vLLM API Error: {response.status_code}, {response.text}")
            return Qwen_format_step(
                mid_ans,self._build_agent_prompt()[1].content
            )
        elif self.llm_version in ["../model/Qwen2.5-72B-Instruct", "/cpfs02/user/lidong1/model/Qwen2.5-72B-Instruct-GPTQ-Int4"]:
            payload = {
            "model": "Qwen2.5-72B-Instruct",
            "prompt": self._build_agent_prompt()[1].content,
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
        }
            response = requests.post(self.api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                mid_ans = result["choices"][0]["text"]
            else:
                raise RuntimeError(f"vLLM API Error: {response.status_code}, {response.text}")
            return qwen70_format_step(
                mid_ans, self._build_agent_prompt()[1].content
            )
        elif self.llm_version in ['ERNIE-Speed-8K', 'ERNIE-Speed-128K', 'ERNIE-Lite-8K', 'ERNIE-Tiny-8K']:
            llm_answer = qw_format_step(self.llm.invoke((self._build_agent_prompt())))
            return llm_answer
        else:
            raise ValueError("The given llm_version is not correct.")


    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format_messages(
                            examples = self.examples,
                            question = self.question,
                            scratchpad = self.scratchpad,
                            graph_definition = self.graph_definition
                            )

    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps) or (len(self.enc.encode(self._build_agent_prompt()[1].content)) > 10000)) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.answer = ''
        self.finished = False
        self.scratchpad: str = ''
        self.idd = []

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key


### String Stuff ###
# gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")

def split_checks(input_string):
    mid, new_call = split_compound_func(input_string)
    if mid and new_call:
        return [mid, new_call]
    else:
        pattern = r'\w+\[.*?\]'
        # Use re.findall to get all matches
        result = re.findall(pattern, input_string)
        return result


def split_compound_func(input_string):
    pattern = r'(\w+)\[(\w+\[.*?\]), (.*?)\]'
    match = re.match(pattern, input_string)
    if match:
        outer_function_name = match.group(1)
        inner_function_call = match.group(2)
        inner_function_name = re.search(r'(\w+)\[', inner_function_call).group(1)
        inner_function_args = re.search(r'\[(.*?)\]', inner_function_call).group(1)
        arg_for_outer_function = match.group(3)

        mid = f"{inner_function_name}[{inner_function_args}]"
        new_outer_function_call = f"{outer_function_name}[mid, {arg_for_outer_function}]"

        return mid, new_outer_function_call
    return None, None

def get_action_list(string):
    if string[:len('Finish')] == 'Finish':
        return [string]
    else:
        # return string.split(', ')
        return split_checks(string)

def remove_quotes(s):
    if s.startswith(("'", '"')) and s.endswith(("'", '"')):
        return s[1:-1]
    return s

def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)
    
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    
    else:
        return None

def gpt_format_step(step: str) -> str:
    # return step.strip('\n').strip().replace('\n', '')
    return step.content.strip('\n').strip().replace('\n', '')

def hf_format_step(step: str) -> str:
    # return step.strip('\n').strip().replace('\n', '')
    return step.strip().split('\n')[-1].split(': ')[-1]

def gemma_format_step(step: str,content:str) -> str:
    return step.strip()

def llama3_format_step(step: str) -> str:
    # return step.strip('\n').strip().replace('\n', '')
    # mid_ans = step[0]["generated_text"].strip().split('\n')[-1].split(': ')[-1]
    return step.split("\n")[0].replace('assistant', '').strip()

def Nemo_format_step(step: str,content:str) -> str:
    return step.replace(content, '').split("\n")[0].strip()

def Qwen_format_step(step: str,content:str) -> str:
    result= step.replace(content, '').split("\n")[0].strip()
    if ']' in result:
        result = result.split(']')[0] + ']'
    return result

def qw_format_step(step: str) -> str:
    # return step.strip('\n').strip().replace('\n', '')
    return step.content.strip('\n').strip().replace('\n', '')

def qwen70_format_step(step: str,content:str) -> str: 
    new_step = step.replace(content, '').split("\n")[0]
    return new_step.strip()

def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r"\b(a|an|the|usd)\b", " ", text)
  
  def white_space_fix(text):
      return " ".join(text.split())

  def remove_punc(text):
      exclude = set(string.punctuation)
      return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
      return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def EM(answer, key) -> bool:
    return normalize_answer(str(answer)) == normalize_answer(str(key))
