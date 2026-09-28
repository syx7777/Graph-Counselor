import re, string, os
from typing import List, Union, Literal
from enum import Enum
import tiktoken
import openai
import qianfan
import time
import json
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

class GraphAgent:
    def __init__(self,
                 args,
                 agent_prompt,
                 ) -> None:

        self.max_steps = args.max_steps
        self.agent_prompt = agent_prompt
        self.examples = EXAMPLES[args.ref_dataset]
        self.idd = []

        self.llm_version = args.llm_version
        if args.llm_version in ['gpt-3.5-turbo', 'gpt-4', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-16k']:
            self.llm = ChatOpenAI(temperature=0,
                                max_tokens=300,
                                model_name=args.llm_version,
                                model_kwargs={"stop": "\n"},
                                )
            self.enc = tiktoken.encoding_for_model("text-davinci-003")
        elif args.llm_version in ["../model/Llama-2-13b-chat-hf", "../model/Mixtral-8x7B-Instruct-v0.1", '../model/Meta-Llama-3.1-70B-Instruct']:
            self.config = AutoConfig.from_pretrained(args.llm_version, use_auth_token=True)
            self.enc = AutoTokenizer.from_pretrained(args.llm_version, use_auth_token=True)
            self.llm = pipeline(
                "text-generation",
                model=args.llm_version,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        elif args.llm_version in ["../model/Mistral-Nemo-Instruct-2407"]:
            self.config = AutoConfig.from_pretrained(args.llm_version, use_auth_token=True)
            self.enc = AutoTokenizer.from_pretrained(args.llm_version, use_auth_token=True)
            self.llm = pipeline(
                "text-generation",
                model=args.llm_version,
                torch_dtype=torch.float16,
                device_map="auto",
                max_new_tokens=2048,
            )
        elif args.llm_version in ["../model/gemma-2-9b-it"]:
            self.config = AutoConfig.from_pretrained(args.llm_version, use_auth_token=True)
            self.enc = AutoTokenizer.from_pretrained(args.llm_version, use_auth_token=True)
            self.llm = pipeline(
                "text-generation",
                model=args.llm_version,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
        elif args.llm_version in ["../model/Qwen2.5-7B-Instruct"]:
            self.config = AutoConfig.from_pretrained(args.llm_version)
            # self.enc = AutoTokenizer.from_pretrained(args.llm_version)
            self.enc = AutoTokenizer.from_pretrained(args.llm_version, trust_remote_code=True)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                args.llm_version,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto"
            )
            self.llm = pipeline(
                "text-generation",
                model=self.llm_model,
                tokenizer=self.enc,
            )
        elif args.llm_version in ['ERNIE-Speed-8K', 'ERNIE-Speed-128K', 'ERNIE-Lite-8K', 'ERNIE-Tiny-8K']:
            self.llm = QianfanChatEndpoint(
                # streaming=True,
                model=args.llm_version,
                temperature=0.0001,
                )
            # self.llm = qianfan.ChatCompletion()
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
        print(action_list)
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
                    self.scratchpad += 'Answer is CORRECT'
                else: 
                    self.scratchpad += 'Answer is INCORRECT'
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

        # self.idd = []
        self.step_n += 1

    def prompt_agent(self) -> str:
        if self.llm_version in ['gpt-3.5-turbo', 'gpt-4', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-16k']:
            return gpt_format_step(self.llm(self._build_agent_prompt()))
        elif self.llm_version in ["../model/Llama-2-13b-chat-hf", "../model/Mixtral-8x7B-Instruct-v0.1"]:
            pad_token_id = self.enc.pad_token_id if self.enc.pad_token_id is not None else self.enc.eos_token_id
            # print('pad_token_id: ', pad_token_id)
            mid_ans = self.llm(self._build_agent_prompt()[1].content,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=13, # \n for llama2 and mixtral
                max_length=self.config.max_position_embeddings*2,
                truncation=True,
                pad_token_id=pad_token_id
                )
            return hf_format_step(
                mid_ans
            )
        elif self.llm_version in ["../model/Meta-Llama-3.1-70B-Instruct"]:
            pad_token_id = self.enc.pad_token_id if self.enc.pad_token_id is not None else self.enc.eos_token_id
            eos_token_id = self.enc.eos_token_id
            content = self._build_agent_prompt()[1].content
            mid_ans = self.llm(content,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=eos_token_id, # . for llama3.1
                # max_length=self.config.max_position_embeddings,
                max_new_tokens=256,
                truncation=True,
                pad_token_id=pad_token_id
                )
            mid_ans = mid_ans[0]["generated_text"].replace(content, '')
            return llama3_format_step(
                mid_ans
            )
        elif self.llm_version in ['../model/Mistral-Nemo-Instruct-2407']:
            pad_token_id = self.enc.pad_token_id if self.enc.pad_token_id is not None else self.enc.eos_token_id
            eos_token_id = self.enc.eos_token_id
            mid_ans = self.llm(
                    self._build_agent_prompt()[1].content,
                    pad_token_id=pad_token_id,
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=1,
                    # max_length=self.config.max_position_embeddings,
                    max_new_tokens=256,
                    truncation=True,
                    eos_token_id=eos_token_id, # \n for Mistral 0
                )
            return Nemo_format_step(
                mid_ans,self._build_agent_prompt()[1].content
            )
        elif self.llm_version in ['../model/gemma-2-9b-it']:
            pad_token_id = self.enc.pad_token_id if self.enc.pad_token_id is not None else self.enc.eos_token_id
            mid_ans = self.llm(
                    self._build_agent_prompt()[1].content,
                    pad_token_id=pad_token_id,
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=1,
                    max_new_tokens=256,
                    # max_length=self.config.max_position_embeddings,
                    truncation=True,
                    # eos_token_id=108, # \n for gemma
                    eos_token_id=227
                )
            return gemma_format_step(
                mid_ans,self._build_agent_prompt()[1].content
            )
        elif self.llm_version in ['../model/Qwen2.5-7B-Instruct']:
            pad_token_id = self.enc.pad_token_id if self.enc.pad_token_id is not None else self.enc.eos_token_id
            eos_token_id = self.enc.eos_token_id
            mid_ans = self.llm(
                    self._build_agent_prompt()[1].content,
                    pad_token_id=pad_token_id,
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=1,
                    # max_length=self.config.max_position_embeddings,
                    max_new_tokens=256,
                    truncation=True,
                    eos_token_id=eos_token_id, # \n eos
                )
            return Qwen_format_step(
                mid_ans,self._build_agent_prompt()[1].content
            )
        elif self.llm_version in ['ERNIE-Speed-8K', 'ERNIE-Speed-128K', 'ERNIE-Lite-8K', 'ERNIE-Tiny-8K']:
            llm_answer = qw_format_step(self.llm.invoke((self._build_agent_prompt())))
            # print(llm_answer)
            return llm_answer
        else:
            raise ValueError("The given llm_version is not correct.")

        return format_step(self.llm(self._build_agent_prompt()))

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
        # return ((self.step_n > self.max_steps) or (len(self.enc.encode(self._build_agent_prompt())) > 3896)) and not self.finished
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
    return step[0]["generated_text"].strip().split('\n')[-1].split(': ')[-1]

def gemma_format_step(step: str,content:str) -> str:
    mid_ans_generated_text = step[0]["generated_text"].replace(content, '')
    while first_non_newline_index < len(mid_ans_generated_text) and mid_ans_generated_text[first_non_newline_index] == '\n':
        first_non_newline_index += 1
    start_index = first_non_newline_index
    newline_index = mid_ans_generated_text.find('\n', start_index)
    if newline_index != -1:
        mid_ans_generated_text = mid_ans_generated_text[start_index:newline_index]
    return mid_ans_generated_text.strip()

def llama3_format_step(step: str) -> str:
    # return step.strip('\n').strip().replace('\n', '')
    # mid_ans = step[0]["generated_text"].strip().split('\n')[-1].split(': ')[-1]
    return step.split("\n")[0].replace('assistant', '').strip()

def Nemo_format_step(step: str,content:str) -> str:
    return step[0]["generated_text"].replace(content, '').split("\n")[0].strip()

def Qwen_format_step(step: str,content:str) -> str:
    result= step[0]["generated_text"].replace(content, '').split("\n")[0].strip()
    if ']' in result:
        result = result.split(']')[0] + ']'
    return result

def qw_format_step(step: str) -> str:
    # return step.strip('\n').strip().replace('\n', '')
    return step.content.strip('\n').strip().replace('\n', '')

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
