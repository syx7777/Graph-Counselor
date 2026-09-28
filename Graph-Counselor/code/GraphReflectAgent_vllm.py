import re, string, os
from typing import List, Union, Literal
from enum import Enum
import tiktoken
import openai
import qianfan
import time
import json
import requests
import time
# from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from graph_prompts import GRAPH_DEFINITION, REFLECTION_HEADER, LAST_TRIAL_HEADER, REFLECTION_AFTER_LAST_TRIAL_HEADER
from graph_fewshots import EXAMPLES, REFLECT_EXAMPLES, EVAL_EXAMPLES, REFLECT_EXAMPLES_BASE, REFLECT_EXAMPLES_SHORT_MULTIPLE
from tools import graph_funcs, retriever
import logging
from transformers import pipeline, AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphReflectAgent_vllm:
    def __init__(self,
                 args,
                 agent_prompt,
                 reflect_prompt,
                 eval_prompt,
                 reflect_prompt_base,
                 reflect_prompt_short_multiple,
                 ) -> None:

        self.max_steps = args.max_steps
        self.agent_prompt = agent_prompt
        self.reflect_prompt = reflect_prompt
        self.reflect_prompt_base = reflect_prompt_base
        self.reflect_prompt_short_multiple = reflect_prompt_short_multiple
        self.eval_prompt = eval_prompt
        self.examples = EXAMPLES[args.ref_dataset]
        self.reflect_examples = REFLECT_EXAMPLES[args.ref_dataset]
        self.reflect_examples_base = REFLECT_EXAMPLES_BASE[args.ref_dataset]
        self.reflect_examples_short_multiple = REFLECT_EXAMPLES_SHORT_MULTIPLE[args.ref_dataset]
        self.eval_examples = EVAL_EXAMPLES[args.ref_dataset]

        self.reflections: List[str] = [] # reflect
        self.reflections_str: str = ''
        self.round_n = 1

        self.api_url = args.api_url
        self.api_url2 = args.api_url2
        self.api_url3 = args.api_url3
        self.llm_version = args.llm_version
        self.eval_llm_version = args.eval_llm_version
        self.reflect_version = args.reflect_version
        self.reflect_prompt_way = args.reflect_prompt

        self.judge_correct = args.judge_correct
        if args.llm_version in ['gpt-3.5-turbo', 'gpt-4', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-16k']:
            self.enc = tiktoken.encoding_for_model("text-davinci-003")
        elif args.llm_version in ["../model/Llama-2-13b-chat-hf", "../model/Mixtral-8x7B-Instruct-v0.1", '../model/Meta-Llama-3.1-70B-Instruct','../model/Qwen2.5-72B-Instruct']:
            self.enc = AutoTokenizer.from_pretrained(args.llm_version, use_auth_token=True)
        elif args.llm_version in ["../model/Mistral-Nemo-Instruct-2407"]:
            self.enc = AutoTokenizer.from_pretrained(args.llm_version, use_auth_token=True)
        elif self.llm_version in ['../model/gemma-2-9b-it', '../model/gemma-2-9b-it']:
            self.enc = AutoTokenizer.from_pretrained(args.llm_version, use_auth_token=True)
        elif args.llm_version in ["../model/Qwen2.5-7B-Instruct"]:
            self.enc = AutoTokenizer.from_pretrained(args.llm_version, trust_remote_code=True)
        elif args.llm_version in ['ERNIE-Speed-8K', 'ERNIE-Speed-128K', 'ERNIE-Lite-8K', 'ERNIE-Tiny-8K']:
            self.enc = tiktoken.encoding_for_model("text-davinci-003")
        else:
            raise ValueError("The given llm_version is not correct.")
        
        self.reflexion_strategy =args.reflexion_strategy
        self.max_reflect =args.max_reflect

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
            self.round_n = 1
            self.reflections_str=""
            self.answer_first = ""
        
        self.question = question
        self.key = answer

        if self.judge_correct == 'groundtruth':
            while not self.is_correct() and self.round_n <= int(self.max_reflect): #/is_correct_LLM
                if (self.is_halted() or self.is_finished()) and not self.is_correct():
                    self.reflect(self.reflexion_strategy)
                    self.round_n += 1
                self.__reset_agent()
                while not self.is_halted() and not self.is_finished():
                    self.step()
        elif self.judge_correct == 'llm':
            while not self.is_correct_LLM() and self.round_n <= int(self.max_reflect): #/is_correct_LLM
                if (self.is_halted() or self.is_finished()) and not self.is_correct_LLM():
                    self.reflect(self.reflexion_strategy)
                    self.round_n += 1
                self.__reset_agent()
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
                    self.scratchpad +='Answer is '+str(self.answer)
                else: 
                    self.scratchpad +='Answer is '+str(self.answer)
                self.finished = True
                self.step_n += 1
                if self.round_n == 1:
                    self.answer_first = self.answer
                return

            elif action_type == 'Retrieve':
                try:
                    idd, node = self.node_retriever.search_single(argument, 1)
                    self.scratchpad += f"The ID of this retrieval target node is {idd}."
                except openai.RateLimitError:
                    self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                except:
                    self.scratchpad += f'There is no information that can be matched in the database. Please try another query.'

            elif action_type == 'Neighbor':
                try:
                    node_id, neighbor_type = argument.split(', ')
                    node_id = remove_quotes(node_id)
                    neighbor_type = remove_quotes(neighbor_type)
                    self.scratchpad += f"The {neighbor_type} neighbors of {node_id} are: " + str(self.graph_funcs.check_neighbours(node_id, neighbor_type)) + '. '
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
                    self.scratchpad += f"The {feature_name} feature of {node_id} are: " + self.graph_funcs.check_nodes(node_id, feature_name) + '. '
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
                    self.scratchpad += f"The {neighbor_type} neighbor node degree of {node_id} are: " + self.graph_funcs.check_degree(node_id, neighbor_type) + '. '
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
            total_tokens = len(self.enc.encode(self._build_agent_prompt_4096()))
            payload = {
            "model": "Llama-2-13b-chat-hf",
            "prompt": self._build_agent_prompt_4096(),
            "max_tokens": min(512, 4096 - total_tokens),
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
            "stop": "\n",
        }
            response = requests.post(self.api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                print('*******************************************************************')
                print(self._build_agent_prompt_4096())
                print(f"Response: {response.json()}")
                print('*******************************************************************')

                mid_ans = result["choices"][0]["text"]
            else:
                error_message = (
                f"vLLM API Error: Status Code {response.status_code}\n"
                f"Response Text: {response.text}\n"
                f"Suggested Action: Please check the API request parameters or contact support."
                )
                print(error_message)
                raise RuntimeError(error_message)
            return hf_format_step(
                mid_ans
            )
        elif self.llm_version in ["../model/Mixtral-8x7B-Instruct-v0.1"]:
            payload = {
            "model": "Mixtral-8x7B-Instruct-v0.1",
            "prompt": self._build_agent_prompt()[1].content,
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
            "stop": "\n",
        }
            response = requests.post(self.api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                # print(f"Response: {response.json()}")
                mid_ans = result["choices"][0]["text"]
            else:
                error_message = (
                f"vLLM API Error: Status Code {response.status_code}\n"
                f"Response Text: {response.text}\n"
                f"Suggested Action: Please check the API request parameters or contact support."
                )
                print(error_message)
                raise RuntimeError(error_message)
            return hf_format_step(
                mid_ans
            )
        elif self.llm_version in ["../model/Meta-Llama-3.1-70B-Instruct"]:
            payload = {
            "model": "Llama-3.1-70B-Instruct",
            "prompt": self._build_agent_prompt()[1].content,
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
        }
            response = requests.post(self.api_url, json=payload)
            if response.status_code == 200:
                result = response.json()
                mid_ans = result["choices"][0]["text"]
            else:
                error_message = (
                f"vLLM API Error: Status Code {response.status_code}\n"
                f"Response Text: {response.text}\n"
                f"Suggested Action: Please check the API request parameters or contact support."
                )
                print(error_message)
                raise RuntimeError(error_message)

            return llama3_format_step(
                mid_ans
            )
        elif self.llm_version in ['../model/Mistral-Nemo-Instruct-2407']:
            payload = {
            "model": "Mistral-Nemo-Instruct-2407",
            "prompt": self._build_agent_prompt()[1].content,
            "max_tokens": 512,
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
                error_message = (
                f"vLLM API Error: Status Code {response.status_code}\n"
                f"Response Text: {response.text}\n"
                f"Suggested Action: Please check the API request parameters or contact support."
                )
                print(error_message)
                raise RuntimeError(error_message)

            return Nemo_format_step(
                mid_ans,self._build_agent_prompt()[1].content
            )
        elif self.llm_version in ['../model/gemma-2-9b-it', '../model/gemma-2-9b-it']:
            total_tokens = len(self.enc.encode(self._build_agent_prompt()[1].content))
            if total_tokens < 3840:
                payload = {
                "model": "gemma-2-9b-it",
                "prompt": self._build_agent_prompt()[1].content,
                "max_tokens": 512,
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
                    error_message = (
                    f"vLLM API Error: Status Code {response.status_code}\n"
                    f"Response Text: {response.text}\n"
                    f"Suggested Action: Please check the API request parameters or contact support."
                    )
                    print(error_message)
                    raise RuntimeError(error_message)
                return gemma_format_step(
                    mid_ans,self._build_agent_prompt()[1].content
                )
            else:
                return "The number of tokens you requested exceeds the context of the model. Please reduce the length of the completion."

        elif self.llm_version in ['../model/Qwen2.5-7B-Instruct']:
            payload = {
            "model": "Qwen2.5-7B-Instruct",
            "prompt": self._build_agent_prompt()[1].content,
            "max_tokens": 512,
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
                error_message = (
                f"vLLM API Error: Status Code {response.status_code}\n"
                f"Response Text: {response.text}\n"
                f"Suggested Action: Please check the API request parameters or contact support."
                )
                print(error_message)
                raise RuntimeError(error_message)

            return Qwen_format_step(
                mid_ans,self._build_agent_prompt()[1].content
            )
        elif self.llm_version in ["../model/Qwen2.5-72B-Instruct"]:
            payload = {
            "model": "Qwen2.5-72B-Instruct",
            "prompt": self._build_agent_prompt()[1].content,
            "max_tokens": 512,
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
                            reflections = self.reflections_str, 
                            question = self.question,
                            scratchpad = truncate_scratchpad2(self.scratchpad, self.enc),
                            graph_definition = self.graph_definition
                            )
    
    def _build_agent_prompt_4096(self) -> str:
        return truncate_scratchpad3(self.agent_prompt.format_messages(
                            examples = self.examples,
                            reflections = self.reflections_str,
                            question = self.question,
                            scratchpad = truncate_scratchpad3(self.scratchpad, self.enc, 1600),
                            graph_definition = self.graph_definition
                            )[1].content, self.enc, 3900)

    def _build_eval_prompt(self) -> str:
        return self.eval_prompt.format_messages(
                            examples = self.eval_examples,
                            question = self.question,
                            scratchpad = self.scratchpad,
                            graph_definition = self.graph_definition
                            )

    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)
    
    def is_correct_LLM(self) -> bool:
        if self.answer=="":
            return False
        payload = {
        "model": self.eval_llm_version,
        "prompt": truncate_scratchpad3(truncate_scratchpad(self._build_eval_prompt()[1].content,self.enc,30000),self.enc,30000),
        "max_tokens": 256,
        "temperature": 0.3,
        "top_p": 1,
        "n": 1,
        "stop": ["\n\n\n"],
    }
        response = requests.post(self.api_url2, json=payload)
        if response.status_code == 200:
            result = response.json()
            mid_ans = result["choices"][0]["text"]
        else:
            raise RuntimeError(f"vLLM API Error: {response.status_code}, {response.text}")

        return eval_format_step(mid_ans)

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps) or (len(self.enc.encode(self._build_agent_prompt()[1].content)) > 10000)) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.answer = ''
        self.finished = False
        self.scratchpad: str = ''

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key

    def reflect(self,strategy) -> None: # reflect
        print('Reflecting...')
        if strategy == "Last_attempt":
            self.reflections = [self.scratchpad]
            self.reflections_str = format_last_attempt(self.question, self.reflections[0],self.enc)
        elif strategy == "Reflexion": 
            # self.reflections += [self.prompt_reflection()]
            self.reflections = [self.prompt_reflection()]
            self.reflections_str = format_reflections(self.reflections)
        elif strategy == "Last_attempt_and_Reflexion": 
            self.reflections_str = format_last_attempt(self.question, self.scratchpad,self.enc)
            self.reflections = [self.prompt_reflection()]
            self.reflections_str += format_reflections(self.reflections, header = REFLECTION_AFTER_LAST_TRIAL_HEADER)
        else:
            raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
        print(self.reflections_str)

    def prompt_reflection(self) -> str:
        if self.reflect_version in ['gpt-3.5-turbo', 'gpt-4', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-16k']:
            return gpt_format_step(self.llm(self._build_reflection_prompt()))
        elif self.reflect_version in ["Llama-2-13b-chat-hf"] :
            total_tokens = len(self.enc.encode(self._build_reflection_prompt_4096()))
            payload = {
            "model": "Llama-2-13b-chat-hf",
            "prompt": self._build_reflection_prompt_4096(),
            "max_tokens": min(512, 4096 - total_tokens),
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
            "stop": "\n\n\n",
        }
            response = requests.post(self.api_url3, json=payload)
            if response.status_code == 200:
                result = response.json()
                print('*******************************************************************')
                print(self._build_reflection_prompt_4096())
                print(f"Response: {response.json()}")
                print('*******************************************************************')
                mid_ans = result["choices"][0]["text"]
            else:
                raise RuntimeError(f"vLLM API Error: {response.status_code}, {response.text}")
            print('*******************************************************************')
            print(f'mid_ans: {mid_ans}')
            print('*******************************************************************')
            return reflect_format_step(mid_ans)
            

        elif self.reflect_version in ["Mixtral-8x7B-Instruct-v0.1"]:
            payload = {
            "model": "Mixtral-8x7B-Instruct-v0.1",
            "prompt": self._build_reflection_prompt()[1].content,
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
            }
            response = requests.post(self.api_url3, json=payload)
            if response.status_code == 200:
                result = response.json()
                mid_ans = result["choices"][0]["text"]
            else:
                raise RuntimeError(f"vLLM API Error: {response.status_code}, {response.text}")
            return reflect_format_step(mid_ans)

        elif self.reflect_version in ["Meta-Llama-3.1-70B-Instruct"]:
            payload = {
            "model": "Llama-3.1-70B-Instruct",
            "prompt": self._build_reflection_prompt()[1].content,
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
            "stop": "\n\n\n",
        }
            total_tokens = len(self._build_reflection_prompt()[1].content)
            if total_tokens < 31700:
                response = requests.post(self.api_url3, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    mid_ans = result["choices"][0]["text"]
                else:
                    raise RuntimeError(f"vLLM API Error: {response.status_code}, {response.text}")
                return reflect_format_step(mid_ans)
            else:
                return ""

        elif self.reflect_version in ["Mistral-Nemo-Instruct-2407"]:
            total_tokens = len(self._build_reflection_prompt()[1].content)
            payload = {
            "model": "Mistral-Nemo-Instruct-2407",
            "prompt": self._build_reflection_prompt()[1].content,
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
            "stop": "\n\n\n",
        }
            if total_tokens < 31700:
                response = requests.post(self.api_url3, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    print(f"Response: {response.json()}")
                    mid_ans = result["choices"][0]["text"]
                else:
                    raise RuntimeError(f"vLLM API Error: {response.status_code}, {response.text}")
                print('*******************************************************************')
                print(f'mid_ans: {mid_ans}')
                print('*******************************************************************')
                return reflect_format_step(mid_ans)
            else:
                return ""
        elif self.reflect_version in ['gemma-2-9b-it']:
            payload = {
            "model": "gemma-2-9b-it",
            "prompt": self._build_reflection_prompt()[1].content,
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
        }
            response = requests.post(self.api_url3, json=payload)
            if response.status_code == 200:
                result = response.json()
                mid_ans = result["choices"][0]["text"]
            else:
                raise RuntimeError(f"vLLM API Error: {response.status_code}, {response.text}")
            print('*******************************************************************')
            print(f'mid_ans: {mid_ans}')
            print('*******************************************************************')
            return reflect_format_step(mid_ans)
        elif self.reflect_version in ['Qwen2.5-7B-Instruct']:

            payload = {
            "model": "Qwen2.5-7B-Instruct",
            "prompt": self._build_reflection_prompt()[1].content,
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
            "stop": "\n\n\n"   
        }
            response = requests.post(self.api_url3, json=payload)
            if response.status_code == 200:
                result = response.json()
                mid_ans = result["choices"][0]["text"]
            else:
                raise RuntimeError(f"vLLM API Error: {response.status_code}, {response.text}")
            print('*******************************************************************')
            print(f'mid_ans: {mid_ans}')
            print('*******************************************************************')
            if self.reflect_prompt_way == "multiple":
                return reflect_format_step(mid_ans)
            else:
                return mid_ans.splitlines()[0]
        elif self.reflect_version in ["Qwen2.5-72B-Instruct"]:
            payload = {
            "model": "Qwen2.5-72B-Instruct",
            "prompt": self._build_reflection_prompt()[1].content,
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
        }
            response = requests.post(self.api_url3, json=payload)
            if response.status_code == 200:
                result = response.json()
                mid_ans = result["choices"][0]["text"]
            else:
                raise RuntimeError(f"vLLM API Error: {response.status_code}, {response.text}")
            return reflect_format_step(mid_ans)
        elif self.reflect_version in ['ERNIE-Speed-8K', 'ERNIE-Speed-128K', 'ERNIE-Lite-8K', 'ERNIE-Tiny-8K']:
            llm_answer = qw_format_step(self.llm.invoke((self._build_reflection_prompt())))
            return llm_answer
        else:
            raise ValueError("The given llm_version is not correct.")
        
    def _build_reflection_prompt(self) -> str:
        if self.reflect_prompt_way == "multiple":
            return self.reflect_prompt.format_messages(
                            examples = self.reflect_examples,
                            question = self.question,
                            scratchpad = truncate_scratchpad(self.scratchpad, self.enc),
                            graph_definition = self.graph_definition
                            )
        elif self.reflect_prompt_way == "short_multiple":
            return self.reflect_prompt_short_multiple.format_messages(
                            examples = self.reflect_examples_short_multiple,
                            question = self.question,
                            scratchpad = truncate_scratchpad(self.scratchpad, self.enc),
                            graph_definition = self.graph_definition
                            )
        else:
            return self.reflect_prompt_base.format_messages(
                            examples = self.reflect_examples_base,
                            question = self.question,
                            scratchpad = truncate_scratchpad(self.scratchpad, self.enc),
                            graph_definition = self.graph_definition
                            )
        
    def _build_reflection_prompt_4096(self) -> str: 
        if self.reflect_prompt_way == "multiple":
            return truncate_scratchpad3(self.reflect_prompt.format_messages(
                            examples = self.reflect_examples,
                            question = self.question,
                            scratchpad = truncate_scratchpad4(self.scratchpad, self.enc),
                            graph_definition = self.graph_definition
                            )[1].content,self.enc)
        elif self.reflect_prompt_way == "short_multiple":
            return truncate_scratchpad3(self.reflect_prompt_short_multiple.format_messages(
                            examples = self.reflect_examples_short_multiple,
                            question = self.question,
                            scratchpad = truncate_scratchpad4(self.scratchpad, self.enc),
                            graph_definition = self.graph_definition
                            )[1].content,self.enc)
        else:
            return truncate_scratchpad3(self.reflect_prompt_base.format_messages(
                            examples = self.reflect_examples_base,
                            question = self.question,
                            scratchpad = truncate_scratchpad4(self.scratchpad, self.enc),
                            graph_definition = self.graph_definition
                            )[1].content,self.enc)
    
    def caculate_round_n(self) -> str:
        return str(self.round_n)



### String Stuff ###
# gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")

def split_checks(input_string):
    pattern = r'\w+\[.*?\]'
    # Use re.findall to get all matches
    result = re.findall(pattern, input_string)
    return result

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

def eval_format_step(step: str) -> bool: 
    match = re.search(r'Judgment:\s*\[(.*?)\]', step)
    if match:
        content = match.group(1)  
        result = content != "No" 
    else:
        result = True
    return result

def reflect_format_step(step:str) -> str:
    end_marker = "END OF REFLECTION"
    if end_marker.lower() in step.lower():
        result = step[:step.lower().find(end_marker.lower()) + len(end_marker)]
    else:
        result = step
    return result


def normalize_answer(s): 
  def remove_articles(text): 
    return re.sub(r"\b(a|an|the|usd)\b", " ", text)
  
  def white_space_fix(text): 
      return " ".join(text.split())

  def remove_punc(text):
      exclude = set(string.punctuation)
      return " ".join(ch for ch in text if ch not in exclude)

  def lower(text):
      return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def EM(answer, key) -> bool:
    return normalize_answer(str(answer)) == normalize_answer(str(key))

def format_last_attempt(question: str,
                        scratchpad: str,
                        tokenizer,
                        header: str = LAST_TRIAL_HEADER):
    return header + f'Question: {question}\n' + truncate_scratchpad(scratchpad, tokenizer).strip('\n').strip() + '\n(END PREVIOUS TRIAL)\n'

def truncate_scratchpad(scratchpad: str, tokenizer, n_tokens: int = 1600) -> str: 
    lines = scratchpad.split('\n')
    observations = filter(lambda x: x.startswith('Observation'), lines)
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
    while len(tokenizer.encode('\n'.join(lines))) > n_tokens:
        if not observations_by_tokens: 
            break
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = largest_observation.split(':')[0] + ': [truncated wikipedia excerpt]'
    return '\n'.join(lines)

def format_reflections(reflections: List[str],
                        header: str = REFLECTION_HEADER) -> str:
    if reflections == []:
        return ''
    else:
        print(reflections)
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])
    
def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '') 

def truncate_scratchpad2(scratchpad: str, tokenizer, n_tokens: int = 20000) -> str: 
    lines = scratchpad.split('\n')
    observations = list(filter(lambda x: x.startswith('Observation'), lines))
    while len(tokenizer.encode('\n'.join(lines))) > n_tokens:
        if not observations:  
            break
        next_observation = observations.pop(0)
        ind = lines.index(next_observation)
        lines[ind] = next_observation.split(':')[0] + ': [truncated wikipedia excerpt]'
    return '\n'.join(lines)

def truncate_scratchpad3(scratchpad: str, tokenizer, n_tokens: int = 3900) -> str: 
    lines = scratchpad.split('\n')
    observations = list(filter(lambda x: x.startswith('Observation'), lines))
    while len(tokenizer.encode('\n'.join(lines))) > n_tokens:
        if len(observations) < 2:  
            lines.pop(0)
        else:
            next_observation = observations.pop(0)
            ind = lines.index(next_observation)
            lines[ind] = next_observation.split(':')[0] + ': [truncated wikipedia excerpt]'
    return '\n'.join(lines)

def truncate_scratchpad4(scratchpad: str, tokenizer, n_tokens: int = 1600) -> str: 
    lines = scratchpad.split('\n')
    observations = filter(lambda x: x.startswith('Observation'), lines)
    actions = filter(lambda x: x.startswith('Action'), lines)
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
    actions_by_tokens = sorted(actions, key=lambda x: len(tokenizer.encode(x)))
    while len(tokenizer.encode('\n'.join(lines))) > n_tokens:
        if not observations_by_tokens:  
            break
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = largest_observation.split(':')[0] + ': [truncated wikipedia excerpt]'
    while len(tokenizer.encode('\n'.join(lines))) > n_tokens:
        if not actions_by_tokens:  
            break  
        largest_actions = actions_by_tokens.pop(-1)
        ind = lines.index(largest_actions)
        lines[ind] = largest_actions.split(':')[0] + ': [truncated wikipedia excerpt]'

    return '\n'.join(lines)