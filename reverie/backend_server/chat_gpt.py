import requests 
import os 
import datetime
import time 
import openai
import numpy as np

import sys
import pickle as pkl
import sklearn
from collections import defaultdict
import os.path as osp
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))


class ChatGPT:
    def __init__(self, model_type="text-davinci-003", endpoint='Completion', temperature=0, tokens=300,
                 presence_penalty=0, frequency_penalty=0,
                 save_folder=None, api_key=None, **kwargs):
        # self.model_engine = 'text-davinci-003'
        self.model_engine = model_type
        self.temperature = temperature
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.max_tokens = tokens

        if save_folder is None:
            save_folder = '../data/llm'
        if not osp.exists(save_folder):
            os.makedirs(save_folder)
        self.save_file = osp.join(save_folder, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.txt')

        self.last_raw_completion = None
        self.messages = []
        self.stop = ["<|im_end|>"]

        self.endpoint = endpoint
        if endpoint == 'Completion':
            self.api = openai.Completion
        else:
            self.api = openai.ChatCompletion

        self.api_key = api_key

    def generate(self, task):
        openai.api_key = self.api_key
        openai.api_type = "open_ai"
        openai.api_base = "https://api.openai.com/v1"
        openai.api_version = None

        if self.endpoint == 'ChatCompletion':
            return self.generate_chat(task)

        # self.messages.append({"role": "system", "content": task})
        # self.messages.append({"role": "user", "content": "Begin!"})
        prompt = f"<|im_start|>system\n{task}\n<|im_end|>\n<|im_start|>assistant\n"

        completion = self.api.create(
            engine=self.model_engine,
            prompt=prompt,
            max_tokens=self.max_tokens,
            n=1,
            stop=self.stop,
            temperature=self.temperature,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            timeout=1000,
        )
        self.last_raw_completion = completion
        self.last_response = completion
        response = completion.choices[0].text
        # self.messages.append({"role": "assistant", "content": response})

        with open(self.save_file, 'a', encoding='utf-8') as f:
            f.write('[Prompt]\n')
            f.write(f'{task}\n')
            f.write('[Response]\n')
            f.write(f'{response}\n')

        return response

    def generate_chat(self, task):
        # messages = [
        #     {"role": "system", "content": task}
        # ]
        messages = [
            {"role": "system", "content": 'You are a helpful assistant. You should follow the instructions and output corresponding outputs'},
            {"role": "user", "content": task}
        ]

        completion = self.api.create(
            model=self.model_engine,
            messages=messages,
            max_tokens=self.max_tokens,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            n=1,
            stop=self.stop,
            temperature=self.temperature,
            timeout=30,
        )
        self.last_raw_completion = completion
        response = completion.choices[0].message['content']
        # self.messages.append({"role": "assistant", "content": response})
        with open(self.save_file, 'a', encoding='utf-8') as f:
            f.write('[Prompt]\n')
            f.write(f'{task}\n')
            f.write('[Response]\n')
            f.write(f'{response}\n')

        return response


class ChatGPTV2A:
    def __init__(self, model_type='3', temperature=0, tokens=300, save_folder=None, frequency_penalty=0,
                 presence_penalty=0, api_key=None, api_version="2023-03-15-preview", sleep=0, **kwargs):
        self.temperature = temperature
        self.tokens = tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.model_type = model_type
        self.sleep = sleep
        self.timeout = kwargs.get('timeout', 45)

        self.api_version = api_version
        # self.api_base = "https://llm.openai.azure.com/"
        self.api_base = "https://new-llm.openai.azure.com/"
        if model_type == '3':
            self.model_name = "gpt-35-turbo"
        else:
            self.model_name = model_type

        self.api_key = api_key

        self.messages = []
        self.accumulated_usage = defaultdict(int)
        if save_folder is None:
            save_folder = '../data/llm'
        if not osp.exists(save_folder):
            os.makedirs(save_folder)
        time_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.save_file = osp.join(save_folder, self.model_name + time_str + '.txt')
        self.usage_save_file = osp.join(save_folder, self.model_name + time_str + '_usage.pkl')
        self.last_response = None

    def generate(self, task=None, message=None, *args, **kwargs):

        openai.api_key = self.api_key
        openai.api_type = "azure"
        openai.api_base = self.api_base
        openai.api_version = self.api_version
        # openai.api_version = "2023-05-15"
        if message is None:
            messages = [
                {"role": "system", "content": 'You are a helpful assistant. You should follow the instructions and output corresponding outputs'},
                {"role": "user", "content": task}
            ]
        else:
            messages = message

        request_kwargs = dict(
            engine=self.model_name,
            messages=messages,
            temperature=self.temperature if 'temperature' not in kwargs else kwargs['temperature'],
            max_tokens=self.tokens if 'max_tokens' not in kwargs else kwargs['max_tokens'],
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=None,
            timeout=self.timeout,
            request_timeout=self.timeout,
        )
        changeable_keys = ['max_tokens', 'temperature', 'frequency_penalty', 'presence_penalty']
        for k in changeable_keys:
            if k in kwargs:
                request_kwargs[k] = kwargs[k]

        # self.messages.append({"role": "system", "content": task})
        # self.messages.append({"role": "user", "content": "Begin!"})
        for _ in range(3):
            try:
                response = openai.ChatCompletion.create(**request_kwargs)
                time.sleep(self.sleep)
                break
            except Exception as e:
                print(e)
                time.sleep(1)

        if 'content' in response['choices'][0]['message']:
            rtn = response['choices'][0]['message']['content']
        else:
            print(response['choices'][0])
            request_kwargs['temperature'] = 0.7
            request_kwargs['messages'] = [{"role": "system", "content": "Do not contain any unsafe contents that will be filtered"}] + request_kwargs['messages']
            response = openai.ChatCompletion.create(**request_kwargs)
            if 'content' not in response['choices'][0]['message']:
                print(response['choices'][0])
                rtn = str(response['choices'][0]["finish_reason"])
            else:
                rtn = response['choices'][0]['message']['content']
        self.last_response = response
        # self.messages.append({"role": "assistant", "content": rtn}
        with open(self.save_file, 'a', encoding='utf-8') as f:
            f.write('[Prompt]\n')
            f.write(f'{task if task is not None else str(message)}\n')
            f.write('[Response]\n')
            f.write(f'{rtn}\n')

        for k in response['usage']:
            self.accumulated_usage[k] += response['usage'][k]
        with open(self.usage_save_file, 'wb') as f:
            pkl.dump(self.accumulated_usage, f)
        return rtn


class ChatGPTRecordWrapper:
    def __init__(self, chatgpt_name='ChatGPTV2A', save_data_path='../data/collected_data', save_data_fn='data.pkl',
                 chatgpt_log_file_name=None, *args, **kwargs):
        self.chatgpt_name = chatgpt_name
        self.chatgpt = eval(chatgpt_name)(*args, **kwargs)
        if chatgpt_log_file_name is not None:
            self.chatgpt.save_file = chatgpt_log_file_name

        self.save_data_path = save_data_path
        self.save_data_fn = save_data_fn
        print(save_data_path, save_data_fn)
        self.save_data_fp = osp.join(save_data_path, save_data_fn) if save_data_fn is not None else None
        self.data = defaultdict(list)
        os.makedirs(save_data_path, exist_ok=True)

        self.run_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

    def generate(self, task=None, message=None, meta_key='default', meta_info=None, **kwargs):
        resp = self.chatgpt.generate(task, message, **kwargs)
        last_raw_resp = self.chatgpt.last_response
        additional_meta_info = {
            'model_name': self.chatgpt_name,
            'run_date': self.run_date,
            'usage': last_raw_resp['usage'].to_dict() if last_raw_resp is not None else {}
        }
        if meta_info is not None:
            additional_meta_info.update(meta_info)

        if resp != '' and self.save_data_fp is not None:
            # should have fcntl, but we are in single thread
            if os.path.exists(self.save_data_fp):
                self.data = pkl.load(open(self.save_data_fp, 'rb'))

            self.data[meta_key].append({
                'input': task if task is not None else message,
                'output': resp,
                'meta_info': additional_meta_info
            })

            with open(self.save_data_fp, 'wb') as f:
                pkl.dump(self.data, f)
        return resp


class LocalLLama:
    def __init__(self, model_path, save_file, temperature=0.01, tokens=300, save_folder=None, frequency_penalty=0,
                 presence_penalty=0, model_type='llama', *args, **kwargs):
        from transformers import LlamaForCausalLM, AutoModelForCausalLM
        from transformers import AutoTokenizer
        from transformers.generation.utils import GenerationConfig
        import torch

        self.tokens = tokens
        self.temperature = temperature
        self.save_file = save_file
        self.model_type = model_type

        if self.model_type == 'llama':
            self.model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif self.model_type == 'baichuan':
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
            # self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float32,
            #                                              trust_remote_code=True)
            # self.model = self.model.cuda()
            self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        else:
            raise NotImplementedError

    def generate(self, prompt=None, message=None, **kwargs):
        if message is not None:
            prompt = message[0]['content']
        temperature=self.temperature if 'temperature' not in kwargs else kwargs['temperature'] + 1
        max_tokens=self.tokens if 'max_tokens' not in kwargs else kwargs['max_tokens']
        top_p = 1 if 'top_p' not in kwargs else kwargs['top_p']

        if self.model_type == 'llama':
            wrapped_prompt = f"""[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>\n{prompt}\n[/INST]"""

            inputs = self.tokenizer(wrapped_prompt, return_tensors='pt')
            inputs = {k: v.cuda() for k, v in inputs.items()}
            results = self.model.generate(**inputs, max_new_tokens=max_tokens,
                                          temperature=temperature, top_p=top_p, do_sample=True)
            len_input = inputs['input_ids'].size(-1)
            results = self.tokenizer.decode(results[0][len_input:])
            rtn = results.strip('</s>')
        elif self.model_type == 'baichuan':
            messages = [{"role": "user", "content": prompt}]
            rtn = self.model.chat(self.tokenizer, messages)
        else:
            raise NotImplementedError

        with open(self.save_file, 'a', encoding='utf-8') as f:
            f.write('[Prompt]\n')
            f.write(f'{prompt}\n')
            f.write('[Response]\n')
            f.write(f'{rtn}\n')

        return rtn


class OpenAIEmbedding:
    def __init__(self, model_type="text-embedding-ada-002", lazy_db_file_name='../data/lazy_embedding.pkl',
                 api_key=None, **kwargs):
        # self.model_engine = 'text-davinci-003'
        self.model_engine = model_type

        self.api_key = api_key
        self.lazy_db_file_name = lazy_db_file_name
        self.lazy_data = defaultdict(dict)
        os.makedirs(lazy_db_file_name, exist_ok=True)

        self.api_version = "2023-03-15-preview"
        self.api_base = "https://llm.openai.azure.com/"
        # self.api_base = "https://new-llm.openai.azure.com/"

    def embedding(self, sentence, model=None):
        openai.api_key = self.api_key
        openai.api_type = "azure"
        openai.api_base = self.api_base
        openai.api_version = self.api_version
        # openai.api_version = "2023-03-15-preview"
        if model is not None:
            engine = model
        else:
            engine = self.model_engine

        if not isinstance(sentence, list):
            text = [sentence]
        else:
            text = sentence

        # with open(self.lazy_db_file_name, 'rb') as f:
        #     self.lazy_data = pkl.load(f)
        # if self.lazy_data

        resp = openai.Embedding.create(
            engine=engine,
            input=text
        )
        # vec = [np.array(resp['data'][i]['embedding']) for i in range(len(np.array(resp['data'])))]
        vec = resp['data'][0]['embedding']
        return vec

    # def cos_sim(self, embedding1, embedding2):
    #     from sklearn.metrics.pairwise import cosine_similarity
    #     scores = cosine_similarity(embedding1, embedding2)
    #
    #     return scores
