import openai 
openai.api_type = "azure"
openai.api_base = "https://new-llm.openai.azure.com/"
openai.api_key = "082b19bc29364b1bb39d2d9fb9b757d4"

def GetChatgptResult(prompt, 
                     engine='gpt-35-turbo', 
                     api_version='2023-03-15-preview',
                     temperature=0.7,
                     max_tokens=2048,
                     top_p=0.75,
                     frequency_penalty=0,
                     presence_penalty=2,
                     **kwargs):
    openai.api_version = api_version 
    response_text = ""
    try:
        response = openai.ChatCompletion.create(
            engine=engine,
            messages = [{"role":"user","content":prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty)
        response_text = response["choices"][0]["message"]["content"]
    except Exception as e:
        print(e)
        response_text = 'Error'

    return response_text


from transformers import LlamaForCausalLM
from transformers import AutoTokenizer
import torch 

class LlmGenerator:
    def __init__(self, model_type='gpt35', 
                 model_path=None, 
                 temperature=0.7, max_tokens=2048,
                 top_p=0.75, frequency_penalty=0, presence_penalty=2):
        self.model_type = model_type
        if model_type == 'gpt35':
            self.gen_func = GetChatgptResult
            self.arg_dict = dict(engine='gpt-35-turbo', 
                                 api_version='2023-03-15-preview',
                                 temperature=temperature,
                                 max_tokens=max_tokens,
                                 top_p=top_p, 
                                 frequency_penalty=frequency_penalty,
                                 presence_penalty=presence_penalty)
        elif model_type == 'gpt4':
            self.gen_func = GetChatgptResult
            self.arg_dict = dict(engine='gpt4', 
                                 api_version='2023-06-01-preview',
                                 temperature=temperature,
                                 max_tokens=max_tokens,
                                 top_p=top_p, 
                                 frequency_penalty=frequency_penalty,
                                 presence_penalty=presence_penalty)
        elif model_type == 'llama':
            self.model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            def gen_func(prompt):
                inputs = self.tokenizer(prompt, return_tensors='pt')
                inputs = {k: v.cuda() for k, v in inputs.items()}
                results = self.model.generate(**inputs, max_new_tokens=max_tokens,
                                              temperature=temperature, top_p=top_p, do_sample=True)
                len_input = inputs['input_ids'].size(-1)
                results = self.tokenizer.decode(results[0][len_input:])
                return results 
            self.gen_func = gen_func 
        else:
            raise ValueError(f'Model type {model_type} is not supported.')

    def generate(self, prompt):
        return self.gen_func(prompt)

    def generate(self, prompt):
        return self.gen_func(prompt, **self.arg_dict)


if __name__ == '__main__':
    prompt = '昨天的当天是明天的什么？'

    # GPT3.5
    generator = LlmGenerator('gpt35')
    response = generator.generate(prompt)
    print(f'GPT3.5 response: {response}')

    # GPT4
    generator = LlmGenerator('gpt4')
    response = generator.generate(prompt)
    print(f'GPT4 response: {response}')

    # Llama
    generator = LlmGenerator('llama', '/cpfs/share/models/LLaMA2_hf/7B')
    response = generator.generate(prompt)
    print(f'Llama2-7B response: {response}')