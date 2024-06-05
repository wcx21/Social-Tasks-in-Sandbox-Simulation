from chat_gpt import ChatGPTRecordWrapper, OpenAIEmbedding, LocalLLama
import os
import datetime

# Copy and paste your OpenAI API Key
openai_api_key = '<OpenAI API Key>'
# Put your name
key_owner = "<Name>"

maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

fs_storage = "../../environment/frontend_server/storage"
fs_temp_storage = "../../environment/frontend_server/temp_storage"

collision_block_id = "32125"

# Verbose
debug = True

open_ai_key = '<OpenAI API Key>'
agent_data_dir = './data/agent'
if not os.path.exists(agent_data_dir):
    os.makedirs(agent_data_dir, exist_ok=True)
chatgpt_log_file_name = os.path.join(agent_data_dir, 'gpt35_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.txt')

llm_kwargs = dict(
            chatgpt_name='ChatGPTV2A',
            model_type='gpt-35-turbo',
            # model_type='gpt-35-turbo-16k',
            presence_penalty=0.1,
            save_folder = './data/memory', # save LLM prompt and response pairs
            save_data_path = './data/collected_data/',
            # save_data_fn = 'pipeline_full_v1_cn_en_common.pkl',
            save_data_fn=None,
            tokens = 1500,
            api_key = open_ai_key,
            temperature = 0.2,
            chatgpt_log_file_name=chatgpt_log_file_name,
        )
chatgpt = ChatGPTRecordWrapper(**llm_kwargs)
chatgptEmbedding = OpenAIEmbedding(api_key=open_ai_key)

gpt4 = None
llama_model = None


def get_gpt4():
    return gpt4


def get_llama():
    return llama_model


def build_gpt4():
    global gpt4
    gpt4_log_file_name = os.path.join(agent_data_dir, 'gpt4_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.txt')
    gpt4_kwargs = dict(
                chatgpt_name='ChatGPTV2A',
                model_type='gpt4',
                # model_type='gpt-35-turbo-16k',
                presence_penalty=0.1,
                save_folder = './data/memory', # save LLM prompt and response pairs
                save_data_path = './data/collected_data/',
                # save_data_fn = 'gpt4_pipeline_full_v1_cn_en_common.pkl',
                save_data_fn=None,
                tokens = 1500,
                api_key = open_ai_key,
                api_version = '2023-06-01-preview',
                temperature = 0.2,
                chatgpt_log_file_name=gpt4_log_file_name,
            )
    gpt4 = ChatGPTRecordWrapper(**gpt4_kwargs)


def build_llama(model_path):
    global llama_model
    if 'llama' in model_path.lower():
        model_type = 'llama'
    elif 'baichuan' in model_path.lower():
        model_type = 'baichuan'
    else:
        raise NotImplementedError

    llama_log_file_name = os.path.join(agent_data_dir, f'{model_type}_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.txt')

    llama_kwargs = dict(
        model_path=model_path,
        temperature=0.2,
        tokens=300,
        presence_penalty=0.1,
        model_type=model_type,
        save_file = llama_log_file_name
    )
    llama_model = LocalLLama(**llama_kwargs)
