import json
import os
import shutil

import constants
import numpy as np
import random
import argparse
from datetime import datetime, timedelta
random.seed(1234)

task_template_dir = './meta_tasks_240208'
storage_root = '../frontend_server/storage'
base_env_name = 'clipped_the_ville_n25'
# gen_env_batch = 'base_1109_'
gen_env_batch = 'base_0209_'


def time_to_str(time: datetime):
    return time.strftime("%Y-%m-%d %H:%M")


def str_to_time(time_str: str):
    return datetime.strptime(time_str, "%Y-%m-%d %H:%M")


TASK_TYPES = ['Public Activity', 'Personal appointment', 'Find partner', 'Online meeting', 'Ask for others']


# def gen_description(task):
#     desc =


def generate_tasks_v3(meta_tasks, n_time_sample=1, people_sample_range=(10, 12), n_people_sample_times=1):
    char_name_list = [cn for cn in constants.character_meta if cn != 'shared']
    # n_time_sample = np.ceil(len(constants.TIME_CHOICE) * time_sample).astype(np.int32)

    curr_tasks = meta_tasks.copy()

    # time

    new_curr_tasks = []
    for _ in range(n_time_sample):
        for curr_task in curr_tasks:
            new_task = curr_task.copy()
            for subgoal in new_task['task']['goal']:
                if subgoal['start_time'] == '#TBD':
                    st = random.choices(constants.TIME_CHOICE, k=1)[0]
                    new_task['goal'] += f' on {time_to_str(st)}'
                    subgoal['start_time'] = time_to_str(st)
                    # print(str_to_time(time_to_str(st)))
            new_curr_tasks.append(new_task)
    curr_tasks = new_curr_tasks

    # num target people
    new_curr_tasks = []
    for curr_task in curr_tasks:
        for _ in range(n_people_sample_times):
            new_task = curr_task.copy()
            target_sample = random.randint(*people_sample_range)
            total_char_num = target_sample + 1
            new_task['n_characters'] = total_char_num
            required_char = [curr_task['performer']]
            # if 'target' in new_task['task']:
            #     required_char += new_task['task']['target']
            # not in tasks_dlc.json
            for subgoal in new_task['task']['goal']:
                if "required_character" in subgoal:
                    required_char += subgoal['required_character']

            required_char = list(set(required_char))
            remaining_char_num = total_char_num - len(required_char)
            remaining_char_list = char_name_list.copy()
            for c in required_char:
                if c in remaining_char_list:
                    remaining_char_list.remove(c)

            other_chars = random.sample(remaining_char_list, k=remaining_char_num)
            all_chars = required_char + other_chars
            new_task['all_chars'] = all_chars
            if new_task['task_type'] == 'Public Activity':
                new_task['task']['target_num'] = int(np.ceil(target_sample / 2) + 1)
            if new_task['task_type'] == 'Find partner':
                new_task['task']['target_num'] = int(np.ceil(target_sample / 3) + 1)
            new_curr_tasks.append(new_task)

    curr_tasks = new_curr_tasks

    for task_id, task in enumerate(curr_tasks):
        for k in ['goal', 'FPG', 'Description']:
            task[k] = task[k].replace('#performer', task['performer'])
            task[k] = task[k].replace('#target_num', str(task['task']['target_num']))
            if 'target' in task['task'] and len(task['task']['target']) > 0:
                task[k] = task[k].replace('#target_0', str(task['task']['target'][0]))
            for i, subgoal in enumerate(task['task']['goal']):
                if subgoal['start_time'] != 'N/A':
                    task[k] = task[k].replace(f'#time_{i}', subgoal['start_time'])
                if subgoal['location'] != 'N/A':
                    task[k] = task[k].replace(f'#location_{i}', subgoal['location'])
        all_chars = task['all_chars']
        task_env_name = gen_env_batch + task['task_type'].lower().replace(' ', '_') + '_' + str(task_id)
        task['env_name'] = task_env_name
        custom_info = {
            'scratch': {
                'currently': ' ' + task['FPG'] + ('. ' if not task['FPG'].endswith('.') else ' ') + task['Description']
            }
        }
        if 'required_spatial_mem' in task:
            custom_info['spatial_memory'] = task['required_spatial_mem']

        # target_char = []
        # for subgoal in task['task']['goal']:
        #     if "required_character" in subgoal:
        #         target_char += subgoal['required_character']
        # if task['performer'] in target_char:
        #     target_char.remove(task['performer'])
        # if len(target_char) > 0:
        #     target_spatial_info = {}
        #     for tar in target_char:

        inject_info = (task['performer'], custom_info)
        gen_env_from_cn(task_env_name, all_chars, inject_info, shared_spatial_mem=task.get('shared_spatial_mem'))

    return curr_tasks


def gen_env_from_cn(env_name, all_chars, inject_info=None, shared_spatial_mem=None):
    base_env_path = os.path.join(storage_root, base_env_name)
    assert os.path.exists(base_env_path), base_env_path + 'not exists'

    new_env_path = os.path.join(storage_root, env_name)
    if os.path.exists(new_env_path):
        shutil.rmtree(new_env_path)
    os.mkdir(new_env_path)
    os.mkdir(os.path.join(new_env_path, 'environment'))
    os.mkdir(os.path.join(new_env_path, 'personas'))
    os.mkdir(os.path.join(new_env_path, 'reverie'))

    env_meta = json.load(open(os.path.join(base_env_path, 'environment', '0.json')))
    env_meta = {k: env_meta[k] for k in all_chars}
    json.dump(env_meta, open(os.path.join(new_env_path, 'environment', '0.json'), 'w', encoding='utf-8'), indent=4)

    reverie_meta = json.load(open(os.path.join(base_env_path, 'reverie', 'meta.json')))
    reverie_meta['persona_names'] = all_chars
    reverie_meta["fork_sim_code"] = base_env_name
    json.dump(reverie_meta, open(os.path.join(new_env_path, 'reverie', 'meta.json'), 'w', encoding='utf-8'), indent=4)

    for cn in all_chars:
        shutil.copytree(os.path.join(base_env_path, 'personas', cn), os.path.join(new_env_path, 'personas', cn))

    if inject_info is not None:
        cn, info = inject_info
        if 'scratch' in info:
            scratch = json.load(open(os.path.join(new_env_path, 'personas', cn, 'bootstrap_memory/scratch.json')))
            for k, v in info['scratch'].items():
                scratch[k] += v
            json.dump(scratch, open(os.path.join(
                new_env_path, 'personas', cn, 'bootstrap_memory/scratch.json'), 'w'), indent=4)

        if 'spatial_memory' in info:
            s_mem = json.load(open(os.path.join(new_env_path, 'personas', cn, 'bootstrap_memory/spatial_memory.json')))
            for k, v in info['spatial_memory'].items():
                if k in s_mem['the Ville']:
                    s_mem['the Ville'][k].update(v)
                else:
                    s_mem['the Ville'][k] = v
            json.dump(s_mem, open(os.path.join(
                new_env_path, 'personas', cn, 'bootstrap_memory/spatial_memory.json'), 'w'), indent=4)

    if shared_spatial_mem is not None:
        print(env_name, "shared_sp")
        for cn in all_chars:
            p_path = os.path.join(new_env_path, 'personas', cn)
            s_mem = json.load(open(os.path.join(p_path, 'bootstrap_memory/spatial_memory.json')))
            s_mem['the Ville'].update(shared_spatial_mem)
            json.dump(s_mem, open(os.path.join(p_path, 'bootstrap_memory/spatial_memory.json'), 'w'), indent=4)


def generate_all_tasks(args):
    all_meta_tasks = json.load(open(os.path.join(args.task_root, args.task_file)))
    all_tasks = generate_tasks_v3(all_meta_tasks)

    json.dump(all_tasks, open(os.path.join('task_0905', 'tasks_dlc.json'), 'w', encoding='utf-8'), indent=4)
    print(f"in total {len(all_tasks)} tasks")


def count_tasks():
    for task_file in os.listdir(args.task_root):
        task_path = os.path.join(args.task_root, task_file)
        print(task_file, len(json.load(open(task_path))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_root', default=task_template_dir)
    # parser.add_argument('-t', '--task_file', default='meta_task_short.json')
    # parser.add_argument('-t', '--task_file', default='meta_task_extend.json')
    parser.add_argument('-t', '--task_file', default='meta_task_dlc_240208.json')
    args = parser.parse_args()

    generate_all_tasks(args)
    # count_tasks()
