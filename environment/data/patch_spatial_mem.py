import json
import os
import shutil

import constants
import numpy as np
import random
import argparse
from datetime import datetime, timedelta
random.seed(1234)

task_template_dir = './meta_tasks_1029'
storage_root = '../frontend_server/base_1124_20ver'
base_env_name = 'clipped_the_ville_n25'
# gen_env_batch = 'base_1109_'
gen_env_batch = 'base_1124_'


def time_to_str(time: datetime):
    return time.strftime("%Y-%m-%d %H:%M")


def str_to_time(time_str: str):
    return datetime.strptime(time_str, "%Y-%m-%d %H:%M")


TASK_TYPES = ['Party', 'Public Activity', 'Personal appointment', 'Find partner', 'Online meeting', 'Ask for others']


def patch_task_env(env_name, meta_task):
    new_env_path = os.path.join(storage_root, env_name)
    assert os.path.exists(new_env_path), new_env_path

    all_chars = os.listdir(os.path.join(new_env_path, 'personas'))
    performer = meta_task['performer']
    if 'shared_spatial_mem' in meta_task:
        print(env_name, "shared_sp")
        for cn in all_chars:
            p_path = os.path.join(new_env_path, 'personas')
            s_mem = json.load(open(os.path.join(p_path, 'bootstrap_memory/spatial_memory.json')))
            s_mem.update(meta_task['shared_spatial_mem'])
            json.dump(s_mem, open(os.path.join(p_path, 'bootstrap_memory/spatial_memory.json'), 'w'), indent=4)

    if 'required_spatial_mem' in meta_task:
        print(env_name, "required_sp")
        p_path = os.path.join(new_env_path, 'personas', performer)
        s_mem = json.load(open(os.path.join(p_path, 'bootstrap_memory/spatial_memory.json')))
        s_mem.update(meta_task['required_spatial_mem'])
        json.dump(s_mem, open(os.path.join(p_path, 'bootstrap_memory/spatial_memory.json'), 'w'), indent=4)


def patch_all_tasks(args):
    all_meta_tasks = json.load(open(os.path.join(args.task_root, args.task_file)))
    for i, task in enumerate(all_meta_tasks):
        env_name = gen_env_batch + task['task_type'].lower().replace(' ', '_') + '_' + str(i)
        patch_task_env(env_name, task)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_root', default=task_template_dir)
    parser.add_argument('-t', '--task_file', default='meta_task_extend4.json')
    args = parser.parse_args()
    patch_all_tasks(args)
