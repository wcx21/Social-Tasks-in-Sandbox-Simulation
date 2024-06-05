import shutil
import os
import argparse
import pickle as pkl
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

worker_num = list(range(0, 20))

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--src')
parser.add_argument('-d', '--dst')
parser.add_argument('-m', '--model')
parser.add_argument('-c', '--max_count', default=1, type=int)

args = parser.parse_args()
dst = args.dst
# model = args.model
max_count = args.max_count

if not os.path.exists(dst):
    os.makedirs(dst)

collected_sims = defaultdict(lambda: defaultdict(int))


for wn in tqdm(worker_num):
    data_path = f"{args.src}/ga_worker{wn}/reverie/backend_server/data/dumped_data"
    sims = os.listdir(data_path)

    for sim in sims:
        print(sim)
        sim_tokens = sim.split('_')
        if 'party' in sim:
            sim_name_token_len = 4
        elif 'ask' in sim:
            sim_name_token_len = 6
        else:
            sim_name_token_len = 5
        sim_name = '_'.join(sim_tokens[:sim_name_token_len]).replace('base', 'sim').replace('1109', '1124')
        conv_path = os.path.join(data_path, sim, 'conversation')
        if not os.path.exists(conv_path):
            continue
        for conv_file in os.listdir(conv_path):
            conv_data = pkl.load(open(os.path.join(conv_path, conv_file), 'rb'))
            task, maze, p1, p2 = conv_data

            performer_name = task['performer']
            guest_name = p1.name if p1.name != performer_name else p2.name

            time_valid = (p1.scratch.curr_time < datetime(2023, 2, 14))
            capacity_valid = collected_sims[sim_name][guest_name] < max_count
            type_valid = task['task_type'] != 'Personal appointment' or \
                         (p1.name in task['task']['target'] + [performer_name] and p2.name in task['task']['target'] + [performer_name])
            # (p1.name in task['task']['goal'][0]['required_character'] and p2.name in task['task']['goal'][0]['required_character']) or \


            if time_valid and capacity_valid and type_valid:
                target_conv_path = os.path.join(dst, sim_name, 'conversation')
                os.makedirs(target_conv_path, exist_ok=True)
                shutil.copy(os.path.join(conv_path, conv_file), os.path.join(target_conv_path, conv_file))
                collected_sims[sim_name][guest_name] += 1

values = []
for k, v in collected_sims.items():
    values += list(v.values())
print(sum(values))

