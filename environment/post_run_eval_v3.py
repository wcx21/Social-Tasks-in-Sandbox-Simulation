import json
import os
import os.path as osp
import sys
import pickle as pkl
import argparse

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))

from post_eval_benchmarks.eval_funcs import eval_main

parser = argparse.ArgumentParser()
parser.add_argument('--sim_root', default='./frontend_server/offline_eval/val')
parser.add_argument('-n', '--sim_name')
parser.add_argument('--task_root', default='data/task_0905')
parser.add_argument('-t', '--task_file', default='tasks_1124.json')


if __name__ == '__main__':
    # traj_file_names = os.listdir(data_root)
    opt = parser.parse_args()
    # sim_root = opt.sim_root
    # sim_name = opt.sim_name
    # sim_name = 'July1_the_ville_isabella_maria_klaus-step-3-20'  # 0.33
    # sim_name = 'full_test_0823_2'  # 0.0
    # sim_name = 'full_test_0825_1'  # 0.0
    # sim_name = 'test_0922_wine_party_1'  # 0.66
    # sim_name = 'test_1009_wine_party_1'  # 0.66
    # sim_name = 'test_1019_wine_party_4'  # 0.66
    # sim_name = 'test_1023_wine_party_2'  # 1.67
    sim_root = opt.sim_root
    task_root = opt.task_root

    tasks = json.load(open(os.path.join(task_root, opt.task_file), encoding='utf-8'))
    # tasks = json.load(open(os.path.join('data/task_0905', 'tasks_1029_minitest.json'), encoding='utf-8'))
    # tasks = json.load(open(os.path.join('data/task_0905', 'tasks_1124.json'), encoding='utf-8'))

    # traj = traj_evaluate_v2.load_traj(sim_root, sim_name)
    rewards = eval_main(tasks, sim_root)
    print(rewards)
