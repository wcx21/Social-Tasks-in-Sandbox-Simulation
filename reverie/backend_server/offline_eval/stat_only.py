import json
import os
import sys
import argparse
import pickle as pkl
import numpy as np
from datetime import datetime

if 'offline_eval' not in os.listdir('.'):
    os.chdir('..')

sys.path.append('.')
sys.path.append('../')
sys.path.append('../..')

from offline_gen_conversation import new_convo, flex_convo_v2
from eval_conversation import eval_convo_dummy, llm_eval_convos
from persona.persona import Persona
import utils


def stat(args):

    eval_path = args.path
    env_root = args.env_root
    result_root = args.result_root

    n_convo_total = 0
    for sim in os.listdir(eval_path):
        conv_path = os.path.join(eval_path, sim, 'conversation')
        if not os.path.exists(conv_path):
            unit_convs = []
            print(f"err, {sim} no conversation saved")
        else:
            unit_convs = os.listdir(conv_path)

        n_unit_convs = len(unit_convs)
        if 'personal_appointment' in sim:
            n_unit_convs = 1
        print(sim, n_unit_convs)
        n_convo_total += n_unit_convs

    print(f"total: {n_convo_total}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default='./data/offline_eval_full')
    parser.add_argument('-s', '--result_root', default='./data/results_1126')
    parser.add_argument('-a', '--agent', default='new')
    parser.add_argument('--re_run', default=False, action='store_true')
    parser.add_argument('-r', '--env_root', default='../../environment/frontend_server/storage')
    # print(os.listdir('../../'))

    args = parser.parse_args()

    stat(args)

