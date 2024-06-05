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


def eval_conversation_main(env_path, env_root):
    conv_path = os.path.join(env_path, 'conversation')
    batch_convos = []
    for unit_conv in os.listdir(conv_path):
        unit_task = pkl.load(open(os.path.join(conv_path, unit_conv), 'rb'))
        batch_convos.append(unit_task)
    task = batch_convos[0][0]
    performer_name = task['performer']
    main_persona = Persona(performer_name, os.path.join(env_root, task['env_name'], 'personas', performer_name))
    main_persona.scratch.curr_time = datetime(year=2023, month=2, day=13)
    all_convos, all_summaries = new_convo(task, main_persona, batch_convos)
    for convo in all_convos:
        eval_convo_dummy(task, convo)


def eval_conversation_main_v2(env_path, env_root, result_dir, llm_model, agent, re_run=False):
    conv_path = os.path.join(env_path, 'conversation')
    this_result_dir = os.path.join(result_dir, '_'.join([llm_model + agent]))
    os.makedirs(this_result_dir, exist_ok=True)

    batch_convos = []
    for unit_conv in os.listdir(conv_path):
        unit_task = pkl.load(open(os.path.join(conv_path, unit_conv), 'rb'))
        batch_convos.append(unit_task)
    task = batch_convos[0][0]
    performer_name = task['performer']
    main_persona = Persona(performer_name, os.path.join(env_root, task['env_name'], 'personas', performer_name))
    main_persona.scratch.curr_time = datetime(year=2023, month=2, day=13)
    # print(os.path.join(this_result_dir, 'all_result.pkl'))
    # exit(0)
    if args.re_gen or not os.path.exists(os.path.join(this_result_dir, 'all_result.pkl')):
        # print(this_result_dir)
        # all_convos, all_summaries, planner = new_convo(task, main_persona, batch_convos)
        all_convos, all_summaries, planner = flex_convo_v2(task, main_persona, batch_convos, llm_model=llm_model, is_naive=args.naive)
        results = [all_convos, all_summaries, planner]
        pkl.dump(results, open(os.path.join(this_result_dir, 'all_result.pkl'), 'wb'))
    else:
        results = pkl.load(open(os.path.join(this_result_dir, 'all_result.pkl'), 'rb'))

    all_convos, all_summaries, planner = results

    mean_scores = []
    all_scores = [[] for _ in range(4)]
    if args.eval:
        if re_run or not os.path.exists(os.path.join(this_result_dir, 'scores.pkl')):
            print(this_result_dir)
            all_scores = llm_eval_convos(task, all_convos, all_summaries)
            pkl.dump(all_scores, open(os.path.join(this_result_dir, 'scores.pkl'), 'wb'))
        else:
            all_scores = pkl.load(open(os.path.join(this_result_dir, 'scores.pkl'), 'rb'))
        mean_scores = [np.mean(s) if len(s) > 0 else 0 for s in all_scores]
        print(task['env_name'])
        print(mean_scores)

    f = open(os.path.join(this_result_dir, 'convs.txt'), 'w', encoding='utf-8')
    for i, conv in enumerate(all_convos):
        for (sp, ut) in conv:
            f.write(f"{sp}: {ut}\n")
        f.write("\n")
        f.write(all_summaries[i] + '\n')
        if len(all_scores[0]) > i:
            f.write('\n' + str([all_scores[j][i] for j in range(4)]))
        f.write("\n\n\n")

    stats = [len(all_scores[0])]

    return task['env_name'], mean_scores, stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default='./data/dumped_data')
    parser.add_argument('-s', '--result_root', default='./data/results_1126')
    parser.add_argument('-m', '--model', default='gpt-35-turbo')
    parser.add_argument('--model_path', default=None)
    parser.add_argument('-a', '--agent', default='new')
    parser.add_argument('--re_gen', default=False, action='store_true')
    parser.add_argument('--re_run', default=False, action='store_true')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--naive', default=False, action='store_true')
    parser.add_argument('--eval_only', default=False, action='store_true')
    parser.add_argument('-r', '--env_root', default='../../environment/frontend_server/storage')
    # print(os.listdir('../../'))

    args = parser.parse_args()

    eval_path = args.path
    env_root = args.env_root
    result_root = args.result_root
    llm_model = args.model

    utils.build_gpt4()
    if llm_model == 'gpt4':
        utils.build_gpt4()
    elif not args.eval_only and ('local' in llm_model or 'llama' in llm_model):
        utils.build_llama(args.model_path)

    all_scores = {}
    micro_ave_scores = [0] * 4
    n_total = 0
    for sim in os.listdir(eval_path):
        # eval_conversation_main(os.path.join(eval_path, sim), env_root)
        task_name, scores, stats = eval_conversation_main_v2(os.path.join(eval_path, sim), env_root, os.path.join(result_root, sim), llm_model, args.agent, args.re_run)
        # all_scores.append(scores)
        all_scores[task_name] = scores + stats
        n_conv = stats[0]
        for i, score in enumerate(scores):
            micro_ave_scores[i] += score * n_conv if score is not np.nan else 0
        n_total += n_conv
    micro_ave_scores = [s / n_total for s in micro_ave_scores]

    for task_name in sorted(all_scores, key=lambda x: int(x.split('_')[-1])):
        print(task_name + '\t' + '\t'.join([str(s) for s in all_scores[task_name]]))
    print('micro_mean' + '\t' + '\t'.join([str(s) for s in micro_ave_scores]))
