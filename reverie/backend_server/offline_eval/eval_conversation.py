import sys
import os
import tqdm
import numpy as np

from persona.cognitive_modules.converse import agent_chat_v2
from persona.cognitive_modules.plan import generate_convo_summary
from persona.target_driven_planning import run_gpt_prompt as tgp_run_gpt_prompt
from persona.target_driven_planning.main_plan import TargetDrivenPlanner

from persona.cognitive_modules.retrieve import new_retrieve
from persona.cognitive_modules.converse import generate_summarize_agent_relationship, generate_one_utterance


def llm_eval_convo(task, convo):
    pass


def eval_convo_dummy(task, convo, *args, **kwargs):
    with open('tmp.txt', 'a+') as f:
        f.write('\n'.join([': '.join([term for term in l]) for l in convo]))
        f.write('\n')


def llm_eval_party(task, conversation, summary):
    try:
        conv_scores, _conv_params = tgp_run_gpt_prompt.gpt_eval_conversation_party(task, conversation)
        print('error with gpt-35')
    except:
        conv_scores, _conv_params = tgp_run_gpt_prompt.gpt_eval_conversation_party(task, conversation, engine='gpt4')

    # conv_scores, _conv_params = tgp_run_gpt_prompt.gpt_eval_conversation_party(task, conversation)
    summary_scores, _summary_params = tgp_run_gpt_prompt.gpt_eval_summary_party(task, summary)

    print(conv_scores, summary_scores)

    rigorous_conv_score = np.product(conv_scores[:3])
    partial_conv_score = np.mean(conv_scores[:3])
    rigorous_summary_score = np.product(summary_scores[:3])
    partial_summary_score = np.mean(summary_scores[:3])
    return rigorous_conv_score, partial_conv_score, rigorous_summary_score, partial_summary_score


def llm_eval_appointment(task, conversation, summary):
    main_persona_name = task['performer']
    bg_persona_name = [speaker for speaker, words in conversation if speaker != main_persona_name][0]
    required_person = task['task']['target'][0]
    if required_person != bg_persona_name:
        return None

    try:
        conv_scores, _conv_params = tgp_run_gpt_prompt.gpt_eval_conversation_appointment(task, conversation)
        print('error with gpt-35')
    except:
        conv_scores, _conv_params = tgp_run_gpt_prompt.gpt_eval_conversation_appointment(task, conversation, engine='gpt4')

    # conv_scores, _conv_params = tgp_run_gpt_prompt.gpt_eval_conversation_appointment(task, conversation)
    summary_scores, _summary_params = tgp_run_gpt_prompt.gpt_eval_summary_appointment(task, summary)

    rigorous_conv_score = np.product(conv_scores[:3])
    partial_conv_score = np.mean(conv_scores[:3])
    rigorous_summary_score = np.product(summary_scores[:3])
    partial_summary_score = np.mean(summary_scores[:3])
    return rigorous_conv_score, partial_conv_score, rigorous_summary_score, partial_summary_score


def llm_eval_find_partner(task, conversation, summary):
    try:
        conv_scores, _conv_params = tgp_run_gpt_prompt.gpt_eval_conversation_find_partner(task, conversation)
        print('error with gpt-35')
    except:
        conv_scores, _conv_params = tgp_run_gpt_prompt.gpt_eval_conversation_find_partner(task, conversation, engine='gpt4')

    summary_scores, _summary_params = tgp_run_gpt_prompt.gpt_eval_summary_find_partner(task, summary)

    rigorous_conv_score = np.product(conv_scores[:3])
    partial_conv_score = np.mean(conv_scores[:3])
    rigorous_summary_score = np.product(summary_scores[:3])
    partial_summary_score = np.mean(summary_scores[:3])
    return rigorous_conv_score, partial_conv_score, rigorous_summary_score, partial_summary_score


def llm_eval_online_meeting(task, conversation, summary):
    try:
        conv_scores, _conv_params = tgp_run_gpt_prompt.gpt_eval_conversation_online_meeting(task, conversation)
        print('error with gpt-35')
    except:
        conv_scores, _conv_params = tgp_run_gpt_prompt.gpt_eval_conversation_online_meeting(task, conversation, engine='gpt4')

    # conv_scores, _conv_params = tgp_run_gpt_prompt.gpt_eval_conversation_online_meeting(task, conversation)
    summary_scores, _summary_params = tgp_run_gpt_prompt.gpt_eval_summary_online_meeting(task, summary)

    print(conv_scores, summary_scores)

    rigorous_conv_score = np.product(conv_scores[:2])
    partial_conv_score = np.mean(conv_scores[:2])
    rigorous_summary_score = np.product(summary_scores[:2])
    partial_summary_score = np.mean(summary_scores[:2])
    return rigorous_conv_score, partial_conv_score, rigorous_summary_score, partial_summary_score


def llm_eval_ask_for_help(task, conversation, summary):
    try:
        conv_scores, _conv_params = tgp_run_gpt_prompt.gpt_eval_conversation_ask_for_help(task, conversation)
        print('error with gpt-35')
    except:
        conv_scores, _conv_params = tgp_run_gpt_prompt.gpt_eval_conversation_ask_for_help(task, conversation, engine='gpt4')

    # conv_scores, _conv_params = tgp_run_gpt_prompt.gpt_eval_conversation_ask_for_help(task, conversation)
    summary_scores, _summary_params = tgp_run_gpt_prompt.gpt_eval_summary_ask_for_help(task, summary)

    print(conv_scores, summary_scores)

    rigorous_conv_score = np.product(conv_scores[:4])
    partial_conv_score = np.mean(conv_scores[:4])
    rigorous_summary_score = np.product(summary_scores[:4])
    partial_summary_score = np.mean(summary_scores[:4])
    return rigorous_conv_score, partial_conv_score, rigorous_summary_score, partial_summary_score


def llm_eval_convos(task, all_convos, all_summaries):
    allowed_task_type = ['Party', 'Public Activity', 'Personal appointment', 'Find partner', 'Online meeting', 'Ask for others']

    task_type = task['task_type']
    task_type_to_eval_func = {
        'Party': llm_eval_party,
        'Public Activity': llm_eval_party,
        'Personal appointment': llm_eval_appointment,
        'Find partner': llm_eval_find_partner,
        'Online meeting': llm_eval_online_meeting,
        'Ask for others': llm_eval_ask_for_help
    }
    total_scores = [[] for _ in range(4)]
    if task_type in task_type_to_eval_func:
        eval_func = task_type_to_eval_func[task_type]
        for i, (conv, summary) in enumerate(zip(all_convos, all_summaries)):
            if len(conv) <= 1:
                print(f"skipped conv {task['env_name']} {conv}")
                scores = [0, 0, 0, 0]
            else:
                scores = eval_func(task, conv, summary)
            if scores is not None:
                for j in range(len(scores)):
                    total_scores[j].append(scores[j])
    return total_scores
