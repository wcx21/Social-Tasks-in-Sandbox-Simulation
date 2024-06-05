"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: converse.py
Description: An extra cognitive module for generating conversations. 
"""
import math
import os
import pickle
import sys
import datetime
import random
sys.path.append('../')

from global_methods import *

from persona.memory_structures.spatial_memory import *
from persona.memory_structures.associative_memory import *
from persona.memory_structures.scratch import *
from persona.cognitive_modules.retrieve import *
from persona.prompt_template.run_gpt_prompt import *

def generate_agent_chat_summarize_ideas(init_persona,
                                        target_persona,
                                        retrieved,
                                        curr_context):
    all_embedding_keys = list()
    for key, val in retrieved.items():
        for i in val:
            all_embedding_keys += [i.embedding_key]
    all_embedding_key_str =""
    for i in all_embedding_keys:
        all_embedding_key_str += f"{i}\n"

    try:
        summarized_idea = run_gpt_prompt_agent_chat_summarize_ideas(init_persona,
                                                                    target_persona, all_embedding_key_str,
                                                                    curr_context)[0]
    except:
        summarized_idea = ""
    return summarized_idea


def generate_summarize_agent_relationship(init_persona,
                                          target_persona,
                                          retrieved):
    all_embedding_keys = list()
    for key, val in retrieved.items():
        for i in val:
            all_embedding_keys += [i.embedding_key]
    all_embedding_key_str =""
    for i in all_embedding_keys:
        all_embedding_key_str += f"{i}\n"

    summarized_relationship = run_gpt_prompt_agent_chat_summarize_relationship(
        init_persona, target_persona,
        all_embedding_key_str)[0]
    return summarized_relationship


def generate_agent_chat(maze,
                        init_persona,
                        target_persona,
                        curr_context,
                        init_summ_idea,
                        target_summ_idea):
    summarized_idea = run_gpt_prompt_agent_chat(maze,
                                                init_persona,
                                                target_persona,
                                                curr_context,
                                                init_summ_idea,
                                                target_summ_idea)[0]
    for i in summarized_idea:
        print (i)
    return summarized_idea


def agent_chat_v1(maze, init_persona, target_persona):
    # Chat version optimized for speed via batch generation
    curr_context = (f"{init_persona.scratch.name} " +
                    f"was {init_persona.scratch.act_description} " +
                    f"when {init_persona.scratch.name} " +
                    f"saw {target_persona.scratch.name} " +
                    f"in the middle of {target_persona.scratch.act_description}.\n")
    curr_context += (f"{init_persona.scratch.name} " +
                     f"is thinking of initating a conversation with " +
                     f"{target_persona.scratch.name}.")

    summarized_ideas = []
    part_pairs = [(init_persona, target_persona),
                  (target_persona, init_persona)]
    for p_1, p_2 in part_pairs:
        focal_points = [f"{p_2.scratch.name}"]
        retrieved = new_retrieve(p_1, focal_points, 50)
        relationship = generate_summarize_agent_relationship(p_1, p_2, retrieved)
        focal_points = [f"{relationship}",
                        f"{p_2.scratch.name} is {p_2.scratch.act_description}"]
        retrieved = new_retrieve(p_1, focal_points, 25)
        summarized_idea = generate_agent_chat_summarize_ideas(p_1, p_2, retrieved, curr_context)
        summarized_ideas += [summarized_idea]

    return generate_agent_chat(maze, init_persona, target_persona,
                               curr_context,
                               summarized_ideas[0],
                               summarized_ideas[1])


def generate_one_utterance(maze, init_persona, target_persona, retrieved, curr_chat):
    performer_name = get_var("performer_name")

    if init_persona.name == performer_name and get_var("performer_llm_model") is not None:
        llm_model = get_var("performer_llm_model")
    else:
        llm_model = 'gpt-35-turbo'
    print(performer_name, init_persona.name, llm_model)
    # Chat version optimized for speed via batch generation
    curr_context = (f"{init_persona.scratch.name} " +
                    f"was {init_persona.scratch.act_description} " +
                    f"when {init_persona.scratch.name} " +
                    f"saw {target_persona.scratch.name} " +
                    f"in the middle of {target_persona.scratch.act_description}.\n")
    curr_context += (f"{init_persona.scratch.name} " +
                     f"is initiating a conversation with " +
                     f"{target_persona.scratch.name}.")

    print ("July 23 5")
    x = run_gpt_generate_iterative_chat_utt(maze, init_persona, target_persona, retrieved, curr_context, curr_chat, engine=llm_model)[0]

    print ("July 23 6")

    print ("adshfoa;khdf;fajslkfjald;sdfa HERE", x)

    return x["utterance"], x["end"]


# def agent_chat_v2(maze, init_persona, target_persona):
#     task = get_var('task')
#     if task is not None:
#         performer = task['performer']
#         if performer in [init_persona.name, target_persona.name]:
#             dump_data_dir = get_var('dump_data_dir')
#             convo_dump_dir = os.path.join(dump_data_dir, 'conversation')
#             os.makedirs(convo_dump_dir, exist_ok=True)
#             convo_file = os.path.join(convo_dump_dir, f'{datetime.datetime.now().strftime("%m%d_%H%M%S")}_{init_persona.name}_{target_persona.name}.pkl')
#             data = [task, maze, init_persona, target_persona]
#             pickle.dump(data, open(convo_file, 'wb'))
#
#     curr_chat = []
#     print ("July 23")
#
#     for i in range(8):
#         focal_points = [f"{target_persona.scratch.name}"]
#         retrieved = new_retrieve(init_persona, focal_points, 35)
#         relationship = generate_summarize_agent_relationship(init_persona, target_persona, retrieved)
#         print ("-------- relationshopadsjfhkalsdjf", relationship)
#         last_chat = ""
#         for i in curr_chat[-4:]:
#             last_chat += ": ".join(i) + "\n"
#         if last_chat:
#             focal_points = [f"{relationship}",
#                             f"{target_persona.scratch.name} is {target_persona.scratch.act_description}",
#                             last_chat]
#         else:
#             focal_points = [f"{relationship}",
#                             f"{target_persona.scratch.name} is {target_persona.scratch.act_description}"]
#         # retrieved = new_retrieve(init_persona, focal_points, 15)
#         retrieved = new_retrieve(init_persona, focal_points, 10)
#         utt, end = generate_one_utterance(maze, init_persona, target_persona, retrieved, curr_chat)
#
#         curr_chat += [[init_persona.scratch.name, utt]]
#         if end:
#             break
#
#
#         focal_points = [f"{init_persona.scratch.name}"]
#         # retrieved = new_retrieve(target_persona, focal_points, 50)
#         retrieved = new_retrieve(target_persona, focal_points, 35)
#         relationship = generate_summarize_agent_relationship(target_persona, init_persona, retrieved)
#         print ("-------- relationshopadsjfhkalsdjf", relationship)
#         last_chat = ""
#         for i in curr_chat[-4:]:
#             last_chat += ": ".join(i) + "\n"
#         if last_chat:
#             focal_points = [f"{relationship}",
#                             f"{init_persona.scratch.name} is {init_persona.scratch.act_description}",
#                             last_chat]
#         else:
#             focal_points = [f"{relationship}",
#                             f"{init_persona.scratch.name} is {init_persona.scratch.act_description}"]
#         # retrieved = new_retrieve(target_persona, focal_points, 15)
#         retrieved = new_retrieve(target_persona, focal_points, 10)
#         utt, end = generate_one_utterance(maze, target_persona, init_persona, retrieved, curr_chat)
#
#         curr_chat += [[target_persona.scratch.name, utt]]
#         if end:
#             break
#
#     print ("July 23 PU")
#     for row in curr_chat:
#         print (row)
#     print ("July 23 FIN")
#
#     return curr_chat


def agent_chat_v2(maze, init_persona, target_persona):
    task = get_var('task')
    if task is not None:
        performer = task['performer']
        if performer in [init_persona.name, target_persona.name]:
            dump_data_dir = get_var('dump_data_dir')
            convo_dump_dir = os.path.join(dump_data_dir, 'conversation')
            os.makedirs(convo_dump_dir, exist_ok=True)
            convo_file = os.path.join(convo_dump_dir, f'{datetime.datetime.now().strftime("%m%d_%H%M%S")}_{init_persona.name}_{target_persona.name}.pkl')
            data = [task, maze, init_persona, target_persona]
            pickle.dump(data, open(convo_file, 'wb'))

    curr_chat = []
    print ("July 23")
    performer_name = get_var("performer_name")
    task_planner = get_var("task_planner")

    p1_talk_func = generate_one_utterance if (init_persona.name != performer_name or task_planner is None) else task_planner.gen_next_convo_utterance
    p2_talk_func = generate_one_utterance if (target_persona.name != performer_name or task_planner is None) else task_planner.gen_next_convo_utterance

    chat_iters = [(init_persona, target_persona, p1_talk_func), (target_persona, init_persona, p2_talk_func)]
    print("====Start Unified Chat====")
    max_round = 8

    for i in range(max_round):
        conv_end = False
        for speaker, listener, talk_func in chat_iters:
            focal_points = [f"{listener.scratch.name}"]
            retrieved = new_retrieve(speaker, focal_points, 15)
            relationship = generate_summarize_agent_relationship(speaker, listener, retrieved)
            print("-------- relationship ---------", relationship)
            last_chat = ""
            for i in curr_chat[-4:]:
                last_chat += ": ".join(i) + "\n"
            if last_chat:
                focal_points = [f"{relationship}",
                                f"{listener.scratch.name} is {listener.scratch.act_description}",
                                last_chat]
            else:
                focal_points = [f"{relationship}",
                                f"{listener.scratch.name} is {listener.scratch.act_description}"]
            # retrieved = new_retrieve(init_persona, focal_points, 15)
            r_num = 5 if last_chat else 7
            retrieved = new_retrieve(speaker, focal_points, r_num)
            utt, end = talk_func(maze, speaker, listener, retrieved, curr_chat)

            curr_chat += [[speaker.scratch.name, utt]]
            if end:
                conv_end = True
                break
        if conv_end:
            break

    return curr_chat


def generate_summarize_ideas(persona, nodes, question):
    statements = ""
    for n in nodes:
        statements += f"{n.embedding_key}\n"
    summarized_idea = run_gpt_prompt_summarize_ideas(persona, statements, question)[0]
    return summarized_idea


def generate_next_line(persona, interlocutor_desc, curr_convo, summarized_idea):
    # Original chat -- line by line generation
    prev_convo = ""
    for row in curr_convo:
        prev_convo += f'{row[0]}: {row[1]}\n'

    next_line = run_gpt_prompt_generate_next_convo_line(persona,
                                                        interlocutor_desc,
                                                        prev_convo,
                                                        summarized_idea)[0]
    return next_line


def generate_inner_thought(persona, whisper):
    inner_thought = run_gpt_prompt_generate_whisper_inner_thought(persona, whisper)[0]
    return inner_thought

def generate_action_event_triple(act_desp, persona):
    """TODO

    INPUT:
      act_desp: the description of the action (e.g., "sleeping")
      persona: The Persona class instance
    OUTPUT:
      a string of emoji that translates action description.
    EXAMPLE OUTPUT:
      "🧈🍞"
    """
    if debug: print ("GNS FUNCTION: <generate_action_event_triple>")
    return run_gpt_prompt_event_triple(act_desp, persona)[0]


def generate_poig_score(persona, event_type, description):
    if debug: print ("GNS FUNCTION: <generate_poig_score>")

    if "is idle" in description:
        return 1

    if event_type == "event" or event_type == "thought":
        return run_gpt_prompt_event_poignancy(persona, description)[0]
    elif event_type == "chat":
        return run_gpt_prompt_chat_poignancy(persona,
                                             persona.scratch.act_description)[0]


def load_history_via_whisper(personas, whispers):
    for count, row in enumerate(whispers):
        persona = personas[row[0]]
        whisper = row[1]

        thought = generate_inner_thought(persona, whisper)

        created = persona.scratch.curr_time
        expiration = persona.scratch.curr_time + datetime.timedelta(days=30)
        s, p, o = generate_action_event_triple(thought, persona)
        keywords = set([s, p, o])
        thought_poignancy = generate_poig_score(persona, "event", whisper)
        thought_embedding_pair = (thought, get_embedding(thought))
        persona.a_mem.add_thought(created, expiration, s, p, o,
                                  thought, keywords, thought_poignancy,
                                  thought_embedding_pair, None)


def open_convo_session(persona, convo_mode):
    if convo_mode == "analysis":
        curr_convo = []
        interlocutor_desc = "Interviewer"

        while True:
            line = input("Enter Input: ")
            if line == "end_convo":
                break

            if int(run_gpt_generate_safety_score(persona, line)[0]) >= 8:
                print (f"{persona.scratch.name} is a computational agent, and as such, it may be inappropriate to attribute human agency to the agent in your communication.")

            else:
                retrieved = new_retrieve(persona, [line], 50)[line]
                summarized_idea = generate_summarize_ideas(persona, retrieved, line)
                curr_convo += [[interlocutor_desc, line]]

                next_line = generate_next_line(persona, interlocutor_desc, curr_convo, summarized_idea)
                curr_convo += [[persona.scratch.name, next_line]]


    elif convo_mode == "whisper":
        whisper = input("Enter Input: ")
        thought = generate_inner_thought(persona, whisper)

        created = persona.scratch.curr_time
        expiration = persona.scratch.curr_time + datetime.timedelta(days=30)
        s, p, o = generate_action_event_triple(thought, persona)
        keywords = set([s, p, o])
        thought_poignancy = generate_poig_score(persona, "event", whisper)
        thought_embedding_pair = (thought, get_embedding(thought))
        persona.a_mem.add_thought(created, expiration, s, p, o,
                                  thought, keywords, thought_poignancy,
                                  thought_embedding_pair, None)
































