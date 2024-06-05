import sys
import os
import tqdm

from persona.cognitive_modules.converse import agent_chat_v2
from persona.cognitive_modules.plan import generate_convo_summary
from persona.target_driven_planning import run_gpt_prompt as tgp_run_gpt_prompt
from persona.target_driven_planning.main_plan import TargetDrivenPlanner

from persona.cognitive_modules.retrieve import new_retrieve
from persona.cognitive_modules.converse import generate_summarize_agent_relationship, generate_one_utterance
from global_methods import get_var, set_var


def unified_agent_chat(maze, init_persona, target_persona, max_round=8,
                       p1_talk_func=generate_one_utterance, p2_talk_func=generate_one_utterance):
    curr_chat = []
    if p1_talk_func is None:
        p1_talk_func = generate_one_utterance
    if p2_talk_func is None:
        p2_talk_func = generate_one_utterance

    chat_iters = [(init_persona, target_persona, p1_talk_func), (target_persona, init_persona, p2_talk_func)]
    print("====Start Unified Chat====")

    for i in range(max_round):
        conv_end = False
        for speaker, listener, talk_func in chat_iters:
            focal_points = [f"{listener.scratch.name}"]
            retrieved = new_retrieve(speaker, focal_points, 20)
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
            retrieved = new_retrieve(speaker, focal_points, 5)
            print(len(retrieved))
            utt, end = talk_func(maze, speaker, listener, retrieved, curr_chat)

            curr_chat += [[speaker.scratch.name, utt]]
            if end:
                conv_end = True
                break
        if conv_end:
            break

    for row in curr_chat:
        print(row)

    return curr_chat


def standard_convo(task, maze, persona1, persona2):
    convo = unified_agent_chat(maze, persona1, persona2)
    performer, guest = (persona1, persona2) if persona1.name == task['performer'] else (persona2, persona1)
    guest_summary = generate_convo_summary(guest, convo)


def new_convo(task, main_persona, batch_convos):
    all_convos = []
    all_summaries = []

    planner = TargetDrivenPlanner(task, main_persona)
    planner.initial_plan()
    for clip in batch_convos:
        task, maze, persona1, persona2 = clip
        performer, guest = (persona1, persona2) if persona1.name == task['performer'] else (persona2, persona1)


        p1_talk_func = planner.gen_next_convo_utterance if persona1.name == main_persona.name else generate_one_utterance
        p2_talk_func = planner.gen_next_convo_utterance if persona1.name == main_persona.name else generate_one_utterance

        planner.start_convo()
        convo = unified_agent_chat(maze, persona1, persona2, p1_talk_func=p1_talk_func, p2_talk_func=p2_talk_func)
        planner.end_convo()
        guest_summary = generate_convo_summary(guest, convo)

        all_convos.append(convo)
        all_summaries.append(guest_summary)

    return all_convos, all_summaries, planner


def flex_convo_v2(task, main_persona, batch_convos, llm_model='gpt-35-turbo', is_naive=False):
    all_convos = []
    all_summaries = []

    planner = TargetDrivenPlanner(task, main_persona, model=llm_model, is_naive=is_naive)
    planner.initial_plan()
    for clip in tqdm.tqdm(batch_convos):
        task, maze, persona1, persona2 = clip
        performer, guest = (persona1, persona2) if persona1.name == task['performer'] else (persona2, persona1)
        set_var('performer_name', performer.name)
        set_var('performer_llm_model', llm_model)

        # p1_talk_func = planner.gen_next_convo_utterance if (persona1.name == main_persona.name and not is_naive) else generate_one_utterance
        # p2_talk_func = planner.gen_next_convo_utterance if (persona2.name == main_persona.name and not is_naive) else generate_one_utterance
        p1_talk_func = planner.gen_next_convo_utterance if persona1.name == main_persona.name else generate_one_utterance
        p2_talk_func = planner.gen_next_convo_utterance if persona2.name == main_persona.name else generate_one_utterance

        planner.start_convo()
        convo = unified_agent_chat(maze, persona1, persona2, p1_talk_func=p1_talk_func, p2_talk_func=p2_talk_func)
        planner.end_convo()
        guest_summary = generate_convo_summary(guest, convo)

        all_convos.append(convo)
        all_summaries.append(guest_summary)

    return all_convos, all_summaries, planner
