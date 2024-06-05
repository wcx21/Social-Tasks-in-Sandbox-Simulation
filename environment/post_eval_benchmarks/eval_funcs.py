import os
import sys
import traceback

from tqdm import tqdm

from .eval_utils import str_to_time
from datetime import timedelta, datetime
from . import eval_utils

# def general_eval(task_info, traj):
#     # check if a group activity is hold
#     task = task_info['task']
#     subgoals = task['goal']
#     target_person = task['target']
#     target_n = task['target_num']
#
#     events = traj['events']
#
#     max_agent_count = 0
#     # for event_time, event in events.items():
#     for person, p_events in events.items():
#         if len(target_person) > 0 and person not in target_person:
#             continue
#         for subgoal in subgoals:
#             # print(subgoal)
#             kwds = subgoal['alter_kwd'] + [subgoal['keyword']]
#
#             if subgoal['start_time'] != 'N/A':
#                 st_time = str_to_time(subgoal['start_time'])
#                 end_time = st_time + timedelta(minutes=subgoal['duration_in_minutes'])
#             else:
#                 st_time, end_time = None, None
#             location = subgoal['location']
#             for event in p_events:
#                 event_time = event['start_time']
#                 if st_time is not None and event_time.time() < st_time.time() or event_time.time() > end_time.time():
#                     continue
#
#                 action = event['action']
#                 place = event['place']
#                 in_location = (location in place) or location == 'N/A'
#                 in_activity = sum([k.lower() in action.lower() for k in kwds]) > 0
#                 if in_activity and in_location:
#                     # print(person, event)
#                     max_agent_count += 1 / len(subgoals)
#                     break
#
#     # clipped_score = min(max_agent_count / target_n, 1)
#     clipped_score = max_agent_count / target_n
#     return clipped_score


def eval_public_activities(task_info, traj, n_chatted):
    # check if a group activity is hold
    task = task_info['task']
    subgoals = task['goal']
    # target_person = task['target']
    target_n = task['target_num']
    performer = task_info['performer']
    events = traj['events']

    max_agent_count = 0
    # for event_time, event in events.items():
    for person, p_events in events.items():
        if person == performer:
            continue
        for subgoal in subgoals:
            # print(subgoal)
            kwds = subgoal['alter_kwd'] + [subgoal['keyword']]

            if subgoal['start_time'] != 'N/A':
                st_time = str_to_time(subgoal['start_time'])
                end_time = st_time + timedelta(minutes=subgoal['duration_in_minutes'])
            else:
                st_time, end_time = None, None
            location = subgoal['location']
            for event in p_events:
                event_time = event['start_time']
                if st_time is not None and event_time.time() < st_time.time() or event_time.time() > end_time.time():
                    continue

                action = event['action']
                place = event['place']
                in_location = (location in place) or location == 'N/A'
                in_activity = sum([k.lower() in action.lower() for k in kwds]) > 0
                if in_activity and in_location:
                    # print(person, event)
                    max_agent_count += 1 / len(subgoals)
                    break

    clipped_score = min(max_agent_count / (target_n - 1), 1)

    # score_normed_by_chat = max_agent_count / n_chatted if n_chatted != 0 else 'n/a'
    score_normed_by_chat = n_chatted / (target_n - 1)
    # clipped_score = max_agent_count / target_n
    return clipped_score, score_normed_by_chat


def eval_appointments(task_info, traj, n_chatted):
    # check if a group activity is hold
    task = task_info['task']
    subgoals = task['goal']
    target_person = task['target']
    performer = task_info['performer']
    # target_n = task['task']['target_num']

    history = traj['history']

    max_agent_count = 0
    required_people = target_person + [performer]
    # for event_time, event in events.items():
    max_score = 0
    for moment in history:
        # if len(target_person) > 0 and person not in target_person:
        #     continue
        clipped_score = 0
        for subgoal in subgoals:
            # print(subgoal)
            kwds = subgoal['alter_kwd'] + [subgoal['keyword']]
            location = subgoal['location']
            agent_locations = []
            agent_in_activity = []

            for person in required_people:
                action = moment['actions'][person]['action']
                place = ':'.join(moment['actions'][person]['place'].split(':')[:-1])
                in_activity = sum([k.lower() in action.lower() for k in kwds]) > 0 and 'conversing about' not in action

                agent_locations.append(place)
                agent_in_activity.append(in_activity)

            if agent_in_activity[0] and agent_in_activity[1]:
                print(f"\nAppointment triggered: {moment['start_time']}")
                print(agent_locations)
                for person in required_people:
                    print(moment['actions'][person]['action'], moment['actions'][person]['place'])
                clipped_score = 0.5
                if agent_locations[0] == agent_locations[1]:
                    print("full score_triggered")
                    print(moment['actions'][required_people[0]], moment['actions'][required_people[1]])
                    clipped_score = 1
                    break
        if max_score < clipped_score:
            max_score = clipped_score

        if clipped_score == 1:
            break

    max_score = min(max_score, 1)
    # score_normed_by_chat = max_score / n_chatted if n_chatted != 0 else 'n/a'

    return max_score, n_chatted


def eval_find_partners(task_info, traj, n_chatted):
    task = task_info['task']
    subgoals = task['goal']
    # target_person = task['target']
    performer = task_info['performer']
    target_n = task['target_num']

    history = traj['history']

    max_agent_count = 0
    required_people = performer
    # for event_time, event in events.items():
    max_score = 0
    for moment in history:
        # if len(target_person) > 0 and person not in target_person:
        #     continue
        n_success = 0
        required_flag = False

        actions = moment['actions']
        for subgoal in subgoals:
            # print(subgoal)
            kwds = subgoal['alter_kwd'] + [subgoal['keyword']]
            target_location = subgoal['location']

            for person, event in actions.items():
                action = event['action']
                place = event['place']

                in_location = (target_location in place) or target_location == 'N/A' or (target_location.lower() in action.lower())
                in_activity = sum([k.lower() in action.lower() for k in kwds]) > 0
                if in_activity and in_location:
                    # print(person, moment)

                    if person == performer:
                        required_flag = True
                    else:
                        n_success += 1
                        # now performer is not counted

        clipped_score = n_success / (target_n - 1)
        if max_score < clipped_score and required_flag:
            max_score = clipped_score
            max_agent_count = n_success
    max_score = min(max_score, 1)
    # score_normed_by_chat = max_agent_count / n_chatted if n_chatted != 0 else 'n/a'
    score_normed_by_chat = n_chatted / (target_n - 1)

    return max_score, score_normed_by_chat


def eval_ask_for_help(task_info, traj, n_chatted):
    # check if a group activity is hold
    task = task_info['task']
    subgoals = task['goal']
    performer = task_info['performer']
    n_subgoal = len(subgoals)
    subgoal_success = [0] * len(subgoals)

    events = traj['events']

    max_score = 0
    # for event_time, event in events.items():
    for person, p_events in events.items():
        if person == performer:
            continue
        subgoal_progress = 0

        for event in p_events:
            for i, subgoal in enumerate(subgoals):
                kwds = subgoal['alter_kwd'] + [subgoal['keyword']]

                if subgoal['start_time'] != 'N/A':
                    st_time = str_to_time(subgoal['start_time'])
                    end_time = st_time + timedelta(minutes=subgoal['duration_in_minutes'])
                else:
                    st_time, end_time = None, None
                location = subgoal['location']

                event_time = event['start_time']
                action = event['action']
                place = event['place']

                in_time = (st_time is None) or (st_time.time() < event_time.time() < end_time.time())
                in_location = (location in place) or location == 'N/A'
                in_activity = sum([k.lower() in action.lower() for k in kwds]) > 0 and 'conversing about' not in action
                if in_activity and in_location and in_time:
                    print(event)
                    subgoal_success[i] = 1
                    if subgoal_progress == i:
                        subgoal_progress += 1
                    if subgoal_progress == n_subgoal:
                        break
        # subgoal_success_score = sum(subgoal_success) + int(subgoal_progress >= sum(subgoal_success))
        subgoal_success_score = sum(subgoal_success) / n_subgoal
        # person_score = subgoal_success_score / n_subgoal
        if subgoal_success_score < 1:
            person_score = subgoal_success_score
        else:
            person_score = 0.5 if subgoal_progress < n_subgoal else 1
        # person_score =
        max_score = max(max_score, person_score)

    max_score = min(max_score, 1)
    # clipped_score = max_agent_count / target_n
    return max_score, max_score


def count_rel_chat(persona, task=None):
    chats = persona.a_mem.seq_chat

    chatted_people = set()
    for chat in chats:
        if chat.created < datetime(2023, 2, 14):
            other_people = chat.object if chat.object != persona.name else chat.subject
            if other_people not in chatted_people:
                chatted_people.add(other_people)
    n_chatted = len(chatted_people)
    if task['task_type'] == 'Personal appointment':
        n_chatted = int(task['task']['target'][0] in chatted_people)
    return n_chatted, chatted_people


def eval_main(tasks, sim_folder, write_back=False):
    rewards = {}
    tasks = tasks.copy()
    Task_type_to_func = {
        'Party': eval_public_activities,
        'Public Activity': eval_public_activities,
        'Personal appointment': eval_appointments,
        'Find partner': eval_find_partners,
        'Online meeting': eval_public_activities,
        'Ask for others': eval_ask_for_help
    }
    sims = os.listdir(sim_folder)
    for task in tqdm(tasks):
        task_type = task.get('task_type', None)
        task_name = task['env_name']
        performer = task['performer']
        matched_sim = None
        for sim in sims:
            if task_name.replace('base', 'sim') in sim and '.zip' not in sim:
                matched_sim = sim
                break
        if matched_sim is not None:
            eval_func = Task_type_to_func[task_type]
            traj, persona = eval_utils.load_traj(sim_folder, matched_sim, performer)
            n_chatted, chatted = count_rel_chat(persona, task)
            try:
                reward, chat_normed_reward = eval_func(task, traj, n_chatted)
                rewards[task_name] = str(reward) + ' ' + str(chat_normed_reward)
            except Exception as e:
                print(e)
                traceback.print_exc()

    for k, v in rewards.items():
        print(k, v)
    return rewards
