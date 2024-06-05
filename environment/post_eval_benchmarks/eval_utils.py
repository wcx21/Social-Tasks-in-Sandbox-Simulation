import json
import logging
import os
import calendar
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'reverie', 'backend_server'))
# print(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'reverie', 'backend_server'))

from collections import defaultdict
from datetime import datetime, timedelta
from persona.persona import Persona

DEFAULT_TASK_ROOT = './data/designed_tasks'
DEFAULT_SIM_ROOT = './frontend_server/storage'


def time_to_str(time: datetime):
    return time.strftime("%Y-%m-%d %H:%M")


def str_to_time(time_str: str):
    return datetime.strptime(time_str, "%Y-%m-%d %H:%M")


def load_move(sim_root, sim_name):
    movement_path = os.path.join(sim_root, sim_name, 'movement')
    movements = []
    for file_name in os.listdir(movement_path):
        movement = json.load(open(os.path.join(movement_path, file_name), encoding='utf-8'))
        movements.append(movement)

    return movements


def parse_ga_time(time_str):
    month_day, year, time = time_str.split(', ')
    h, m, s = time.split(':')
    month, day = month_day.split(' ')
    month = list(calendar.month_name).index(month)
    year, month, day, h, m, s = [int(o) for o in [year, month, day, h, m, s]]
    dt = datetime(year, month, day, h, m, s)
    return dt


def parse_movements_to_diary(movements: list, time_interval_sec=30):
    if len(movements) == 0:
        return {}

    time_step = timedelta(seconds=time_interval_sec)
    diary = defaultdict(list)
    for movement in movements:
        event_time = parse_ga_time(movement["meta"]["curr_time"])

        for person in movement['persona']:
            event = movement['persona'][person]
            action, place = event['description'].split(' @ ')

            d_event = {
                'start_time': event_time,
                'end_time': event_time + time_step,
                'action': action,
                'place': place
            }
            diary[person].append(d_event)

    # perform combinations
    for person, p_diary in diary.items():
        p_diary = sorted(p_diary, key=lambda x: x['start_time'])
        new_p_diary = []
        curr_event = p_diary[0]
        for i in range(1, len(p_diary)):
            next_event = p_diary[i]
            if next_event['action'] == curr_event['action'] and next_event['place'] == curr_event['place']:
                curr_event['end_time'] = next_event['end_time']
            else:
                new_p_diary.append(curr_event)
                curr_event = next_event
        new_p_diary.append(curr_event)
        diary[person] = new_p_diary

    return diary


def parse_movements_to_history(movements: list, time_interval_sec=30):
    if len(movements) == 0:
        return {}

    time_step = timedelta(seconds=time_interval_sec)
    history = list()
    for movement in movements:
        actions = dict()
        event_time = parse_ga_time(movement["meta"]["curr_time"])

        for person in movement['persona']:
            event = movement['persona'][person]
            action, place = event['description'].split(' @ ')

            actions[person] = {
                'action': action,
                'place': place
            }

        moment = {
            'start_time': event_time,
            'end_time': event_time + time_step,
            'actions': actions
        }
        history.append(moment)

    return history


def load_traj(sim_root, sim_name, performer):
    movements = load_move(sim_root, sim_name)
    if len(movements) < 2800:
        print(f"Warning, {sim_name} seems incomplete, only {len(movements)} steps")
    meta = json.load(open(os.path.join(sim_root, sim_name, 'reverie', 'meta.json'), encoding='utf-8'))
    time_interval_sec = meta['sec_per_step']
    traj = {
        'events': parse_movements_to_diary(movements, time_interval_sec),
        'history': parse_movements_to_history(movements, time_interval_sec),
        'time_step': timedelta(seconds=time_interval_sec)
    }
    main_persona = Persona(performer, os.path.join(sim_root, sim_name, 'personas', performer))

    return traj, main_persona
