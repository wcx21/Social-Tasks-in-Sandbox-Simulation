import re
import datetime
import sys
import ast

sys.path.append('../../')

from global_methods import *
from persona.target_driven_planning.run_gpt_prompt import *


class TargetDrivenPlanner:
    def __init__(self, task, persona, model='gpt-35-turbo', is_naive=False):
        self.task = task
        self.persona = persona
        self.model = model
        self.is_naive = is_naive
        if 'llama' in self.model or 'local' in self.model:
            self.conv_func = gen_iterative_chat_utt_with_plan_simple
        else:
            self.conv_func = gen_iterative_chat_utt_with_plan
        # self.initial_plan()

    def initial_plan(self):
        if self.is_naive:
            return
        self.general_plan, _ = get_general_plan(self.persona, self.task, engine=self.model)
        self.schedule_reminder, _ = get_daily_req_reminder(self.persona, self.task, self.general_plan, engine=self.model)
        self.conversation_reminder, _ = get_conversation_reminder(self.persona, self.task, self.general_plan, engine=self.model)

        self.task_desc = f"Goal: {self.task['goal']}\n Criteria: {self.task['Description']}"

    def gen_new_daily_req(self, persona, wake_up_hour):
        outputs = run_gpt_prompt_daily_plan_for_task(persona, wake_up_hour, self.task, self.schedule_reminder, engine=self.model)
        return outputs

    def start_convo(self):
        pass

    def gen_next_convo_utterance(self, maze, speaker, listener, retrieved, curr_chat):
        curr_context = (f"{speaker.scratch.name} " +
                        f"was {speaker.scratch.act_description} " +
                        f"when {speaker.scratch.name} " +
                        f"saw {listener.scratch.name} " +
                        f"in the middle of {listener.scratch.act_description}.\n")
        curr_context += (f"{speaker.scratch.name} " +
                         f"is initiating a conversation with " +
                         f"{listener.scratch.name}.")
        if self.is_naive:
            plan_info = ''
        else:
            plan_info = f"""
Today is Monday 2023-02-13. \n
{speaker.name} is performing a task, which has the information as follows:
{self.task_desc}
\n
To perform the task, {speaker.name} will draw up the plan and execute it step by step. The plan should focus on how to achieve higher score in the task, which suggests {speaker.name} to carefully consider the goal and criteria.
{speaker.name} has decomposed the task into few keypoints or subgoals:
{self.general_plan}\n

{speaker.name} has planned a checklist for the conversation:
{self.conversation_reminder}
\n
Please keep the checklist in mind when generating the utterance of {speaker.name}.
"""
        output, info = self.conv_func(maze, speaker, listener, retrieved, curr_context, curr_chat,
                                                        engine=self.model, plan_info=plan_info)
        return output["utterance"], output["end"]

    def end_convo(self):
        pass
