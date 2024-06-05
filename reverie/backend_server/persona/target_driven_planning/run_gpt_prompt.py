import re
import datetime
import sys
import ast

from persona.prompt_template.run_gpt_prompt import extract_first_json_dict

sys.path.append('../../')

from global_methods import *
from persona.prompt_template.gpt_structure import *
from persona.prompt_template.print_prompt import *


def get_general_plan(persona, task, engine='gpt-35-turbo', verbose=False):

    def create_prompt_input(persona, task):
        task_desc = f"Goal: {task['goal']}\n Criteria: {task['Description']}"
        prompt_input = [
            persona.name,
            persona.scratch.get_str_iss(),
            task_desc
        ]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        cr = gpt_response.strip() + '\n'
        return cr

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt="")
        except:
            return False
        return True

    def get_fail_safe():
        fs = 8
        return fs

    gpt_param = {"engine": engine, "max_tokens": 300,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0.1, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/target_driven_planning/tgp_prompt_templates/get_general_plan_v1.txt"
    prompt_input = create_prompt_input(persona, task)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()

    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def get_daily_req_reminder(persona, task, general_plan, engine='gpt-35-turbo', verbose=False):

    def create_prompt_input(persona, task):
        task_desc = f"Goal: {task['goal']}\n Criteria: {task['Description']}"
        prompt_input = [
            persona.name,
            persona.scratch.get_str_iss(),
            task_desc,
            general_plan,
            "2023.2.13"
        ]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        cr = gpt_response.strip().strip('\n') + '\n'
        return cr

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt="")
        except:
            return False
        return True

    def get_fail_safe():
        fs = 8
        return fs

    gpt_param = {"engine": engine, "max_tokens": 300,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0.1, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/target_driven_planning/tgp_prompt_templates/get_daily_req_reminder_v1.txt"
    prompt_input = create_prompt_input(persona, task)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()

    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def get_daily_plan_reminder(persona, task):
    pass


def get_conversation_reminder(persona, task, general_plan, engine='gpt-35-turbo', verbose=False):

    def create_prompt_input(persona, task):
        task_desc = f"Goal: {task['goal']}\n Criteria: {task['Description']}"
        prompt_input = [
            persona.name,
            persona.scratch.get_str_iss(),
            task_desc,
            general_plan,
        ]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        cr = gpt_response.strip().strip('\n') + '\n'
        return cr

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt="")
        except:
            return False
        return True

    def get_fail_safe():
        fs = 8
        return fs

    gpt_param = {"engine": engine, "max_tokens": 300,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0.1, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/target_driven_planning/tgp_prompt_templates/get_conversation_reminder_v1.txt"
    prompt_input = create_prompt_input(persona, task)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()

    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_daily_plan_for_task(persona, wake_up_hour, task, daily_req_reminder,
                              test_input=None, engine='gpt-35-turbo', verbose=False):
    """
  Basically the long term planning that spans a day. Returns a list of actions
  that the persona will take today. Usually comes in the following form:
  'wake up and complete the morning routine at 6:00 am',
  'eat breakfast at 7:00 am',..
  Note that the actions come without a period.

  INPUT:
    persona: The Persona class instance
  OUTPUT:
    a list of daily actions in broad strokes.
  """
    task_desc = f"Goal: {task['goal']}\n Criteria: {task['Description']}"

    prompt_input = []
    prompt_input += [persona.scratch.get_str_iss()]
    prompt_input += [persona.scratch.get_str_lifestyle()]
    prompt_input += [persona.scratch.get_str_curr_date_str()]
    prompt_input += [persona.scratch.get_str_firstname()]
    prompt_input += [f"{str(wake_up_hour)}:00 am"]
    prompt_input += [task_desc]
    prompt_input += [daily_req_reminder]

    def __func_clean_up(gpt_response, prompt=""):
        cr = []
        _cr = gpt_response.split(")")
        for i in _cr:
            if i[-1].isdigit():
                i = i[:-1].strip()
                if i[-1] == "." or i[-1] == ",":
                    cr += [i[:-1].strip()]
        return cr

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt="")
        except:
            return False
        return True

    def get_fail_safe():
        fs = ['wake up and complete the morning routine at 6:00 am',
              'eat breakfast at 7:00 am',
              'read a book from 8:00 am to 12:00 pm',
              'have lunch at 12:00 pm',
              'take a nap from 1:00 pm to 4:00 pm',
              'relax and watch TV from 7:00 pm to 8:00 pm',
              'go to bed at 11:00 pm']
        return fs

    gpt_param = {"engine": engine, "max_tokens": 500,
                 "temperature": 1, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/target_driven_planning/tgp_prompt_templates/daily_planning_v1.txt"

    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()

    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)
    output = ([f"wake up and complete the morning routine at {wake_up_hour}:00 am"]
              + output)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def gen_iterative_chat_utt_with_plan(maze, init_persona, target_persona, retrieved, curr_context, curr_chat,
                                     plan_info, test_input=None, engine='gpt-35-turbo', verbose=False):
    def create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat):
        persona = init_persona
        prev_convo_insert = "\n"
        if persona.a_mem.seq_chat:
            for i in persona.a_mem.seq_chat:
                if i.object == target_persona.scratch.name:
                    v1 = int((persona.scratch.curr_time - i.created).total_seconds() / 60)
                    prev_convo_insert += f'{str(v1)} minutes ago, {persona.scratch.name} and {target_persona.scratch.name} were already {i.description} This context takes place after that conversation.'
                    break
        if prev_convo_insert == "\n":
            prev_convo_insert = ""
        if persona.a_mem.seq_chat:
            if int((persona.scratch.curr_time - persona.a_mem.seq_chat[-1].created).total_seconds() / 60) > 480:
                prev_convo_insert = ""
        print(prev_convo_insert)

        curr_sector = f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"
        curr_arena = f"{maze.access_tile(persona.scratch.curr_tile)['arena']}"
        curr_location = f"{curr_arena} in {curr_sector}"

        retrieved_str = ""
        for key, vals in retrieved.items():
            for v in vals:
                retrieved_str += f"- {v.description}\n"

        convo_str = ""
        for i in curr_chat:
            convo_str += ": ".join(i) + "\n"
        if convo_str == "":
            convo_str = "[The conversation has not started yet -- start it!]"

        init_iss = f"Here is Here is a brief description of {init_persona.scratch.name}.\n{init_persona.scratch.get_str_iss()}"
        prompt_input = [init_iss, init_persona.scratch.name, retrieved_str, prev_convo_insert,
                        curr_location, curr_context, init_persona.scratch.name, target_persona.scratch.name,
                        convo_str, init_persona.scratch.name, target_persona.scratch.name, init_persona.scratch.name, init_persona.scratch.name,
                        init_persona.scratch.name
                        ]

        special_instruction = plan_info
        prompt_input.append(special_instruction)

        return prompt_input

    def __chat_func_clean_up(gpt_response, prompt=""):
        gpt_response = extract_first_json_dict(gpt_response)

        cleaned_dict = dict()
        cleaned = []
        for key, val in gpt_response.items():
            cleaned += [val]
        cleaned_dict["utterance"] = cleaned[0]
        cleaned_dict["end"] = True
        if "f" in str(cleaned[1]) or "F" in str(cleaned[1]):
            cleaned_dict["end"] = False

        return cleaned_dict

    def __chat_func_validate(gpt_response, prompt=""):
        print("ugh...")
        try:
            print(extract_first_json_dict(gpt_response))
            assert extract_first_json_dict(gpt_response) is not None

            return True
        except:
            return False

    def get_fail_safe():
        cleaned_dict = dict()
        cleaned_dict["utterance"] = "..."
        cleaned_dict["end"] = False
        return cleaned_dict

    print("11")
    prompt_template = "persona/target_driven_planning/tgp_prompt_templates/iterative_convo_v1.txt"
    prompt_input = create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat)
    print("22")
    prompt = generate_prompt(prompt_input, prompt_template)
    print(prompt)
    fail_safe = get_fail_safe()
    gpt_param = {"engine": engine, "max_tokens": 250,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}

    output = safe_generate_response(prompt, gpt_param, 3, fail_safe,
                                                __chat_func_validate, __chat_func_clean_up, verbose)
    # output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
    #                                             __chat_func_validate, __chat_func_clean_up, verbose)
    print(output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def gen_iterative_chat_utt_with_plan_simple(maze, init_persona, target_persona, retrieved, curr_context, curr_chat,
                                     plan_info, test_input=None, engine='gpt-35-turbo', verbose=False):
    def create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat):
        persona = init_persona
        prev_convo_insert = "\n"
        if persona.a_mem.seq_chat:
            for i in persona.a_mem.seq_chat:
                if i.object == target_persona.scratch.name:
                    v1 = int((persona.scratch.curr_time - i.created).total_seconds() / 60)
                    prev_convo_insert += f'{str(v1)} minutes ago, {persona.scratch.name} and {target_persona.scratch.name} were already {i.description} This context takes place after that conversation.'
                    break
        if prev_convo_insert == "\n":
            prev_convo_insert = ""
        if persona.a_mem.seq_chat:
            if int((persona.scratch.curr_time - persona.a_mem.seq_chat[-1].created).total_seconds() / 60) > 480:
                prev_convo_insert = ""
        print(prev_convo_insert)

        curr_sector = f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"
        curr_arena = f"{maze.access_tile(persona.scratch.curr_tile)['arena']}"
        curr_location = f"{curr_arena} in {curr_sector}"

        retrieved_str = ""
        for key, vals in retrieved.items():
            for v in vals:
                retrieved_str += f"- {v.description}\n"

        convo_str = ""
        for i in curr_chat:
            convo_str += ": ".join(i) + "\n"
        if convo_str == "":
            convo_str = "[The conversation has not started yet -- start it!]"

        init_iss = f"Here is Here is a brief description of {init_persona.scratch.name}.\n{init_persona.scratch.get_str_iss()}"
        prompt_input = [init_iss, init_persona.scratch.name, retrieved_str, prev_convo_insert,
                        curr_location, curr_context, init_persona.scratch.name, target_persona.scratch.name,
                        convo_str, init_persona.scratch.name, target_persona.scratch.name, init_persona.scratch.name, init_persona.scratch.name,
                        init_persona.scratch.name
                        ]

        special_instruction = plan_info
        prompt_input.append(special_instruction)

        return prompt_input

    def __chat_func_clean_up(gpt_response, prompt=""):
        gpt_response = gpt_response.strip()
        llama_clean = gpt_response.split('\n\n')
        if 'llama' in engine.lower() and len(llama_clean) > 1 and llama_clean[0].endswith('in the conversation:'):
            gpt_response = llama_clean[1].strip('"')

        cleaned_dict = dict()
        cleaned_dict["utterance"] = gpt_response
        cleaned_dict["end"] = False

        return cleaned_dict

    def __chat_func_validate(gpt_response, prompt=""):
        return True
        # print("ugh...")
        # try:
        #     print(extract_first_json_dict(gpt_response))
        #     assert extract_first_json_dict(gpt_response) is not None
        #
        #     return True
        # except:
        #     return False

    def get_fail_safe():
        cleaned_dict = dict()
        cleaned_dict["utterance"] = "..."
        cleaned_dict["end"] = False
        return cleaned_dict

    prompt_template = "persona/target_driven_planning/tgp_prompt_templates/iterative_convo_simple_v1.txt"
    prompt_input = create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat)
    prompt = generate_prompt(prompt_input, prompt_template)
    print(prompt)
    fail_safe = get_fail_safe()
    gpt_param = {"engine": engine, "max_tokens": 250,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}

    output = safe_generate_response(prompt, gpt_param, 3, fail_safe,
                                                __chat_func_validate, __chat_func_clean_up, verbose)
    # output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
    #                                             __chat_func_validate, __chat_func_clean_up, verbose)
    print(output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

def update_conversation_checklist(persona, task):
    pass


# def gpt_eval_conversation(task, conversation):
#     convo_content = "\n".join([
#         f"{speaker}: {words}" for speaker, words in conversation
#     ])
#
#     def create_prompt_input(persona, target_persona, statements, test_input=None):
#         prompt_input = [statements, persona.scratch.name, target_persona.scratch.name]
#         return prompt_input
#
#     def __func_clean_up(gpt_response, prompt=""):
#         return gpt_response.split('"')[0].strip()
#
#     def __func_validate(gpt_response, prompt=""):
#         try:
#             __func_clean_up(gpt_response, prompt)
#             return True
#         except:
#             return False
#
#     def get_fail_safe():
#         return "..."
#
#     # ChatGPT Plugin ===========================================================
#     def __chat_func_clean_up(gpt_response, prompt=""):  ############
#         return gpt_response.split('"')[0].strip()
#
#     def __chat_func_validate(gpt_response, prompt=""):  ############
#         try:
#             __func_clean_up(gpt_response, prompt)
#             return True
#         except:
#             return False
#
#     print("asdhfapsh8p9hfaiafdsi;ldfj as DEBUG 18")  ########
#     gpt_param = {"engine": "text-davinci-002", "max_tokens": 15,
#                  "temperature": 0, "top_p": 1, "stream": False,
#                  "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
#     prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_chat_relationship_v2.txt"  ########
#     prompt_input = create_prompt_input(persona, target_persona, statements)  ########
#     prompt = generate_prompt(prompt_input, prompt_template)
#     example_output = 'Jane Doe is working on a project'  ########
#     special_instruction = 'The output should be a string that responds to the question.'  ########
#     fail_safe = get_fail_safe()  ########
#     output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
#                                             __chat_func_validate, __chat_func_clean_up, True)
#     if output != False:
#         return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def gpt_eval_conversation_party(task, conversation, engine='gpt-35-turbo'):
    convo_content = "\n".join([
        f"{speaker}: {words}" for speaker, words in conversation
    ])
    main_persona_name = task['performer']
    bg_persona_name = [speaker for speaker, words in conversation if speaker != main_persona_name][0]
    activity_name = task['task']['goal'][0]['keyword']
    activity_datetime = task['task']['goal'][0]['start_time']
    activity_loc = task['task']['goal'][0]['location']

    def create_prompt_input():
        prompt_input = [
            main_persona_name,
            bg_persona_name,
            convo_content,
            activity_name,
            activity_datetime,
            activity_loc,
        ]

        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        scores = gpt_response.split('\n')
        assert len(scores) == 6
        scores[0] = int('yes' in scores[0].lower())
        scores[1] = int('yes' in scores[1].lower())
        scores[2] = int('yes' in scores[2].lower())

        return scores

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return False
        # return "..."
    # ChatGPT Plugin ===========================================================

    gpt_param = {"engine": engine, "max_tokens": 100,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/target_driven_planning/tgp_prompt_templates/eval_conversation_party_v1.txt"  ########
    prompt_input = create_prompt_input()  ########
    prompt = generate_prompt(prompt_input, prompt_template)
    # print(prompt_input, prompt)

    fail_safe = get_fail_safe()  ########
    output = safe_generate_response(prompt, gpt_param, 3, fail_safe,
                                            __func_validate, __func_clean_up, True)
    if output != False:
        # print(output)
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def gpt_eval_summary_party(task, summary):
    # convo_content = "\n".join([
    #     # f"{speaker}: {words}" for speaker, words in conversation
    # ])
    # main_persona_name = task['performer']
    # bg_persona_name = [speaker for speaker, words in conversation if speaker != main_persona_name][0]
    activity_name = task['task']['goal'][0]['keyword']
    activity_datetime = task['task']['goal'][0]['start_time']
    activity_loc = task['task']['goal'][0]['location']

    def create_prompt_input():
        prompt_input = [
            '',
            '',
            summary,
            activity_name,
            activity_datetime,
            activity_loc,
        ]

        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        scores = gpt_response.split('\n')
        scores[0] = int('yes' in scores[0].lower())
        scores[1] = int('yes' in scores[1].lower())
        scores[2] = int('yes' in scores[2].lower())

        return scores

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return "..."

    # ChatGPT Plugin ===========================================================

    gpt_param = {"engine": "gpt-35-turbo", "max_tokens": 100,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/target_driven_planning/tgp_prompt_templates/eval_summary_party_v1.txt"  ########
    prompt_input = create_prompt_input()  ########
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()  ########
    output = safe_generate_response(prompt, gpt_param, 3, fail_safe,
                                            __func_validate, __func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def gpt_eval_conversation_appointment(task, conversation, engine='gpt-35-turbo'):
    convo_content = "\n".join([
        f"{speaker}: {words}" for speaker, words in conversation
    ])
    main_persona_name = task['performer']
    bg_persona_name = [speaker for speaker, words in conversation if speaker != main_persona_name][0]
    activity_name = task['task']['goal'][0]['keyword']
    activity_datetime = task['task']['goal'][0]['start_time']
    activity_loc = task['task']['goal'][0]['location']

    def create_prompt_input():
        prompt_input = [
            main_persona_name,
            bg_persona_name,
            convo_content,
            activity_name,
            activity_datetime,
            activity_loc,
        ]

        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        scores = gpt_response.split('\n')
        assert len(scores) == 6
        scores[0] = int('yes' in scores[0].lower())
        scores[1] = int('yes' in scores[1].lower())
        scores[2] = int('yes' in scores[2].lower())

        return scores

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return False
        # return "..."
    # ChatGPT Plugin ===========================================================

    gpt_param = {"engine": engine, "max_tokens": 100,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/target_driven_planning/tgp_prompt_templates/eval_conversation_appointment_v1.txt"  ########
    prompt_input = create_prompt_input()  ########
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()  ########
    output = safe_generate_response(prompt, gpt_param, 3, fail_safe,
                                            __func_validate, __func_clean_up, True)
    if output != False:
        # print(output)
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def gpt_eval_summary_appointment(task, summary):
    # convo_content = "\n".join([
    #     # f"{speaker}: {words}" for speaker, words in conversation
    # ])
    # main_persona_name = task['performer']
    # bg_persona_name = [speaker for speaker, words in conversation if speaker != main_persona_name][0]
    activity_name = task['task']['goal'][0]['keyword']
    activity_datetime = task['task']['goal'][0]['start_time']
    activity_loc = task['task']['goal'][0]['location']

    def create_prompt_input():
        prompt_input = [
            '',
            '',
            summary,
            activity_name,
            activity_datetime,
            activity_loc,
        ]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        scores = gpt_response.split('\n')
        scores[0] = int('yes' in scores[0].lower())
        scores[1] = int('yes' in scores[1].lower())
        scores[2] = int('yes' in scores[2].lower())

        return scores

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return "..."

    gpt_param = {"engine": "gpt-35-turbo", "max_tokens": 100,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/target_driven_planning/tgp_prompt_templates/eval_summary_party_v1.txt"  ########
    prompt_input = create_prompt_input()  ########
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()  ########
    output = safe_generate_response(prompt, gpt_param, 3, fail_safe,
                                            __func_validate, __func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def gpt_eval_conversation_find_partner(task, conversation, engine='gpt-35-turbo'):
    convo_content = "\n".join([
        f"{speaker}: {words}" for speaker, words in conversation
    ])
    main_persona_name = task['performer']
    bg_persona_name = [speaker for speaker, words in conversation if speaker != main_persona_name][0]
    activity_name = task['task']['goal'][0]['keyword']
    activity_datetime = task['task']['goal'][0]['start_time']
    activity_loc = task['task']['goal'][0]['location']

    def create_prompt_input():
        prompt_input = [
            main_persona_name,
            bg_persona_name,
            convo_content,
            activity_name,
            activity_datetime,
            activity_loc,
        ]

        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        scores = gpt_response.split('\n')
        assert len(scores) == 6
        scores[0] = int('yes' in scores[0].lower())
        scores[1] = int('yes' in scores[1].lower())
        scores[2] = int('yes' in scores[2].lower())

        return scores

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return False
        # return "..."

    # ChatGPT Plugin ===========================================================

    gpt_param = {"engine": engine, "max_tokens": 100,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/target_driven_planning/tgp_prompt_templates/eval_conversation_find_partner_v1.txt"  ########
    prompt_input = create_prompt_input()  ########
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()  ########
    output = safe_generate_response(prompt, gpt_param, 3, fail_safe,
                                    __func_validate, __func_clean_up, True)
    if output != False:
        # print(output)
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def gpt_eval_summary_find_partner(task, summary):
    # convo_content = "\n".join([
    #     # f"{speaker}: {words}" for speaker, words in conversation
    # ])
    # main_persona_name = task['performer']
    # bg_persona_name = [speaker for speaker, words in conversation if speaker != main_persona_name][0]
    activity_name = task['task']['goal'][0]['keyword']
    activity_datetime = task['task']['goal'][0]['start_time']
    activity_loc = task['task']['goal'][0]['location']

    def create_prompt_input():
        prompt_input = [
            '',
            '',
            summary,
            activity_name,
            activity_datetime,
            activity_loc,
        ]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        scores = gpt_response.split('\n')
        scores[0] = int('yes' in scores[0].lower())
        scores[1] = int('yes' in scores[1].lower())
        scores[2] = int('yes' in scores[2].lower())

        return scores

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return False
        # return "..."

    gpt_param = {"engine": "gpt-35-turbo", "max_tokens": 100,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/target_driven_planning/tgp_prompt_templates/eval_summary_find_partner_v1.txt"  ########
    prompt_input = create_prompt_input()  ########
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()  ########
    output = safe_generate_response(prompt, gpt_param, 3, fail_safe,
                                    __func_validate, __func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def gpt_eval_conversation_online_meeting(task, conversation, engine='gpt-35-turbo'):
    convo_content = "\n".join([
        f"{speaker}: {words}" for speaker, words in conversation
    ])
    main_persona_name = task['performer']
    bg_persona_name = [speaker for speaker, words in conversation if speaker != main_persona_name][0]
    activity_name = task['task']['goal'][0]['keyword']
    activity_datetime = task['task']['goal'][0]['start_time']
    activity_loc = task['task']['goal'][0]['location']

    def create_prompt_input():
        prompt_input = [
            main_persona_name,
            bg_persona_name,
            convo_content,
            activity_name,
            activity_datetime,
            activity_loc,
        ]

        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        scores = gpt_response.split('\n')

        scores[0] = int('yes' in scores[0].lower())
        scores[1] = int('yes' in scores[1].lower())

        return scores

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return False
        # return "..."

    # ChatGPT Plugin ===========================================================

    gpt_param = {"engine": engine, "max_tokens": 100,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/target_driven_planning/tgp_prompt_templates/eval_conversation_online_meeting_v1.txt"  ########
    prompt_input = create_prompt_input()  ########
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()  ########
    output = safe_generate_response(prompt, gpt_param, 3, fail_safe,
                                    __func_validate, __func_clean_up, True)
    if output != False:
        # print(output)
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def gpt_eval_summary_online_meeting(task, summary):
    # convo_content = "\n".join([
    #     # f"{speaker}: {words}" for speaker, words in conversation
    # ])
    # main_persona_name = task['performer']
    # bg_persona_name = [speaker for speaker, words in conversation if speaker != main_persona_name][0]
    activity_name = task['task']['goal'][0]['keyword']
    activity_datetime = task['task']['goal'][0]['start_time']
    activity_loc = task['task']['goal'][0]['location']

    def create_prompt_input():
        prompt_input = [
            '',
            '',
            summary,
            activity_name,
            activity_datetime,
            activity_loc,
        ]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        scores = gpt_response.split('\n')
        scores[0] = int('yes' in scores[0].lower())
        scores[1] = int('yes' in scores[1].lower())

        return scores

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return False
        # return "..."

    gpt_param = {"engine": "gpt-35-turbo", "max_tokens": 100,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/target_driven_planning/tgp_prompt_templates/eval_summary_online_meeting_v1.txt"  ########
    prompt_input = create_prompt_input()  ########
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()  ########
    output = safe_generate_response(prompt, gpt_param, 3, fail_safe,
                                    __func_validate, __func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def gpt_eval_conversation_ask_for_help(task, conversation, engine='gpt-35-turbo'):
    convo_content = "\n".join([
        f"{speaker}: {words}" for speaker, words in conversation
    ])
    main_persona_name = task['performer']
    bg_persona_name = [speaker for speaker, words in conversation if speaker != main_persona_name][0]
    activity_name_1 = task['task']['goal'][0]['keyword']
    activity_loc_1 = task['task']['goal'][0]['location']
    activity_name_2 = task['task']['goal'][1]['keyword']
    activity_loc_2 = task['task']['goal'][1]['location']

    prompt_input = [
        main_persona_name,
        bg_persona_name,
        convo_content,
        activity_name_1,
        activity_loc_1,
        activity_name_2,
        activity_loc_2,
    ]

    def __func_clean_up(gpt_response, prompt=""):
        scores = gpt_response.split('\n')

        scores[0] = int('yes' in scores[0].lower())
        scores[1] = int('yes' in scores[1].lower())
        scores[2] = int('yes' in scores[2].lower())
        scores[3] = int('yes' in scores[3].lower())

        return scores

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return False
        # return "..."

    # ChatGPT Plugin ===========================================================

    gpt_param = {"engine": engine, "max_tokens": 100,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/target_driven_planning/tgp_prompt_templates/eval_conversation_ask_for_help_v1.txt"  ########

    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()  ########
    output = safe_generate_response(prompt, gpt_param, 3, fail_safe,
                                    __func_validate, __func_clean_up, True)
    if output != False:
        # print(output)
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def gpt_eval_summary_ask_for_help(task, summary):
    activity_name_1 = task['task']['goal'][0]['keyword']
    activity_loc_1 = task['task']['goal'][0]['location']
    activity_name_2 = task['task']['goal'][1]['keyword']
    activity_loc_2 = task['task']['goal'][1]['location']

    prompt_input = [
        '',
        '',
        summary,
        activity_name_1,
        activity_loc_1,
        activity_name_2,
        activity_loc_2,
    ]

    def __func_clean_up(gpt_response, prompt=""):
        scores = gpt_response.split('\n')
        scores[0] = int('yes' in scores[0].lower())
        scores[1] = int('yes' in scores[1].lower())
        scores[2] = int('yes' in scores[2].lower())
        scores[3] = int('yes' in scores[3].lower())
        return scores

    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            return False

    def get_fail_safe():
        return "..."

    gpt_param = {"engine": "gpt-35-turbo", "max_tokens": 100,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    prompt_template = "persona/target_driven_planning/tgp_prompt_templates/eval_summary_ask_for_help_v1.txt"  ########
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()  ########
    output = safe_generate_response(prompt, gpt_param, 3, fail_safe,
                                    __func_validate, __func_clean_up, True)
    if output != False:
        return output, [output, prompt, gpt_param, prompt_input, fail_safe]
