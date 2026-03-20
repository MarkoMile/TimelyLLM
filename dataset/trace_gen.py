import re
import random
import json

def random_time(min_time, max_time):
    return random.uniform(min_time, max_time)

class ActionDelay:
    def __init__(self):
        pass

    def delay_map(self, command, robot_type='drone', robot_system='typefly'):
        value = self.extract_value(command)
        delay = 0

        match_list = []

        if robot_type == 'drone' and robot_system == 'typefly':
            s_value = [1.5275,	3.055,	4.5825,	6.11,	7.6375,	9.165,	10.6925,	12.22]
            sa_value = [1.5675,	3.135,	4.7025,	6.27,	7.8375,	9.405,	10.9725,	12.54]

            # unit:s
            keywords = {
                r'\bmf\b': lambda v: v * 0.0072 + 1.8144,
                r'\bmb\b': lambda v: v * 0.0072 + 1.8144,
                r'\bml\b': lambda v: v * 0.0072 + 1.8144,
                r'\bmr\b': lambda v: v * 0.0072 + 1.8144,
                r'\bmu\b': lambda v: 8e-7*v**3 - 0.0003*v**2+0.0388*v+1.3195,
                r'\bmd\b': lambda v: 8e-7*v**3 - 0.0003*v**2+0.0388*v+1.3195,
                r'\btc\b': lambda v: v * 0.0165 + 0.784,
                r'\btu\b': lambda v: v * 0.0165 + 0.784,
                r'\box\b': lambda _: 0.001,
                r'\boy\b': lambda _: 0.001,
                r'\bow\b': lambda _: 0.001,
                r'\boh\b': lambda _: 0.001,
                r'\bod\b': lambda _: 0.001,
                r'\biv\b': lambda _: 0.001,
                r'(?<![tr])\bp\b': lambda _: 0.041, # probe
                r'(?<!m)\bl\b': lambda _: 0.001,  # log
                r'(?<!m)\bd\b': lambda v: v,  # delay
                r'\btp\b': lambda _: 0.001,
                r'\brp\b': lambda _: 0.001,
                r'\bg\(': lambda _: 2.5,
                r'\bs\(': lambda _: random.choice(s_value),
                r'\bsa\(': lambda _: random.choice(sa_value)
            }

        elif robot_type == 'robotarm' and robot_system == 'typefly':
            # unit:s
            keywords = {
                r'\biv\b': lambda _: 0.001,
                r'(?<!m)\bl\b': lambda _: 0.001,  # log
                r'(?<!m)\bd\b': lambda v: v,  # delay
                r'\bgt\b': lambda _: random_time(0.8, 5.5),  # goto
                r'\bpi\b': lambda _: random_time(3, 4.2), # pick
                r'\bdr\b': lambda _: random_time(3, 4.2), # drop
                r'\bsc\b': lambda _: random_time(1, 3),  # scan
                }
        
        elif robot_type == 'robotarm' and robot_system == 'FLTRNN':
            # unit:s
            keywords = {
                r'\biv\b': lambda _: 0.001,
                r'(?<!m)\bl\b': lambda _: 0.001,  # log
                r'(?<!m)\bd\b': lambda v: v,  # delay
                # for robot-arm-fltrnn
                r'\bmove_to\b': lambda _: random_time(1, 3),  # move_to
                r'\btake\b': lambda _: random_time(3, 4.2), # take
                r'\bdrop\b': lambda _: random_time(3, 4.2) # drop
                }
        
        elif robot_type == 'robotdog' and robot_system == 'typefly':
            s_value = [0.8005, 1.601, 2.4015, 3.202, 4.0025, 4.803, 5.6035, 6.404]

            # unit:s
            keywords = {
                r'\bmf\b': lambda v: v * 0.01125 + 0.375,
                r'\bmb\b': lambda v: v * 0.01125 + 0.375,
                r'\bml\b': lambda v: v * 0.0205 + 0.42,
                r'\bmr\b': lambda v: v * 0.0205 + 0.42,
                r'\btc\b': lambda v: v * 0.0111 + 0.3,
                r'\btu\b': lambda v: v * 0.0111 + 0.3,
                r'\box\b': lambda _: 0.001,
                r'\boy\b': lambda _: 0.001,
                r'\bow\b': lambda _: 0.001,
                r'\boh\b': lambda _: 0.001,
                r'\bod\b': lambda _: 0.001,
                r'\biv\b': lambda _: 0.001,
                r'(?<!m)\bl\b': lambda _: 0.001,  # log
                r'(?<!m)\bd\b': lambda v: v,  # delay
                r'\btp\b': lambda _: 0.001,
                r'\brp\b': lambda _: 0.001,
                r'\bg\(': lambda _: random.uniform(0.9375, 4.5495), 
                r'\bs\(': lambda _: random.choice(s_value),
                r'\bsd\b': lambda _: 1, #sound
            }

        # for key, func in keywords.items():
        #     matches = re.finditer(key, command)
        #     for match in matches:
        #         # print(match)
        #         matched_text = match.group(0)
        #         result = func(value)
        #         delay += result
        #         match_list.append((matched_text, round(result, 6)))
        # Sort the matches by the order they appear in the command
        sorted_matches = sorted(
            (match.start(), key, func) 
            for key, func in keywords.items() 
            for match in re.finditer(key, command)
        )

        # Process each match in order
        for _, key, func in sorted_matches:
            matched_text = re.search(key, command).group(0)
            result = func(value)
            delay += result
            match_list.append((matched_text, round(result, 6)))

        return round(delay, 6), match_list
        
    def extract_value(self, command):
        match = re.search(r'\((\d+)\)', command)
        if match:
            return float(match.group(1))
        return 0 
    
    def chat_delay_map(self, command):
        word_count = len(command.split())
        reading_speed = 5 # 5 words per second
        reading_time = word_count / reading_speed
        return round(reading_time, 6)
    


if __name__ == "__main__":

    # -------------------------------- drone ------------------------------------------ #
    description = {
        0: " Scene: [] Task: [A] Could you find the cat? If so, go to it.",
        1: " Scene: [countertop_1 x:0.48 y:0.55 width:0.49 height:0.91] Task: [Q] Are there keys and a wallet on the countertop?",
        2: " Scene: [cabinet_1 x:0.48 y:0.55 width:0.49 height:0.91] Task: [A] Move up and then check the top of the cabinet, if you see a book, take a picture of it.",
        3: " Scene: [bed_1 x:0.48 y:0.55 width:0.49 height:0.91] Task: [A] Move down and try to find the toy under the bed.",
        4: " Scene: [table_1 x:0.28 y:0.15 width:0.2 height:0.19], Task: [A] Can you find scissors for cutting paper on the table?",
        5: " Scene: [aerial_robot x:0.48 y:0.51 width:0.2 height:0.4] Task: [E] You provided me with plan mf,100, but I encountered a failure because a aerial robot is approaching me. Execution history: mf(50)",
        6: " Scene: [dog x:0.48 y:0.51 width:0.2 height:0.4] Task: [E] You provided me with plan ml,100, but I encountered a failure because a dog is attempting to catch me. Execution history: ml(50)",
        7: " Scene: [boy x:0.48 y:0.51 width:0.2 height:0.4] Task: [E] You provided me with plan mf,100, but I encountered a failure because a boy is passing by. Execution history: mf(50)",

    }
    
    plan = {
        0: " ?s('cat')==True{g('cat')}->False",
        1: " ?iv('keys')&iv('wallet')->{l('Yes');->True}l('No');->False",
        2: " mu(40);?iv('book')==True{tp();->True}->False",
        3: " md(40);?iv('toy')==True{g('toy')}->False;",
        4: " ?s('scissors')==True{g('scissors')}->False;",
        5: "mr(50);mf(50);",
        6: "mu(50);ml(50);",
        7: "d(3);mf(50);"

    }

    tuf = {
        0: [1, 1, 1.5],
        1: [1, 1, 1.5],
        2: [1, 1, 1.5],
        3: [1, 1, 1.5],
        4: [1, 1, 1.5],
        5: [2, 0.2, 0.5],
        6: [2, 0.2, 0.5],
        7: [2, 0.2, 0.5]
    }


    plan_time = [0.295053, 0.241694, 0.454337, 0.332765, 0.349534, 0.29676, 0.168722, 0.168722, 0.168432]

    # -------------------------------- robot arm ------------------------------------------ #
    description = {
    0: " Scene: [], Task: [A] Pick up the red box and drop it on the blue box. Then, pick up the green box and drop it on the red box.",
    1: " Scene: [], Task: [A] Pick up the pencil and drop it in the pen holder. Then, pick up the eraser and drop it near the pencil. Finally, pick up the notebook and drop it on the desk.",
    2: " Scene: [], Task: [A] Pick up the apple and drop it near the carrot. Then, pick up the tomato and drop it near the cucumber. Finally, pick up the orange and drop it near the lemon."
    }
    
    plan = {
        0: "?sc('red box')==True{gt('red box');pi('red box');?sc('blue box')==True{gt('blue box');dr();};?sc('green box')==True{gt('green box');pi('green box');?sc('red box')==True{gt('red box');dr();};};",
        1: "?sc('pencil')==True{gt('pencil');pi('pencil');?sc('pen holder')==True{dr();};}?sc('eraser')==True{gt('eraser');pi('eraser');?sc('pencil')==True{gt('pencil');d(0.1);dr();};}?sc('notebook')==True{gt('notebook');pi('notebook');?sc('desk')==True{gt('desk');dr();};}",
        2: "?sc('apple')==True{gt('apple');pi('apple');?sc('carrot')==True{gt('carrot');dr();};}?sc('tomato')==True{gt('tomato');pi('tomato');?sc('cucumber')==True{gt('cucumber');dr();};}?sc('orange')==True{gt('orange');pi('orange');?sc('lemon')==True{gt('lemon');dr();};}"
    }

    tuf = {
        0: [1, 1, 1.5],
        1: [1, 1, 1.5],
        2: [1, 1, 1.5]
    }


    plan_time = [1.198629, 1.761111, 1.676002]

    # -------------------------------- chatbot ------------------------------------------ #
    description = {
    0: " You are in Time Square in New York . You get an unknown call . Everything the person on the line says starts to unfold around you as if predicting the future .",
    1: " How to make a peanut butter and jelly sandwich?",
    2: " A story about a modern-day Roman Empire."
    }

    tuf = {
        0: [1, 1, 1.5],
        1: [1, 1, 1.5],
        2: [1, 1, 1.5]
    }

    plan_time = [125.563044, 89.692, 112.657375]

    # -------------------------------- robot arm - FLTRNN------------------------------------------ #
    description = {
    0: " Scene: [red_box x:0.48 y:0.55 width:0.49 height:0.91], Task: [A] Pick up the red box and drop it on the blue box. Then, pick up the green box and drop it on the red box.",
    1: " Scene: [pencil x:0.48 y:0.55 width:0.1 height:0.2], Task: [A] Pick up the pencil and drop it in the pen holder. Then, pick up the eraser and drop it near the pencil. Finally, pick up the notebook and drop it on the desk.",
    2: " Scene: [apple:0.48 y:0.55 width:0.49 height:0.35], Task: [A] Pick up the apple and drop it near the carrot. Then, pick up the tomato and drop it near the cucumber. Finally, pick up the orange and drop it near the lemon."
    }
    
    plan = {
        0: "def task(): move_to(object_x('red_box'), object_y('red_box')); take('red_box'); move_to(object_x('blue_box'), object_y('blue_box')); drop('red_box'); move_to(object_x('green_box'), object_y('green_box')); take('green_box'); move_to(object_x('red_box'), object_y('red_box')); drop('green_box');",
        1: "def task(): move_to(object_x('pencil'), object_y('pencil')); take('pencil'); move_to(object_x('pen_holder'), object_y('pen_holder')); drop('pencil'); move_to(object_x('eraser'), object_y('eraser')); take('eraser'); move_to(object_x('pencil'), object_y('pencil')); drop('eraser'); move_to(object_x('notebook'), object_y('notebook')); take('notebook'); move_to(object_x('desk'), object_y('desk')); drop('notebook'); # done",
        2: "def task(): \nmove_to(object_x('apple'), object_y('apple')); take('apple'); move_to(object_x('carrot'), object_y('carrot')); drop('apple'); move_to(object_x('tomato'), object_y('tomato')); take('tomato'); move_to(object_x('cucumber'), object_y('cucumber')); drop('tomato'); move_to(object_x('orange'), object_y('orange')); take('orange'); move_to(object_x('lemon'), object_y('lemon')); drop('orange'); # done"
    }

    tuf = {
        0: [1, 1, 1.5],
        1: [1, 1, 1.5],
        2: [1, 1, 1.5]
    }


    plan_time = [1.693300, 2.021555, 1.931211]

    # -------------------------------- robot dog - TypeFly------------------------------------------ #
    description = {
    0: " Scene: [], Task: [A] Could you find a garbage can? If so, go to it.",
    1: " Scene: [door: x:0.48 y:0.55 width:0.49 height:0.91], Task: [A] Can you find a door? If you can, go to it.",
    2: " Scene: [], Task: [A] First, go to the yard. Then, if you see someone there, make a sound."   
    }
    
    plan = {
        0: "?s('garbage can')==True{g('garbage can')};",
        1: "?iv('door')==True{g('door')};",
        2: "?iv('yard')==True{g('yard');?iv('person')==True{sd()}->True}->False"
    }

    tuf = {
        0: [1, 1, 1.5],
        1: [1, 1, 1.5],
        2: [1, 1, 1.5]
    }


    plan_time = [0.286791, 0.234312, 0.441306]

    # -------------------------------- start ------------------------------------------ #

    virtualdelay = ActionDelay()
    # task_exe_time = virtualdelay.delay_map(plan)
    # print(task_exe_time)

    task_traces = []
    for trace_id in range(len(description)):
        task_input = description[trace_id].strip()
        task_plan = plan[trace_id].strip()
        # task_exe_time, match_list = virtualdelay.delay_map(task_plan)
        task_exe_time, match_list = virtualdelay.delay_map(task_plan, robot_type='robotdog', robot_system='typefly')
        task_tuf = tuf[trace_id]
        task_traces.append({
            'trace_id': trace_id,
            'robot_type': 'robotdog',
            'robot_system': 'typefly',
            'task_input': task_input,
            'task_plan': task_plan,
            'plan_time': plan_time[trace_id],
            'task_exe_time': task_exe_time,
            'exe_time_detail': match_list,
            'task_tuf': task_tuf
        })
    
    print(task_traces)
    output_file = './dataset/trace_set_robotdog.json'

    # Write the task traces to a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(task_traces, json_file, indent=4)

    # output example:
    task_traces = [
        {'trace_id': 0, 'task_input': 'Scene: [], Task: [A] Go and take a picture of the table.', 'task_plan': "?_1=s('table')==True{g(_1);tp}", 'plan_time':0.295053, 'task_exe_time': 8.611, 'exe_time_detail': [('tp', 0.001), ('g(', 2.5), ('s(', 6.11)]}, 
        {'trace_id': 1, 'task_input': 'Scene: [] Task: [A] Could you find the cat? If so, go to it.', 'task_plan': "?s('cat')==True{g('cat')}->False", 'plan_time':0.241694, 'task_exe_time': 7.0825, 'exe_time_detail': [('g(', 2.5), ('s(', 4.5825)]}, 
        {'trace_id': 2, 'task_input': 'Scene: [countertop_1 x:0.48 y:0.55 width:0.49 height:0.91] Task: [Q] Are there keys and a wallet on the countertop?', 'task_plan': "?iv('keys')&iv('wallet')->{l('Yes');->True}l('No');->False", 'plan_time':0.454337, 'task_exe_time': 0.004, 'exe_time_detail': [('iv', 0.001), ('iv', 0.001), ('l', 0.001), ('l', 0.001)]}, 
        {'trace_id': 3, 'task_input': 'Scene: [cabinet_1 x:0.48 y:0.55 width:0.49 height:0.91] Task: [A] Move up and then check the top of the cabinet, if you see a book, take a picture of it.', 'task_plan': "mu(40);?iv('book')==True{tp();->True}->False", 'plan_time':0.332765, 'task_exe_time': 2.4447, 'exe_time_detail': [('mu', 2.4427), ('iv', 0.001), ('tp', 0.001)]}, 
        {'trace_id': 4, 'task_input': 'Scene: [bed_1 x:0.48 y:0.55 width:0.49 height:0.91] Task: [A] Move down and try to find the toy under the bed.', 'task_plan': "md(40);?iv('toy')==True{g('toy')}->False;", 'plan_time':0.349534, 'task_exe_time': 4.9437, 'exe_time_detail': [('md', 2.4427), ('iv', 0.001), ('g(', 2.5)]}, 
        {'trace_id': 5, 'task_input': 'Scene: [table_1 x:0.28 y:0.15 width:0.2 height:0.19], Task: [A] Can you find something for cutting paper on the table?', 'task_plan': "?s('scissors')==True{g('scissors')}->False;", 'plan_time':0.29676, 'task_exe_time': 5.555, 'exe_time_detail': [('g(', 2.5), ('s(', 3.055)]}, 
        {'trace_id': 6, 'task_input': 'Scene: [] Task: [A] Turn around and go to the table behind you', 'task_plan': 'tc(180);rp;', 'plan_time':0.134937, 'task_exe_time': 3.755, 'exe_time_detail': [('tc', 3.754), ('rp', 0.001)]}, 
        {'trace_id': 7, 'task_input': 'Scene: [] Task: [A] Can you find something for me to eat? If you can, go for it and return. Otherwise, find and go to something drinkable.', 'task_plan': "?_1=sa('Any edible target here?');?_1!=False{g(_1);->True}->?_2=sa('Any drinkable target here?');?_2!=False{g(_2)};", 'plan_time':0.900123, 'task_exe_time': 9.7025, 'exe_time_detail': [('g(', 2.5), ('g(', 2.5), ('sa(', 1.5675), ('sa(', 3.135)]}, 
        {'trace_id': 8, 'task_input': 'Scene: [] Task: [A] Find a chair with a laptop on it.', 'task_plan': "_1=sa('Any chair with a laptop on it?');?_1!=False{g(_1)}", 'plan_time':0.455442, 'task_exe_time': 4.0675, 'exe_time_detail': [('g(', 2.5), ('sa(', 1.5675)]}
        ]