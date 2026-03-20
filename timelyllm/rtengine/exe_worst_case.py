import time
import re

class ExeWorstEst:
    def __init__(self):
        pass

    def delay_wc_map(self, command, robot_type='drone'):
        value = self.extract_value(command)
        delay = 0

        if robot_type == 'drone':
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
                r'(?<![tr])\bp\b': lambda _: 0.041,
                r'(?<!m)\bl\b': lambda _: 0.001,
                r'(?<!m)\bd\b': lambda v: v,
                r'\btp\b': lambda _: 0.001,
                r'\brp\b': lambda _: 0.001,
                r'\bg\(': lambda _: 2.5,
                r'\bs\(': lambda _: 1.5275,
                r'\bsa\(': lambda _: 1.5675
            }
        elif robot_type == 'robotarm':
            # unit:s
            keywords = {
                r'\box\b': lambda _: 0.001,
                r'\boy\b': lambda _: 0.001,
                r'\bow\b': lambda _: 0.001,
                r'\boh\b': lambda _: 0.001,
                r'\bod\b': lambda _: 0.001,
                r'\biv\b': lambda _: 0.001,
                r'(?<![tr])\bp\b': lambda _: 0.041,
                r'(?<!m)\bl\b': lambda _: 0.001,
                r'(?<!m)\bd\b': lambda v: v,
                r'\btp\b': lambda _: 0.001,
                r'\brp\b': lambda _: 0.001,
                r'\bg\(': lambda _: 2.5,
                r'\bs\(': lambda _: 1.5275,
                r'\bsa\(': lambda _: 1.5675,
                r'\bgt\b': lambda _: 0.8,  # goto
                r'\bpi\b': lambda _: 3, # pick
                r'\bdr\b': lambda _: 3, # drop
                r'\bsc\b': lambda _: 1,  # scan
                # for robot-arm-fltrnn
                r'\bmove_to\b': lambda _: 1,  # move_to
                r'\btake\b': lambda _: 3, # take
                r'\bdrop\b': lambda _: 3 # drop
            }
        elif robot_type == 'robotdog':
            keywords = {
                r'\bm\b': lambda v: v * 0.01125 + 0.375,
                r'\bmf\b': lambda v: v * 0.01125 + 0.375,
                r'\bmb\b': lambda v: v * 0.01125 + 0.375,
                r'\bml\b': lambda v: v * 0.0205 + 0.42,
                r'\bmr\b': lambda v: v * 0.0205 + 0.42,
                r'\br\b': lambda v: v * 0.0111 + 0.3,
                r'\btc\b': lambda v: v * 0.0111 + 0.3,
                r'\btu\b': lambda v: v * 0.0111 + 0.3,
                r'\box\b': lambda _: 0.001,
                r'\boy\b': lambda _: 0.001,
                r'\bow\b': lambda _: 0.001,
                r'\boh\b': lambda _: 0.001,
                r'\bod\b': lambda _: 0.001,
                r'\biv\b': lambda _: 0.001,
                r'(?<!m)\bl\b': lambda _: 0.001,
                r'(?<!m)\bd\b': lambda v: v,
                r'\btp\b': lambda _: 0.001,
                r'\brp\b': lambda _: 0.001,
                r'\bg\(': lambda _: 0.9375,
                r'\bs\(': lambda _: 0.8005,
                r'\bsd\(': lambda _: 0.8405
                # r'\bsd\(': lambda _: 1    # sound
            }
        elif robot_type == 'robotcar':
            # speed = 0.3 m/s
            keywords = {
                r'\bm\b': lambda v: v * 0.0425 - 0.385,
                r'\bmf\b': lambda v: v * 0.0425 - 0.385,
                r'\bmb\b': lambda v: v * 0.0425 - 0.385,
                r'\bml\b': lambda v: v * 0.0388 + 0.196,
                r'\bmr\b': lambda v: v * 0.0388 + 0.196,
                r'\br\b': lambda v: v * 0.0051 + 0.94,
                r'\btc\b': lambda v: v * 0.0051 + 0.94,
                r'\btu\b': lambda v: v * 0.0051 + 0.94,
                r'\box\b': lambda _: 0.001,
                r'\boy\b': lambda _: 0.001,
                r'\bow\b': lambda _: 0.001,
                r'\boh\b': lambda _: 0.001,
                r'\bod\b': lambda _: 0.001,
                r'\biv\b': lambda _: 0.001,
                r'(?<![tr])\bp\b': lambda _: 0.041,
                r'(?<!m)\bl\b': lambda _: 0.001,
                r'(?<!m)\bd\b': lambda v: v,
                r'\btp\b': lambda _: 0.001,
                r'\brp\b': lambda _: 0.001,
                r'\bg\(': lambda _: 1.74,
                r'\bs\(': lambda _: 1.1705,
                r'\bsa\(': lambda _: 1.2105,
                r'\bsd\(': lambda _: 1.2105
            }
        else:
            print("Warning: Add support for new robot type")

        for key, func in keywords.items():
            matches = re.findall(key, command)
            for match in matches:
                # print(match)
                delay += func(value)
            # if re.search(key, command):
            #     print(key)
            #     delay += func(value)

        return round(delay, 6)
        
    def extract_value(self, command):
        match = re.search(r'\((\d+)\)', command)
        if match:
            return int(match.group(1))
        return 0 
    
    def chat_delay_map(self, command):
        # for chatbot, the reading speed is 5 words per second
        # word_count = len(command.split())
        # reading_speed = 5 # 5 words per second
        # reading_time = word_count / reading_speed
        # return round(reading_time, 6)

        # for speech chatbot
        word_count = len(command.split())
        reading_speed = 3 # 5 words per second
        reading_time = word_count / reading_speed
        return round(reading_time, 6)