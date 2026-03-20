from typing import List, Tuple, Union
import re
from enum import Enum
import time
from typing import Optional
from threading import Thread
from queue import Queue
from rtengine.skillset import SkillSet
# from skillset import SkillSet

def detect_action(command):
    # unit:s
    keywords = {
        r'\bm\b': lambda v: v * 0.0072 + 1.8144,
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
        r'\bsa\(': lambda _: 1.5675,
        r'\bgt\b': lambda _: 0.8,  # goto
        r'\bpi\b': lambda _: 3, # pick
        r'\bdr\b': lambda _: 3, # drop
        r'\bsc\b': lambda _: 1  # scan
    }

    # Sort the matches by the order they appear in the command
    sorted_matches = sorted(
        (match.start(), key, func) 
        for key, func in keywords.items() 
        for match in re.finditer(key, command)
    )

    if not sorted_matches:
        exe_flag = False
    else:
        exe_flag = True

    return exe_flag

def print_debug(*args):
    print(*args)
    # pass

MiniSpecValueType = Union[int, float, bool, str, None]

class ParsingState(Enum):
    CODE = 0
    ARGUMENTS = 1
    CONDITION = 2
    LOOP_COUNT = 3
    SUB_STATEMENTS = 4

class MiniSpecProgram:
    def __init__(self, env: Optional[dict] = None) -> None:
        self.statements: List[Statement] = []
        self.depth = 0
        self.finished = False
        self.ret = False
        if env is None:
            self.env = {}
        else:
            self.env = env
        self.current_statement = Statement(self.env)

    def parse(self, code_instance: str, exec: bool = False, sub_flag: bool = False) -> bool:
        self.current_statement = Statement(self.env)
        for chunk in code_instance:
            if isinstance(chunk, str):
                code = chunk
            else:
                code = chunk.choices[0].delta.content
            if code == None or len(code) == 0:
                continue
            for c in code:
                if self.current_statement.parse(c, exec):
                    if not sub_flag:
                        if self.current_statement.parsed_code == code_instance:
                            if detect_action(code_instance) > 0:
                                self.current_statement = Statement(self.env)
                                return True
                    else:
                        return True
                        # print_debug("Adding statement: ", self.current_statement, exec)
                        # self.statements.append(self.current_statement)
                        # if not self.current_statement.flag_func:
                        #     continue
                        # else:
                        #     return True
                        # print(f"current statement: {self.current_statement}")
                        # return True, self.current_statement
                    # self.current_statement = Statement(self.env)
                if sub_flag:
                    if c == '{':
                        self.depth += 1
                    elif c == '}':
                        if self.depth == 0:
                            self.finished = True
                            # return True, self.current_statement
                            return True
                        self.depth -= 1
        # return False, str("Not executable")
        return False
    
class Statement:
    # execution_queue: Queue['Statement'] = None
    low_level_skillset: SkillSet = None
    high_level_skillset: SkillSet = None
    def __init__(self, env: dict) -> None:
        self.code_buffer: str = ''
        self.parsing_state: ParsingState = ParsingState.CODE
        self.condition: Optional[str] = None
        self.loop_count: Optional[int] = None
        self.action: str = ''
        self.allow_digit: bool = False
        self.executable: bool = False
        self.ret: bool = False
        self.sub_statements: Optional[MiniSpecProgram] = None
        self.env = env
        self.read_argument: bool = False
        self.flag_func = False
        self.parsed_code: str = ''

    def get_env_value(self, var) -> MiniSpecValueType:
        if var not in self.env:
            raise Exception(f'Variable {var} is not defined')
        return self.env[var]

    def parse(self, code: str, exec: bool = False) -> bool:
        self.parsed_code += code
        for c in code:
            match self.parsing_state:
                case ParsingState.CODE:
                    if c == '?' and not self.read_argument:
                        self.action = 'if'
                        self.parsing_state = ParsingState.CONDITION
                    elif c == ';' or c == '}' or c == ')':
                        if c == ')':
                            self.code_buffer += c
                            self.read_argument = False
                        self.action = self.code_buffer
                        # print_debug(f'SP Action: {self.code_buffer}')
                        self.executable = True
                        if exec and self.action != '':
                            # self.execution_queue.put(self)
                            pass
                        return True
                    else:
                        if c == '(':
                            self.read_argument = True
                        if c.isalpha() or c == '_':
                            self.allow_digit = True
                        self.code_buffer += c
                    if c.isdigit() and not self.allow_digit:
                        self.action = 'loop'
                        self.parsing_state = ParsingState.LOOP_COUNT
                case ParsingState.CONDITION:
                    if c == '{':
                        # print_debug(f'SP Condition: {self.code_buffer}')
                        self.condition = self.code_buffer
                        self.executable = detect_action(self.code_buffer)
                        if exec:
                            # self.execution_queue.put(self)
                            pass
                        self.sub_statements = MiniSpecProgram(self.env)
                        self.parsing_state = ParsingState.SUB_STATEMENTS
                        self.ret = True
                        if self.executable:
                            return True
                    if c == ')' or c == ';':
                        return True
                    else:
                        self.code_buffer += c
                case ParsingState.LOOP_COUNT:
                    if c == '{':
                        # print_debug(f'SP Loop: {self.code_buffer}')
                        self.loop_count = int(self.code_buffer)
                        # except ValueError as e:
                        #     print(f"Error converting to int: {e}")
                        self.executable = True
                        if exec:
                            # self.execution_queue.put(self)
                            pass
                        self.sub_statements = MiniSpecProgram(self.env)
                        self.parsing_state = ParsingState.SUB_STATEMENTS
                    else:
                        self.code_buffer += c
                case ParsingState.SUB_STATEMENTS:
                    if self.sub_statements.parse([c], sub_flag=True):
                        return True
        return False
    

if __name__ == "__main__":
        program = MiniSpecProgram()
        code = "?_1=sa('Any object for cutting paper on the table?')"
        signal = program.parse(code, True)
        print(signal)
