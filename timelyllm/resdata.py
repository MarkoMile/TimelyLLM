# For generate the text summary of the log file
import re
import os
import json

def parse_log(log_data):
    # Dictionary to hold task information
    tasks = {}
    # Regular expressions to match log entries
    gen_req_pattern = re.compile(r"Generating request from task (\d+), time: ([\d\.]+)")
    start_exec_pattern = re.compile(r"Start executing task (\d+) for agent \d+ on time ([\d\.]+) with plan (.*)")
    finish_exec_pattern = re.compile(r"Finish executing task (\d+) for agent \d+ on time ([\d\.]+) with plan (.+)")
    added_task_pattern = re.compile(r"Added task (\d+), time: ([\d.]+)")
    output_task_pattern = re.compile(r"Output for task (\d+): (.*), time: ([\d.]+)")


    for line in log_data:
        # Check for generating request
        line = line.strip()
        gen_match = gen_req_pattern.match(line)
        if gen_match:
            task_id = gen_match.group(1)
            gen_time = float(gen_match.group(2))
            if task_id not in tasks:
                tasks[task_id] = {
                    'start_times': [],
                    'plans': [],
                    'finish_times': [],
                    'gen_time': gen_time,
                    'add_time': [],
                    'output_time':[]
                }
    
        # Check for start executing
        start_match = start_exec_pattern.match(line)
        if start_match:
            task_id = start_match.group(1)
            start_time = float(start_match.group(2))
            plan = start_match.group(3)
            if task_id in tasks:
                tasks[task_id]['start_times'].append(start_time)
                tasks[task_id]['plans'].append(plan)
        
        # Check for finish executing
        finish_match = finish_exec_pattern.match(line)
        if finish_match:
            task_id = finish_match.group(1)
            finish_time = float(finish_match.group(2))
            if task_id in tasks:
                tasks[task_id]['finish_times'].append(finish_time)

        # check for add and output
        added_task_match = added_task_pattern.match(line)
        if added_task_match:
            task_id = added_task_match.group(1)
            time = float(added_task_match.group(2))
            if task_id in tasks:
                tasks[task_id]['add_time'].append(time)
        
        output_task_match = output_task_pattern.match(line)
        if output_task_match:
            task_id = output_task_match.group(1)
            plan = output_task_match.group(2)
            time = float(output_task_match.group(3))
            if task_id in tasks:
                tasks[task_id]['output_time'].append(time)
    
    return tasks

def summarize_log(tasks, output_path):

    # Calculate the metrics for each task and write to file
    output_data = []
    with open(output_path, 'w') as output_file:
        for index, (task_id, info) in enumerate(tasks.items()):
            # if index < 10: # for robot
            if index < 10: # for chatbot
                continue  # Skip the first 10 tasks
            if info['start_times'] and info['finish_times'] and info['output_time']:
                # Only iterate over subplans that have matching start/finish times
                n_subplans = min(len(info['plans']), len(info['start_times']), len(info['finish_times']))
                if n_subplans == 0:
                    continue
                response_time = info['start_times'][0] - info['gen_time']
                completion_time = info['output_time'][-1] - info['gen_time']
                plan = ''.join(info['plans'][:n_subplans])

                subplan_output = []
                for index, subplan in enumerate(info["plans"][:n_subplans]):
                    execution_time = info['finish_times'][index] - info['start_times'][index]
                    if index ==0:
                        subplan_response_time = response_time
                    else:
                        subplan_response_time = info['start_times'][index] - info['finish_times'][index-1]
                    subplan_output.append({"subplan": subplan,
                                            "exe": f"{execution_time:.6f}",
                                            "response": f"{subplan_response_time:.6f}",
                                            })

                output = {
                    "Task": task_id,
                    "Plan": plan,
                    "Response Time": f"{response_time:.6f}",
                    "Completion Time": f"{completion_time:.6f}",
                    "SubPlan Time": subplan_output
                }

                output_data.append(output)
                # print(output)
                
                # output_file.write(output)
        json.dump(output_data, output_file, indent=4)
        


def read_log_file(file_path):
    output_file_path = file_path.replace('.log', '_data.json')
    with open(file_path, 'r') as file:
        log_data = file.readlines()
    tasks = parse_log(log_data)
    summarize_log(tasks, output_file_path)

def read_latest_log_file(directory):
    # List all files in the directory
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.log')]
    # Sort log files by modification time in descending order
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    # Read the most recent log file
    if files:
        log_file_path = files[0]
        output_file_path = log_file_path.replace('.log', '_data.json')
        with open(log_file_path, 'r') as file:
            log_data = file.readlines()
        tasks = parse_log(log_data)
        summarize_log(tasks, output_file_path)
    else:
        print("No log files found in the directory.")


if __name__ == "__main__":       
    # read log file
    file_path = './logs/03-19-02.log'
    read_log_file(file_path)

    # # read latest log file
    # directory = 'logs'
    # read_latest_log_file(directory)