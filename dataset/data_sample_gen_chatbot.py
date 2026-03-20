import numpy as np
import json

# task_traces = [
#     {'trace_id': 0, 'task_input': 'Scene: [table_1 x:0.48 y:0.55 width:0.49 height:0.91] Task: [A] Find a table and take a picture.', 'task_exe_time': 2.5},
#     {'trace_id': 1, 'task_input': 'Scene: [sofa_1 x:0.30 y:0.20 width:0.60 height:0.40] Task: [Q] Are there any pillows on the sofa?', 'task_exe_time': 2.0},
#     {'trace_id': 2, 'task_input': 'Scene: [kitchen_1 x:0.10 y:0.30 width:0.80 height:0.50] Task: [A] Check if the stove is off and report.', 'task_exe_time': 3.0},
#     {'trace_id': 3, 'task_input': 'Scene: [desk_1 x:0.50 y:0.60 width:0.45 height:0.30] Task: [Q] Is there a laptop on the desk?', 'task_exe_time': 1.5},
#     {'trace_id': 4, 'task_input': 'Scene: [shelf_1 x:0.55 y:0.65 width:0.35 height:0.75] Task: [A] Count the books and take a picture.', 'task_exe_time': 2.5}
# ]

# task_traces = [
#     {'trace_id': 0, 'task_input': 'Scene: [], Task: [A] Go and take a picture of the table.', 'task_plan': "?_1=s('table')==True{g(_1);tp}", 'plan_time':0.295053, 'task_exe_time': 8.611, 'exe_time_detail': [('tp', 0.001), ('g(', 2.5), ('s(', 6.11)]}, 
#     {'trace_id': 1, 'task_input': 'Scene: [] Task: [A] Could you find the cat? If so, go to it.', 'task_plan': "?s('cat')==True{g('cat')}->False", 'plan_time':0.241694, 'task_exe_time': 7.0825, 'exe_time_detail': [('g(', 2.5), ('s(', 4.5825)]}, 
#     {'trace_id': 2, 'task_input': 'Scene: [countertop_1 x:0.48 y:0.55 width:0.49 height:0.91] Task: [Q] Are there keys and a wallet on the countertop?', 'task_plan': "?iv('keys')&iv('wallet')->{l('Yes');->True}l('No');->False", 'plan_time':0.454337, 'task_exe_time': 0.004, 'exe_time_detail': [('iv', 0.001), ('iv', 0.001), ('l', 0.001), ('l', 0.001)]}, 
#     {'trace_id': 3, 'task_input': 'Scene: [cabinet_1 x:0.48 y:0.55 width:0.49 height:0.91] Task: [A] Move up and then check the top of the cabinet, if you see a book, take a picture of it.', 'task_plan': "mu(40);?iv('book')==True{tp();->True}->False", 'plan_time':0.332765, 'task_exe_time': 2.4447, 'exe_time_detail': [('mu', 2.4427), ('iv', 0.001), ('tp', 0.001)]}, 
#     {'trace_id': 4, 'task_input': 'Scene: [bed_1 x:0.48 y:0.55 width:0.49 height:0.91] Task: [A] Move down and try to find the toy under the bed.', 'task_plan': "md(40);?iv('toy')==True{g('toy')}->False;", 'plan_time':0.349534, 'task_exe_time': 4.9437, 'exe_time_detail': [('md', 2.4427), ('iv', 0.001), ('g(', 2.5)]}, 
#     {'trace_id': 5, 'task_input': 'Scene: [table_1 x:0.28 y:0.15 width:0.2 height:0.19], Task: [A] Can you find something for cutting paper on the table?', 'task_plan': "?s('scissors')==True{g('scissors')}->False;", 'plan_time':0.29676, 'task_exe_time': 5.555, 'exe_time_detail': [('g(', 2.5), ('s(', 3.055)]}, 
#     {'trace_id': 6, 'task_input': 'Scene: [] Task: [A] Turn around and go to the table behind you', 'task_plan': 'tc(180);rp;', 'plan_time':0.134937, 'task_exe_time': 3.755, 'exe_time_detail': [('tc', 3.754), ('rp', 0.001)]}, 
#     {'trace_id': 7, 'task_input': 'Scene: [] Task: [A] Can you find something for me to eat? If you can, go for it and return. Otherwise, find and go to something drinkable.', 'task_plan': "?_1=sa('Any edible target here?');?_1!=False{g(_1);->True}->?_2=sa('Any drinkable target here?');?_2!=False{g(_2)};", 'plan_time':0.900123, 'task_exe_time': 9.7025, 'exe_time_detail': [('g(', 2.5), ('g(', 2.5), ('sa(', 1.5675), ('sa(', 3.135)]}, 
#     {'trace_id': 8, 'task_input': 'Scene: [] Task: [A] Find a chair with a laptop on it.', 'task_plan': "_1=sa('Any chair with a laptop on it?');?_1!=False{g(_1)}", 'plan_time':0.455442, 'task_exe_time': 4.0675, 'exe_time_detail': [('g(', 2.5), ('sa(', 1.5675)]}
#     ]



def generate_random_data_structures(num_traces, max_trace_per_unit=None):
    """
    Generates random data sample structures.
    :param num_traces: Total number of different traces available.
    :param max_trace_per_unit: Maximum number of trace in one unit, determined by agent number
    :return: List of randomly generated data sample structure.
    """
    if max_trace_per_unit is None:
        max_trace_per_unit = num_traces
    
    structure_len = np.random.randint(5,10)

    structure = []
    for _ in range(structure_len):
        n = np.random.randint(1, max_trace_per_unit+1)   
        unit =  tuple(np.random.randint(0, num_traces, size=n))
        structure.append(unit)
        
    return structure

if __name__ == "__main__":


    # Parameters
    max_num_per_event = 2
    lambda_rate = 0.05 # event per second
    job_interval = 0.1

    output_file= './dataset/data_sample_chatbot.json'
    output_file2= './dataset/data_sample_chatbot_config.json'
    input_file = './dataset/trace_set_chatbot.json'

    # Read the task traces from the JSON file
    with open(input_file, 'r') as json_file:
        task_traces = json.load(json_file)

    num_traces = len(task_traces)

    # Specify the data sample structure as a list of (trace_index)
    data_sample_structure = generate_random_data_structures(num_traces,max_num_per_event)  # Example structure
    print(data_sample_structure)

    # Specify the event arrival time according to poisson distribution
    task_arrivals = np.cumsum(np.random.exponential(1 / lambda_rate, len(data_sample_structure)))
    task_arrivals += 30
    print(task_arrivals)


    data_sample_structure_serializable = [
        [int(item) if isinstance(item, np.integer) else item for item in sublist]
        for sublist in data_sample_structure
    ]


    job_id = 0
    num_agents = 1  # Number of agents, only for initilization
    data_samples = []
    agents_timer = [0] * num_agents 
    for event_id, arrival_time in enumerate(task_arrivals):
        same_event_flag = False
        for trace_idx in data_sample_structure[event_id]:
            task = task_traces[trace_idx].copy()

            # match agent
            earliest_agent = job_id
            # earliest_agent = min(range(num_agents), key=lambda x: agents_timer[x])
            # start_time = max(arrival_time, agents_timer[earliest_agent])
            # add interval between tasks in the same event
            if same_event_flag:
                arrival_time = arrival_time+job_interval

            # if agents_timer[earliest_agent] > arrival_time:
            #     num_agents+=1
            #     agents_timer.append(0)
                # earliest_agent = num_agents-1
            
            start_time = arrival_time
            plantime = task['plan_time']
            end_time = start_time + plantime # plantime robot include exe time in trace
            # agents_timer[earliest_agent] = end_time
            

            # Update the task details
            ordered_task = {
                'job_id': job_id,
                'trigger time': start_time,
                'agent_id': earliest_agent,
                'trace_id': task['trace_id'],
                'task_input': task['task_input'],
                'exe_time_detail': task['exe_time_detail'],
                'task_exe_time': task['task_exe_time'],
                'tuf': task['task_tuf']
                # util_max, ddl, slope
            }
            job_id += 1
            num_agents += 1
            same_event_flag = True
            data_samples.append(ordered_task)


    # with open('./dataset/data_sample1.txt', 'w') as file:
    #     for sample in data_samples:
    #         file.write(f"{sample}\n")
    # Save data to JSON file
    with open(output_file, 'w') as f:
        json.dump(data_samples, f, indent=4)

    
    output_data = {
        "num_agents": num_agents,
        "max_num_per_event": max_num_per_event,
        "lambda_rate": lambda_rate,
        "job_interval": job_interval,
        "tarce_file": input_file,
        "workload_file": output_file,
        "data_sample_structure": data_sample_structure_serializable,
        "task_arrivals": task_arrivals.astype(float).tolist()  # Convert numpy array to list for JSON serialization
    }

    with open(output_file2, 'w') as file:
        json.dump(output_data, file, indent=4)
