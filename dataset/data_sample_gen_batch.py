import numpy as np
import json


def generate_task_trace(task_traces, max_batch_size=16, min_batch_interval=5):
    num_traces = len(task_traces)
    job_id = 0
    data_samples = []
    
    trigger_time = 30.00000  # Initial trigger time
    agent_id = 0
    
    batch_size = 1
    while batch_size <= max_batch_size:
        batch_tasks = [0] * batch_size  # All use the first trace
        
        task_id = 0
        for trace_idx in batch_tasks:
            task = task_traces[trace_idx].copy()
            trigger_time_new = trigger_time + 0.0001 * task_id
            task_id += 1
            
            ordered_task = {
                'job_id': job_id,
                'trigger time': float(trigger_time_new),
                'agent_id': agent_id,
                'trace_id': task['trace_id'],
                'task_input': task['task_input'],
                'exe_time_detail': task['exe_time_detail'],
                'task_exe_time': task['task_exe_time'],
                'tuf': task['task_tuf']
            }
            data_samples.append(ordered_task)
            job_id += 1
            agent_id += 1  # Ensure each task is assigned to a different agent
        
        trigger_time += min_batch_interval  # Ensure at least 5s interval between batches
        batch_size += 1
    
    return data_samples, agent_id


if __name__ == "__main__":
    input_file = './dataset/trace_set_batch.json'
    output_file = './dataset/data_sample_batch.json'
    output_config_file = './dataset/data_sample_batch_config.json'
    
    # Read task traces
    with open(input_file, 'r') as json_file:
        task_traces = json.load(json_file)
    
    data_samples, num_agents = generate_task_trace(task_traces)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(data_samples, f, indent=4)

    
    output_data = {
        "num_agents": num_agents,
        "max_num_per_event": 16,
        "lambda_rate": None,
        "job_interval": None,
        "tarce_file": input_file,
        "workload_file": output_file
    }
    
    with open(output_config_file, 'w') as file:
        json.dump(output_data, file, indent=4)
