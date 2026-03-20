import matplotlib.pyplot as plt
# from executor.virtual_wrapper import ActionDelay
import json
import numpy as np
import math
# from IPython.display import Image


# FLTRNN
normal_file = './timelyllm/logs/exp744_fltrnn_vllm_data.json' # vllm
interpreter_file = './timelyllm/logs/exp744_fltrnn_timelyllm_data.json' # our method
data_sample_file = './dataset/data_sample_robotarm_fltrnn.json'


with open(data_sample_file, 'r') as file:
    data = json.load(file)

def task_type_map(task_id):
    for job in data:
        if job['job_id'] == int(task_id):
            return job['trace_id']
    return None

class TUF:
    def __init__(self,request_list_path):
        self.taskdata = {}
        with open(request_list_path, 'r') as file:
            self.task_database = json.load(file)
    
    def read_tuf(self, task_id):
        for task_item in self.task_database:
            if task_item['job_id'] == int(task_id):
                return task_item['tuf']
            
    def tuf(self, act_time, task_id):
        tuf_set = self.read_tuf(task_id)
        if act_time < tuf_set[1]:
            utility = tuf_set[0]
        else:
            utility = tuf_set[0] * (act_time-tuf_set[2])/(tuf_set[1]-tuf_set[2])
            #(5-10*act_time)/3
        return utility

def read_data(filename):
    tasks = []
    response_times = []
    response_times_type1 = []
    response_times_type2 = []
    average_response_times = []
    response_time_sums = []
    task_waiting_times = {}
    task_response_times = {}
    task_utility = {}
    # virtualdelay = ActionDelay()
    TUFReader = TUF(data_sample_file)
    with open(filename, 'r') as file:
        data = json.load(file)
        for task in data:
            task_id = task["Task"]
            plan = task["Plan"]
            # base_time = virtualdelay.delay_map(plan)
            response_time = float(task["Response Time"].split()[0])
            subplan_response_times = [float(sp["response"]) for sp in task["SubPlan Time"][1:]]
            response_time_sum = sum(subplan_response_times) + response_time

            tasks.append(task_id)
            response_times.append(response_time)
            response_time_sums.append(response_time_sum)
            
            task_type_id = task_type_map(task_id)
            if task_type_id in [0,1,2,3,4]:
                response_times_type1.append(response_time)
            elif task_type_id in [5,6,7]:
                response_times_type2.append(response_time)

            if task_type_id not in task_waiting_times:
                task_waiting_times[task_type_id] = []
            task_waiting_times[task_type_id].append(response_time_sum)

            if task_type_id not in task_response_times:
                task_response_times[task_type_id] = []
            task_response_times[task_type_id].append(response_time)

            utility = TUFReader.tuf(response_time, task_id)
            if task_type_id not in task_utility:
                task_utility[task_type_id] = []
            task_utility[task_type_id].append(utility)

    average_waiting_times = {task_type_id: sum(times) / len(times) for task_type_id, times in task_waiting_times.items()}
    average_response_times = {task_type_id: sum(times) / len(times) for task_type_id, times in task_response_times.items()}
    average_utility = {task_type_id: sum(times) / len(times) for task_type_id, times in task_utility.items()}

    std_waiting_times = {
    task_type_id: math.sqrt(sum((x - average_waiting_times[task_type_id]) ** 2 for x in times) / len(times))
    for task_type_id, times in task_waiting_times.items()
    }

    std_response_times = {
        task_type_id: math.sqrt(sum((x - average_response_times[task_type_id]) ** 2 for x in times) / len(times))
        for task_type_id, times in task_response_times.items()
    }

    std_utility = {
        task_type_id: math.sqrt(sum((x - average_utility[task_type_id]) ** 2 for x in times) / len(times))
        for task_type_id, times in task_utility.items()
    }

    return tasks, response_times, response_time_sums, response_times_type1, response_times_type2, average_waiting_times, average_response_times, average_utility, std_waiting_times, std_response_times, std_utility


# Reading data from the files
normal_tasks, normal_response_times, normal_completion_times,normal_response_times_type1, normal_response_times_type2, normal_average_waiting_time, normal_average_response_time, normal_average_utility, normal_std_waiting_times, normal_std_response_times, normal_std_utility = read_data(normal_file)
# vLLM-stream experiment removed (only vLLM vs TickingLLM)
interpret_tasks, interpret_response_times, interpret_completion_times, interpret_response_times_type1, interpret_response_times_type2, interpret_average_waiting_time, interpret_average_response_time, interpreter_average_utility, interpreter_std_waiting_times, interpreter_std_response_times, interpreter_std_utility = read_data(interpreter_file)
# interpret_tasks2, interpret_response_times2, interpret_completion_times2 = read_data('./logs/09-03-09_data.txt')


colors = plt.get_cmap('tab20c').colors

fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
# Sorting the keys and values by Task ID
sorted_keys = sorted(normal_average_utility.keys())
sorted_normal_values = [normal_average_utility[key] for key in sorted_keys]
sorted_interpret_values = [interpreter_average_utility[key] for key in sorted_keys]

bar_width = 0.3
keys = list(normal_average_utility.keys())

# Plotting bars
bars1 = plt.bar([x - bar_width/2 for x in range(len(sorted_keys))],
                sorted_normal_values,
                width=bar_width, color=colors[7], label='vLLM', align='center', hatch='//', edgecolor='white')
bars2 = plt.bar([x + bar_width/2 for x in range(len(sorted_keys))],
                sorted_interpret_values,
                width=bar_width, color=colors[5], label='TickingLLM', align='center', hatch='-', edgecolor='white')

# Adding value labels on top of each bar
for bar in bars1:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)

for bar in bars2:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)
    
plt.xticks(range(len(sorted_keys)), ['Object Stack', 'Desk Clean', 'Item Classify'], fontsize=25, rotation=10)
plt.yticks(size=25)
plt.rcParams["legend.title_fontsize"] = 25
plt.legend(title="Robotic System: FLTRNN", fontsize=25)
# plt.xlabel('Task Type ID', size=22)
plt.ylabel('Utility', size=25)
# plt.title('Average Response Time Utility per Task Type', size=22)
# plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, zorder=1)
plt.tight_layout()
plt.savefig('avg_utility_per_task_robot_arm_fltrnn.jpg')


# --- Plot 2: Waiting Time ---
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

sorted_normal_wait = [normal_average_waiting_time[key] * 1000 for key in sorted_keys]
sorted_interpret_wait = [interpret_average_waiting_time[key] * 1000 for key in sorted_keys]
sorted_normal_wait_err = [normal_std_waiting_times[key] * 1000 for key in sorted_keys]
sorted_interpret_wait_err = [interpreter_std_waiting_times[key] * 1000 for key in sorted_keys]

bar_width = 0.3
colors_wait = plt.get_cmap('tab20c').colors

bars1 = plt.bar([x - bar_width/2 for x in range(len(sorted_keys))],
                sorted_normal_wait, width=bar_width, color='none',
                yerr=sorted_normal_wait_err, capsize=5, ecolor='gray',
                label='vLLM', align='center',
                edgecolor=colors_wait[3], linewidth=4, hatch='//')

bars2 = plt.bar([x + bar_width/2 for x in range(len(sorted_keys))],
                sorted_interpret_wait, width=bar_width, color='none',
                yerr=sorted_interpret_wait_err, capsize=5, ecolor='gray',
                label='TimeLyLLM', align='center',
                edgecolor=colors_wait[0], linewidth=4, hatch='-')

for bar in bars1:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=10)

plt.xticks(range(len(sorted_keys)), ['Object Stack', 'Desk Clean', 'Item Classify'], fontsize=25, rotation=10)
plt.yticks(size=25)
plt.rcParams["legend.title_fontsize"] = 25
plt.legend(title="Robotic System: FLTRNN", fontsize=25)
plt.ylabel('Time (ms)', size=25)
plt.tight_layout()
plt.savefig('avg_delay_per_task_robot_arm_fltrnn.jpg')


# --- Statistics ---
def print_stats(metric_name, vllm_dict, tlm_dict, scale=1, unit='', higher_is_better=False):
    keys = sorted_keys
    vllm_avg = np.mean([vllm_dict[k] * scale for k in keys])
    tlm_avg = np.mean([tlm_dict[k] * scale for k in keys])
    if higher_is_better:
        diff = (tlm_avg - vllm_avg) / abs(vllm_avg) * 100 if vllm_avg != 0 else float('inf')
        diff_label = f"improvement={diff:+.1f}%"
    else:
        diff = (vllm_avg - tlm_avg) / vllm_avg * 100 if vllm_avg != 0 else float('inf')
        diff_label = f"reduction={diff:.1f}%"
    print(f"  {metric_name:22s}: vLLM={vllm_avg:.1f}{unit}, TimeLyLLM={tlm_avg:.1f}{unit}, {diff_label}")

print("=" * 60)
print_stats("Time Utility", normal_average_utility, interpreter_average_utility, higher_is_better=True)
print_stats("Waiting Time", normal_average_waiting_time, interpret_average_waiting_time, scale=1000, unit='ms')
print("=" * 60)