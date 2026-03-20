import matplotlib.pyplot as plt
import json
import numpy as np
import math

# Paths use fixed experiment-name logs (set via --log-name in rtllm.py)
normal_file = './timelyllm/logs/exp741_vllm_high_data.json'
interpreter_file = './timelyllm/logs/exp741_timelyllm_high_data.json'
data_sample_file = './dataset/data_sample_1.json'

with open(data_sample_file, 'r') as file:
    data_hw = json.load(file)

def task_type_map(task_id):
    for job in data_hw:
        if job['job_id'] == int(task_id):
            return job['trace_id']
    return None

class TUF:
    def __init__(self, request_list_path):
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
            utility = tuf_set[0] * (act_time - tuf_set[2]) / (tuf_set[1] - tuf_set[2])
        return utility

def read_data(filename):
    task_waiting_times = {}
    task_response_times = {}
    task_utility = {}
    TUFReader = TUF(data_sample_file)
    with open(filename, 'r') as file:
        data = json.load(file)
        for task in data:
            task_id = task["Task"]
            response_time = float(task["Response Time"].split()[0])
            subplan_response_times = [float(sp["response"]) for sp in task["SubPlan Time"][1:]]
            response_time_sum = sum(subplan_response_times) + response_time

            task_type_id = task_type_map(task_id)

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

    average_waiting_times = {k: sum(v) / len(v) for k, v in task_waiting_times.items()}
    average_response_times = {k: sum(v) / len(v) for k, v in task_response_times.items()}
    average_utility = {k: sum(v) / len(v) for k, v in task_utility.items()}

    std_waiting_times = {
        k: math.sqrt(sum((x - average_waiting_times[k]) ** 2 for x in v) / len(v))
        for k, v in task_waiting_times.items()
    }
    std_response_times = {
        k: math.sqrt(sum((x - average_response_times[k]) ** 2 for x in v) / len(v))
        for k, v in task_response_times.items()
    }

    return average_waiting_times, average_response_times, average_utility, std_waiting_times, std_response_times


# Reading data
normal_avg_waiting, normal_avg_response, normal_avg_utility, normal_std_waiting, normal_std_response = read_data(normal_file)
tlm_avg_waiting, tlm_avg_response, tlm_avg_utility, tlm_std_waiting, tlm_std_response = read_data(interpreter_file)

colors = plt.get_cmap('tab20').colors
sorted_keys = sorted(normal_avg_utility.keys())
bar_width = 0.35


# --- Plot 1: Time Utility ---
fig, ax = plt.subplots(figsize=(14, 4), dpi=200)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

ax.bar([x - bar_width/2 for x in range(len(sorted_keys))],
       [normal_avg_utility[k] for k in sorted_keys],
       width=bar_width, color=colors[3], label='vLLM', align='center', hatch='//', edgecolor='white')

ax.bar([x + bar_width/2 for x in range(len(sorted_keys))],
       [tlm_avg_utility[k] for k in sorted_keys],
       width=bar_width, color=colors[2], label='TimeLyLLM', align='center', hatch='\\', edgecolor='white')

ax.set_xticks(range(len(sorted_keys)))
ax.set_xticklabels(sorted_keys, fontsize=20)
ax.set_xlim(-0.5, len(sorted_keys) - 0.5)
ax.tick_params(axis='y', labelsize=20)
ax.set_ylim(0, 2)
ax.set_xlabel('Task Type ID', fontsize=20)
ax.set_ylabel('Utility', fontsize=20)
ax.legend(fontsize=20, loc='upper left')

plt.tight_layout()
plt.savefig('avg_utility_per_task.jpg')


# --- Plot 2: Response Time ---
normal_resp_vals = [normal_avg_response[k] * 1000 for k in sorted_keys]
tlm_resp_vals = [tlm_avg_response[k] * 1000 for k in sorted_keys]
normal_resp_errs = [normal_std_response[k] * 1000 for k in sorted_keys]
tlm_resp_errs = [tlm_std_response[k] * 1000 for k in sorted_keys]

fig, ax = plt.subplots(figsize=(14, 4), dpi=200)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

ax.bar([x - bar_width/2 for x in range(len(sorted_keys))],
       normal_resp_vals, width=bar_width, color='none',
       yerr=normal_resp_errs, capsize=5, ecolor='gray', label='vLLM',
       align='center', edgecolor=colors[3], linewidth=4, hatch='//')

ax.bar([x + bar_width/2 for x in range(len(sorted_keys))],
       tlm_resp_vals, width=bar_width, color='none',
       yerr=tlm_resp_errs, capsize=5, ecolor='gray', label='TimeLyLLM',
       align='center', edgecolor=colors[2], linewidth=4, hatch='-')

ax.set_xticks(range(len(sorted_keys)))
ax.set_xticklabels(sorted_keys, fontsize=20)
ax.set_xlim(-0.5, len(sorted_keys) - 0.5)
ax.tick_params(axis='y', labelsize=20)
resp_max = max(v + e for v, e in zip(normal_resp_vals + tlm_resp_vals, normal_resp_errs + tlm_resp_errs))
ax.set_ylim(0, resp_max * 1.15)
ax.set_xlabel('Task Type ID', fontsize=20)
ax.set_ylabel('Time (ms)', fontsize=20)
ax.legend(fontsize=20)

plt.tight_layout()
plt.savefig('avg_response_per_task.jpg')


# --- Statistics ---
normal_keys = [k for k in sorted_keys if k <= 4]   # trace 0-4
urgent_keys = [k for k in sorted_keys if k >= 5]   # trace 5-7

def print_stats(metric_name, vllm_dict, tlm_dict, scale=1, unit='', higher_is_better=False):
    for label, keys in [("All tasks", sorted_keys), ("Normal tasks (0-4)", normal_keys), ("Urgent tasks (5-7)", urgent_keys)]:
        vllm_avg = np.mean([vllm_dict[k] * scale for k in keys])
        tlm_avg = np.mean([tlm_dict[k] * scale for k in keys])
        if higher_is_better:
            diff = (tlm_avg - vllm_avg) / abs(vllm_avg) * 100 if vllm_avg != 0 else float('inf')
            diff_label = f"improvement={diff:+.1f}%"
        else:
            diff = (vllm_avg - tlm_avg) / vllm_avg * 100 if vllm_avg != 0 else float('inf')
            diff_label = f"reduction={diff:.1f}%"
        print(f"  {label:22s}: vLLM={vllm_avg:.1f}{unit}, TimeLyLLM={tlm_avg:.1f}{unit}, {diff_label}")

print("=" * 60)
print("Time Utility:")
print_stats("Time Utility", normal_avg_utility, tlm_avg_utility, higher_is_better=True)
print()
print("Response Time:")
print_stats("Response Time", normal_avg_response, tlm_avg_response, scale=1000, unit='ms')
print("=" * 60)
