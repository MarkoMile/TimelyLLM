import matplotlib.pyplot as plt
# from executor.virtual_wrapper import ActionDelay
import json
import numpy as np
import math
# from IPython.display import Image
import statistics


data_sample_file = './dataset/data_sample_1.json'  # high workload
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
    utility_all_normal = []
    utility_all_urgent = []
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

            if task_type_id not in task_waiting_times:
                task_waiting_times[task_type_id] = []
            task_waiting_times[task_type_id].append(response_time_sum)

            if task_type_id not in task_response_times:
                task_response_times[task_type_id] = []
            task_response_times[task_type_id].append(response_time)

            utility = TUFReader.tuf(response_time, task_id)
            if task_type_id in [0,1,2,3,4]:
                response_times_type1.append(response_time)
                utility_all_normal.append(utility)
            elif task_type_id in [5,6,7]:
                response_times_type2.append(response_time)
                utility_all_urgent.append(utility)
            # utility_all.append(utility)
            if task_type_id not in task_utility:
                task_utility[task_type_id] = []
            task_utility[task_type_id].append(utility)

    average_waiting_times = {task_type_id: sum(times) / len(times) for task_type_id, times in task_waiting_times.items()}
    average_response_times = {task_type_id: sum(times) / len(times) for task_type_id, times in task_response_times.items()}
    average_utility = {task_type_id: sum(times) / len(times) for task_type_id, times in task_utility.items()}


    # average_utility_all = statistics.mean(utility_all)
    avergae_utility_all_normal = statistics.mean(utility_all_normal)
    std_utility_all_normal = statistics.stdev(utility_all_normal)
    avergae_utility_all_urgent = statistics.mean(utility_all_urgent)
    std_utility_all_urgent = statistics.stdev(utility_all_urgent)

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

    return avergae_utility_all_normal, std_utility_all_normal, avergae_utility_all_urgent, std_utility_all_urgent, average_utility



# ab for batch
data_files = {
    'FCFS': ['./timelyllm/logs/exp743_fcfs_data.json'],
    'EDF': ['./timelyllm/logs/exp743_edf_data.json'],
    'TickingLLM': ['./timelyllm/logs/exp743_timelyllm_data.json']
}

utility_all_values = {}
utility_task_values = {}
utility_error_values = {}

for method, file_paths in data_files.items():
    avergae_utility_all_normal, std_utility_all_normal, avergae_utility_all_urgent, std_utility_all_urgent, utility_task = read_data(file_paths[0])
    sorted_keys = sorted(utility_task.keys())
    sorted_values = [utility_task[key] for key in sorted_keys]
    utility_task_values[method] = sorted_values
    
    # Average of first five values as normal task utility
    normal_task_utility = sum(sorted_values[:5]) / 5
    # Average of last three values as urgent task utility
    urgent_task_utility = sum(sorted_values[5:]) / 3
    
    # Store both values
    utility_task_values[method] = {
        'normal_task': avergae_utility_all_normal,
        'urgent_task': avergae_utility_all_urgent
    }

    utility_error_values[method] = {
        'normal_task': std_utility_all_normal,
        'urgent_task': std_utility_all_urgent
    }




# Plot
methods = list(utility_task_values.keys())
normal_utilities = [utility_task_values[method]['normal_task'] for method in methods]
urgent_utilities = [utility_task_values[method]['urgent_task'] for method in methods]
normal_errors = [utility_error_values[method]['normal_task'] for method in methods]
urgent_errors = [utility_error_values[method]['urgent_task'] for method in methods]

fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
task_ids = np.array(list(sorted_keys))
colors = plt.get_cmap('tab20').colors
bar_width = 0.2
index = np.arange(len(methods))
plt.bar(index, normal_utilities, width=bar_width, label='Normal Tasks', align='center', color=colors[3], hatch='//', edgecolor='white')
plt.bar(index+bar_width/2, urgent_utilities, width=bar_width, label='Urgent Tasks', align='edge', color=colors[2], hatch='\\', edgecolor='white')
plt.xticks(index + bar_width / 2, methods, size =25)
plt.yticks(size=30)
# plt.xlabel('Batch Method', size=25)
plt.ylabel('Utility', size=30)
# plt.title('Utility Comparison of Different Methods')
# plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, zorder=1)
plt.legend(fontsize = 25)
plt.show()
plt.tight_layout()
plt.savefig('ab_study_schedule.jpg', bbox_inches='tight')


# plt.figure(figsize=(18, 6), dpi=200)
# task_ids = np.array(list(sorted_keys))

# bar_width = 0.2
# index = np.arange(len(task_ids))
# # colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(utility_task_values)))

# colors = ['#2f2f2f', '#9b9b9b', '#f9b29b', '#b6342f'] 

# for i, (method, task_values) in enumerate(utility_task_values.items()):
#     # plt.bar(index + i * bar_width, task_values, bar_width, label=method, color=colors[i])
#     bars = plt.bar(index + i * bar_width, list(task_values), bar_width, label=method, color=colors[i])
#     for bar in bars:
#         plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.2f}', 
#                  ha='center', va='bottom', fontsize=10)
# # Adding labels and title
# plt.xlabel('Task ID', size=22)
# plt.ylabel('Utility', size=22)
# # plt.title('Utility per Task for Different Methods')
# plt.xticks(index + bar_width * 1.5, task_ids)
# plt.legend(loc='lower left', fontsize=20)
# plt.savefig('ab_study.jpg')


# plt.figure(figsize=(10, 4), dpi=200)

# fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
# ax.spines['top'].set_linewidth(1.5)
# ax.spines['right'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['bottom'].set_linewidth(1.5)
# task_ids = np.array(list(sorted_keys))

# bar_width = 0.15  # Slightly narrower bars
# index = np.arange(len(task_ids))

# # colors = ['#2f2f2f', '#9b9b9b', '#f9b29b', '#b6342f'] 
# colors = plt.get_cmap('tab20c').colors

# for i, (method, task_values) in enumerate(utility_task_values.items()):
#     if i == 0 or i == 1:
#         bars = plt.bar(index + i * bar_width, list(task_values), bar_width, label=method, color=colors[16+i], zorder=2)
#     else:
#         bars = plt.bar(index + i * bar_width, list(task_values), bar_width, label=method, color=colors[4-i], zorder=2)
#     # for bar in bars:
#     #     plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.2f}', 
#     #              ha='center', va='bottom', fontsize=5, zorder=3)

# # Adding labels and title
# plt.xlabel(r'Task ID', size=15)
# plt.ylabel(r'Utility', size=15)

# plt.xticks(index + bar_width * 1.5, task_ids, size=15)
# plt.yticks(size=15)

# # Adding grid and setting it behind bars
# plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, zorder=1)
# # plt.legend(loc='upper left', fontsize=20)
# # Move legend to the top of the plot
# plt.legend(bbox_to_anchor=(1, 1.2), ncol=5, fontsize=12)
# # Save the figure with a higher resolution
# plt.tight_layout()
# plt.savefig('ab_study.jpg', bbox_inches='tight')