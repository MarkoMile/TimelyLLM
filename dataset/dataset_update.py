import json


with open("./dataset/data_sample_12.json", "r") as f:
    data_sample = json.load(f)

with open("./dataset/trace_set_3_real.json", "r") as f:
    trace_set_real = json.load(f)


trace_map = {trace["trace_id"]: trace["exe_time_detail"] for trace in trace_set_real}


for entry in data_sample:
    trace_id = entry["trace_id"]
    if trace_id in trace_map:
        entry["exe_time_detail"] = trace_map[trace_id]
        entry["task_exe_time"] = sum(time[1] for time in trace_map[trace_id])  


updated_file_path = "./dataset/data_sample_2.json"
with open(updated_file_path, "w") as f:
    json.dump(data_sample, f, indent=4)

print(f"Updated data saved to {updated_file_path}")
