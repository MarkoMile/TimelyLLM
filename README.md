# TimelyLLM: Time-sensitive LLM Serving System for Physical-I/O Limited Agents

## Code Structure
```
timelyllm/
├── rtllm.py                      # Main entry point for launching the system
├── config.py                     # Experiment configuration and presets
├── resdata.py                    # Summarize results from log files
│
├── rtengine/                     # Real-time LLM scheduling engine
│   ├── vllm_llm_scheduler.py     # Main scheduler based on vLLM (multiple scheduling policies)
│   ├── vllm_llm_engine_usage.py  # Custom vLLM engine with stop checker
│   ├── stop_rule.py              # MiniSpec-based stop rule for segmented generation
│   ├── minispec_interpreter.py   # MiniSpec language interpreter
│   ├── exe_worst_case.py         # Worst-case execution time estimation
│   ├── skillset.py               # Robot skill definitions
│   ├── skill_item.py             # Skill item data structure
│   └── prompt/                   # Prompt templates for different robot systems
│       ├── prompt_drone_typefly.txt
│       ├── prompt_robot_arm.txt
│       ├── prompt_baseline_fltrnn.txt
│       ├── prompt_chatbot.txt
│       └── ...
│
├── request/                      # Request generation
│   └── read_request.py           # Read task datasets and generate requests
│
├── executor/                     # Task execution
│   ├── virtual_wrapper.py        # Virtual executor (replay from traces)
│   ├── realworld_wrapper.py      # Real-world robot executor
│   ├── speech_wrapper.py         # TTS wrapper (piper/espeak-ng/flite)
│   ├── speech_output.py          # Speech output with Piper TTS
│   └── minispec_executor/        # MiniSpec plan executor
│
├── util/                         # Utilities
│   ├── util.py                   # Task data structure
│   ├── log_config.py             # Thread-safe lazy logger
│   └── memory_monitor.py         # GPU/CPU memory monitoring
│
└── logs/                         # Experiment log output

dataset/                          # Dataset
├── trace_set_*.json              # Trace files
└── data_sample_*.json            # Workload files

fig_plot/                         # Result visualization
├── performance_compare_vllm.py   # TimeLyLLM vs vLLM comparison plots
├── ablation_study_sched.py       # Scheduling algorithm ablation study
└── robot_arm_res_fltrnn.py       # Robot arm (FLTRNN) result plots
```

## Quick Start

### Prerequisites
- Python 3.10
- NVIDIA GPU with CUDA 12.1 compatible driver (tested on a single RTX 4090)
- [uv](https://docs.astral.sh/uv/) package manager
- Download LLM model (e.g., Meta-Llama-3-8B-Instruct)


### Option A: Jupyter Notebook (Recommended)

One-click setup and execution — environment setup, experiments, and result processing are all included.

> **Choose either Local or Docker below.**

**Local:**
```bash
uv sync --extra notebook
source .venv/bin/activate
```
**Open `run_experiments.ipynb`, select the `.venv` kernel, and run all cells.**

**Docker:**
```bash
sudo docker build -t timelyllm .
sudo docker run --gpus all -p 8888:8888 -v $(pwd)/model:/app/model \
    --entrypoint uv timelyllm \
    run jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```
**Open the URL printed in the terminal, then open `run_experiments.ipynb` and run all cells.**

### Option B: CLI

<details>
<summary>Manual setup and run via command line</summary>

**Running Experiments (Local):**
```bash
# Setup
uv sync
source .venv/bin/activate

cd timelyllm

# --- Exp 7.4.1: TimeLyLLM vs vLLM (High workload, Drone) ---
python3 rtllm.py --preset exp741_timelyllm_high --log-name exp741_timelyllm_high
python3 rtllm.py --preset exp741_vllm_high      --log-name exp741_vllm_high
# Process results
python3 -c "from resdata import read_log_file; read_log_file('./logs/exp741_timelyllm_high.log')"
python3 -c "from resdata import read_log_file; read_log_file('./logs/exp741_vllm_high.log')"
# Plot
python3 ../fig_plot/performance_compare_vllm.py

# --- Exp 7.4.3: Scheduling Algorithms ---
python3 rtllm.py --preset exp743_timelyllm --log-name exp743_timelyllm
python3 rtllm.py --preset exp743_edf       --log-name exp743_edf
python3 rtllm.py --preset exp743_fcfs      --log-name exp743_fcfs
# Process results
python3 -c "from resdata import read_log_file; read_log_file('./logs/exp743_timelyllm.log')"
python3 -c "from resdata import read_log_file; read_log_file('./logs/exp743_edf.log')"
python3 -c "from resdata import read_log_file; read_log_file('./logs/exp743_fcfs.log')"
# Plot
python3 ../fig_plot/ablation_study_sched.py

# --- Exp 7.4.4: Different Robot Systems (FLTRNN, Robot Arm) ---
python3 rtllm.py --preset exp744_fltrnn_timelyllm  --log-name exp744_fltrnn_timelyllm
python3 rtllm.py --preset exp744_fltrnn_vllm       --log-name exp744_fltrnn_vllm
# Process results
python3 -c "from resdata import read_log_file; read_log_file('./logs/exp744_fltrnn_timelyllm.log')"
python3 -c "from resdata import read_log_file; read_log_file('./logs/exp744_fltrnn_vllm.log')"
# Plot
python3 ../fig_plot/robot_arm_res_fltrnn.py
```

**Running Experiments (Docker):**
```bash
# Build
sudo docker build -t timelyllm .

# Each experiment runs as a separate container; logs are saved to /app/timelyllm/logs/ inside the container.
# Use -v to mount model and persist logs to host.

# --- Exp 7.4.1: TimeLyLLM vs vLLM (High workload, Drone) ---
docker run --gpus all -v $(pwd)/model:/app/model -v $(pwd)/logs:/app/timelyllm/logs timelyllm \
    --preset exp741_timelyllm_high --log-name exp741_timelyllm_high
docker run --gpus all -v $(pwd)/model:/app/model -v $(pwd)/logs:/app/timelyllm/logs timelyllm \
    --preset exp741_vllm_high --log-name exp741_vllm_high
# Process results
docker run --gpus all -v $(pwd)/logs:/app/timelyllm/logs -e PYTHONPATH=/app/timelyllm --entrypoint uv timelyllm \
    run python3 -c "from resdata import read_log_file; read_log_file('./timelyllm/logs/exp741_timelyllm_high.log')"
docker run --gpus all -v $(pwd)/logs:/app/timelyllm/logs -e PYTHONPATH=/app/timelyllm --entrypoint uv timelyllm \
    run python3 -c "from resdata import read_log_file; read_log_file('./timelyllm/logs/exp741_vllm_high.log')"
# Plot
docker run --gpus all -v $(pwd)/logs:/app/timelyllm/logs -e PYTHONPATH=/app/timelyllm --entrypoint uv timelyllm \
    run python3 ./fig_plot/performance_compare_vllm.py

# --- Exp 7.4.3: Scheduling Algorithms ---
docker run --gpus all -v $(pwd)/model:/app/model -v $(pwd)/logs:/app/timelyllm/logs timelyllm \
    --preset exp743_timelyllm --log-name exp743_timelyllm
docker run --gpus all -v $(pwd)/model:/app/model -v $(pwd)/logs:/app/timelyllm/logs timelyllm \
    --preset exp743_edf --log-name exp743_edf
docker run --gpus all -v $(pwd)/model:/app/model -v $(pwd)/logs:/app/timelyllm/logs timelyllm \
    --preset exp743_fcfs --log-name exp743_fcfs
# Process results
docker run --gpus all -v $(pwd)/logs:/app/timelyllm/logs -e PYTHONPATH=/app/timelyllm --entrypoint uv timelyllm \
    run python3 -c "from resdata import read_log_file; read_log_file('./timelyllm/logs/exp743_timelyllm.log')"
docker run --gpus all -v $(pwd)/logs:/app/timelyllm/logs -e PYTHONPATH=/app/timelyllm --entrypoint uv timelyllm \
    run python3 -c "from resdata import read_log_file; read_log_file('./timelyllm/logs/exp743_edf.log')"
docker run --gpus all -v $(pwd)/logs:/app/timelyllm/logs -e PYTHONPATH=/app/timelyllm --entrypoint uv timelyllm \
    run python3 -c "from resdata import read_log_file; read_log_file('./timelyllm/logs/exp743_fcfs.log')"
# Plot
docker run --gpus all -v $(pwd)/logs:/app/timelyllm/logs -e PYTHONPATH=/app/timelyllm --entrypoint uv timelyllm \
    run python3 ./fig_plot/ablation_study_sched.py

# --- Exp 7.4.4: Different Robot Systems (FLTRNN, Robot Arm) ---
docker run --gpus all -v $(pwd)/model:/app/model -v $(pwd)/logs:/app/timelyllm/logs timelyllm \
    --preset exp744_fltrnn_timelyllm --log-name exp744_fltrnn_timelyllm
docker run --gpus all -v $(pwd)/model:/app/model -v $(pwd)/logs:/app/timelyllm/logs timelyllm \
    --preset exp744_fltrnn_vllm --log-name exp744_fltrnn_vllm
# Process results
docker run --gpus all -v $(pwd)/logs:/app/timelyllm/logs -e PYTHONPATH=/app/timelyllm --entrypoint uv timelyllm \
    run python3 -c "from resdata import read_log_file; read_log_file('./timelyllm/logs/exp744_fltrnn_timelyllm.log')"
docker run --gpus all -v $(pwd)/logs:/app/timelyllm/logs -e PYTHONPATH=/app/timelyllm --entrypoint uv timelyllm \
    run python3 -c "from resdata import read_log_file; read_log_file('./timelyllm/logs/exp744_fltrnn_vllm.log')"
# Plot
docker run --gpus all -v $(pwd)/logs:/app/timelyllm/logs -e PYTHONPATH=/app/timelyllm --entrypoint uv timelyllm \
    run python3 ./fig_plot/robot_arm_res_fltrnn.py
```

**CLI Overrides:**

Any preset parameter can be overridden from the command line:

```bash
# Local
python3 rtllm.py --preset exp741_timelyllm_high --agent-num 20 --log-name my_experiment

# Docker
docker run --gpus all -v $(pwd)/model:/app/model -v $(pwd)/logs:/app/timelyllm/logs timelyllm \
    --preset exp741_timelyllm_high --agent-num 20 --log-name my_experiment
```

Key CLI options:
- `--log-name`: Custom log filename (without `.log` extension)
- `--agent-num`: Number of agents
- `--run-mode`: Scheduling mode (`rtllm`, `vllm`, etc.)
- `--model-path`: Path to LLM model

</details>