import argparse
import sys
from dataclasses import dataclass, field
from typing import List, Optional


ROBOT_DEFAULTS = {
    'typefly': (0.09, 10),
    'fltrnn': (0.16, 20),
    'chatbot': (0.16, 20),
}


@dataclass
class ExperimentConfig:
    # Core experiment settings
    agent_num: int = 42
    executor_mode: str = 'virtual'
    run_mode: str = 'rtllm'
    robot_system: str = 'typefly'
    robot_type: str = 'drone'

    # Executor settings
    enable_memory_monitor: bool = False
    use_async_executor: bool = False
    max_workers: Optional[int] = None  # None = min(agent_num, 8)

    # Audio/speech settings
    real_audio_task_ids: Optional[List[int]] = None
    whisper_device: str = 'cpu'
    tts_engine: str = 'piper'

    # Scheduling settings
    batchsize: int = 6
    seg_exe: Optional[float] = None   # None = derived from robot_system
    lmax: Optional[int] = None        # None = derived from robot_system
    comm_time: float = 0.008

    # Engine sizing. Upstream hardcoded these for a 24GB RTX 4090. On a larger
    # card the default KV cache is big enough that the memory-side admission
    # constraint may never bind, leaving that path untested -- lower
    # gpu_memory_utilization to reproduce the original pressure regime.
    # See PORT_PLAN.md D4.
    gpu_memory_utilization: float = 0.8
    max_model_len: int = 4000
    max_num_seqs: int = 8

    # Paths
    model_path: str = '../model/Meta-Llama-3-8B-Instruct'
    request_list_path: str = '../dataset/data_sample_1.json'
    prompt_path: str = '/rtengine/prompt/prompt_drone_typefly.txt'
    prompt_speech_path: Optional[str] = '/rtengine/prompt/prompt_doctor.txt'

    # Runtime
    run_duration: int = 10000  # seconds

    def __post_init__(self):
        # Derive seg_exe and lmax from robot_system if not explicitly set
        if self.robot_system in ROBOT_DEFAULTS:
            default_seg, default_lmax = ROBOT_DEFAULTS[self.robot_system]
            if self.seg_exe is None:
                self.seg_exe = default_seg
            if self.lmax is None:
                self.lmax = default_lmax
        else:
            if self.seg_exe is None or self.lmax is None:
                raise ValueError(
                    f"Unknown robot_system '{self.robot_system}'. "
                    f"Must set seg_exe and lmax explicitly or use one of: {list(ROBOT_DEFAULTS.keys())}"
                )
        if self.max_workers is None:
            self.max_workers = min(self.agent_num, 8)


# === Experiment Presets ===

PRESETS = {
    # --- 7.4.1: TimeLyLLM vs vLLM (High workload, Drone) ---
    'exp741_timelyllm_high': {
        'run_mode': 'rtllm',
        'robot_system': 'typefly', 'robot_type': 'drone',
        'agent_num': 42,
        'request_list_path': '../dataset/data_sample_1.json',
        'prompt_path': '/rtengine/prompt/prompt_drone_typefly.txt',
    },
    'exp741_vllm_high': {
        'run_mode': 'vllm',
        'robot_system': 'typefly', 'robot_type': 'drone',
        'agent_num': 42,
        'request_list_path': '../dataset/data_sample_1.json',
        'prompt_path': '/rtengine/prompt/prompt_drone_typefly.txt',
    },

    # --- 7.4.3: Scheduling Algorithms (High workload) ---
    'exp743_timelyllm': {
        'run_mode': 'rtllm-fixed',
        'robot_system': 'typefly', 'robot_type': 'drone',
        'agent_num': 42,
        'batchsize': 6,
        'request_list_path': '../dataset/data_sample_1.json',
        'prompt_path': '/rtengine/prompt/prompt_drone_typefly.txt',
    },
    'exp743_edf': {
        'run_mode': 'rtllm-edf-fixed',
        'robot_system': 'typefly', 'robot_type': 'drone',
        'agent_num': 42,
        'batchsize': 6,
        'request_list_path': '../dataset/data_sample_1.json',
        'prompt_path': '/rtengine/prompt/prompt_drone_typefly.txt',
    },
    'exp743_fcfs': {
        'run_mode': 'rtllm-fcfs-fixed',
        'robot_system': 'typefly', 'robot_type': 'drone',
        'agent_num': 42,
        'batchsize': 6,
        'request_list_path': '../dataset/data_sample_1.json',
        'prompt_path': '/rtengine/prompt/prompt_drone_typefly.txt',
    },

    # --- 7.4.4: Different Robot Systems (FLTRNN, Robot Arm) ---
    'exp744_fltrnn_timelyllm': {
        'run_mode': 'rtllm',
        'robot_system': 'fltrnn', 'robot_type': 'robotarm',
        'agent_num': 94,
        'request_list_path': '../dataset/data_sample_robotarm_fltrnn.json',
        'prompt_path': '/rtengine/prompt/prompt_baseline_fltrnn.txt',
    },
    'exp744_fltrnn_vllm': {
        'run_mode': 'vllm',
        'robot_system': 'fltrnn', 'robot_type': 'robotarm',
        'agent_num': 94,
        'request_list_path': '../dataset/data_sample_robotarm_fltrnn.json',
        'prompt_path': '/rtengine/prompt/prompt_baseline_fltrnn.txt',
    },
}


def build_config() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description='TimeLyLLM experiment runner')
    parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()),
                        help='Load a named experiment preset')
    parser.add_argument('--list-presets', action='store_true',
                        help='List all available presets and exit')

    # Per-field overrides
    parser.add_argument('--agent-num', type=int)
    parser.add_argument('--executor-mode', type=str, choices=['virtual', 'real'])
    parser.add_argument('--run-mode', type=str)
    parser.add_argument('--robot-system', type=str)
    parser.add_argument('--robot-type', type=str)
    parser.add_argument('--no-async', action='store_true', help='Disable async executor')
    parser.add_argument('--enable-memory-monitor', action='store_true')
    parser.add_argument('--real-audio-task-ids', type=int, nargs='+')
    parser.add_argument('--whisper-device', type=str, choices=['cpu', 'cuda'])
    parser.add_argument('--tts-engine', type=str, choices=['auto', 'piper', 'espeak-ng', 'flite'])
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--seg-exe', type=float)
    parser.add_argument('--lmax', type=int)
    parser.add_argument('--comm-time', type=float)
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--request-list-path', type=str)
    parser.add_argument('--prompt-path', type=str)
    parser.add_argument('--prompt-speech-path', type=str)
    parser.add_argument('--run-duration', type=int)
    parser.add_argument('--max-workers', type=int)
    parser.add_argument('--gpu-memory-utilization', type=float)
    parser.add_argument('--max-model-len', type=int)
    parser.add_argument('--max-num-seqs', type=int)
    parser.add_argument('--log-name', type=str, help='Custom log filename (without .log extension)')

    args = parser.parse_args()

    if args.list_presets:
        print("Available presets:\n")
        for name, overrides in PRESETS.items():
            print(f"  {name}:")
            for k, v in overrides.items():
                print(f"    {k}: {v}")
            print()
        sys.exit(0)

    # Start with defaults
    kwargs = {}

    # Layer 1: Apply preset
    if args.preset:
        kwargs.update(PRESETS[args.preset])

    # Layer 2: Apply CLI overrides (only non-None values)
    cli_mapping = {
        'agent_num': args.agent_num,
        'executor_mode': args.executor_mode,
        'run_mode': args.run_mode,
        'robot_system': args.robot_system,
        'robot_type': args.robot_type,
        'real_audio_task_ids': args.real_audio_task_ids,
        'whisper_device': args.whisper_device,
        'tts_engine': args.tts_engine,
        'batchsize': args.batchsize,
        'seg_exe': args.seg_exe,
        'lmax': args.lmax,
        'comm_time': args.comm_time,
        'model_path': args.model_path,
        'request_list_path': args.request_list_path,
        'prompt_path': args.prompt_path,
        'prompt_speech_path': args.prompt_speech_path,
        'run_duration': args.run_duration,
        'max_workers': args.max_workers,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'max_model_len': args.max_model_len,
        'max_num_seqs': args.max_num_seqs,
    }
    for key, val in cli_mapping.items():
        if val is not None:
            kwargs[key] = val

    if args.no_async:
        kwargs['use_async_executor'] = False
    if args.enable_memory_monitor:
        kwargs['enable_memory_monitor'] = True

    cfg = ExperimentConfig(**kwargs)
    print(f"Config: {cfg}")
    return cfg
