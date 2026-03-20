import multiprocessing
from queue import Empty
import threading
import time
import random
import heapq
import torch
import sys
import argparse

# Parse --log-name early, before any module that imports the logger
def _peek_log_name():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--log-name', type=str, default=None)
    args, _ = parser.parse_known_args()
    return args.log_name

_log_name = _peek_log_name()
if _log_name:
    import util.log_config as _lc
    _lc.LOG_NAME = _log_name

from request.read_request import RequestGenerator
from rtengine.vllm_llm_scheduler import infer_start

from queue import PriorityQueue
from multiprocessing import Process
from multiprocessing.managers import BaseManager

from executor.virtual_wrapper import task_executor_pool_async, task_executor
from config import build_config

# Optional: Enable memory monitoring
try:
    from util.memory_monitor import MemoryMonitor
    MEMORY_MONITOR_AVAILABLE = True
except ImportError:
    MEMORY_MONITOR_AVAILABLE = False
    print("Memory monitor not available (psutil not installed)")


class Manager(BaseManager):
    pass


Manager.register('get_priorityQueue', PriorityQueue)
            

if __name__ == "__main__":
    # construct queue for storing requests

    process_manager = Manager()
    process_manager.start()
    task_queue = process_manager.get_priorityQueue()
    # Load configuration from preset / CLI args
    cfg = build_config()

    # Clear old log file with the same name to prevent appending stale data
    if _log_name:
        import os
        old_log = f'./logs/{_log_name}.log'
        if os.path.exists(old_log):
            os.remove(old_log)
            print(f"Removed old log file: {old_log}")

    # construct queues for storing LLM generated results
    result_queues = {i: multiprocessing.Queue() for i in range(cfg.agent_num)}
    for i, queue in result_queues.items():
        queue.id = i

    # Signal to stop the processes
    global_stop_signal = multiprocessing.Event()

    # Start memory monitoring if enabled
    memory_monitor = None
    if cfg.enable_memory_monitor and MEMORY_MONITOR_AVAILABLE:
        memory_monitor = MemoryMonitor(log_interval=10.0)
        memory_monitor.start_monitoring()
        print(f"Memory monitoring enabled for {cfg.agent_num} agents with pooled executor")

    # Start the processes
    request_gen = RequestGenerator(
        task_queue, cfg.agent_num, cfg.request_list_path, global_stop_signal,
        cfg.run_mode, cfg.seg_exe, cfg.comm_time, cfg.robot_type, cfg.robot_system,
        cfg.real_audio_task_ids, cfg.whisper_device
    )
    p1 = multiprocessing.Process(target=request_gen.generate_requests, args=())

    p2 = multiprocessing.Process(
        target=infer_start,
        args=(task_queue, result_queues, cfg.prompt_path, cfg.agent_num,
              global_stop_signal, cfg.run_mode, cfg.robot_system, cfg.batchsize,
              cfg.seg_exe, cfg.lmax, cfg.comm_time, cfg.real_audio_task_ids,
              cfg.prompt_speech_path, cfg.model_path)
    )

    if cfg.executor_mode == 'virtual':
        if cfg.use_async_executor:
            from executor.virtual_wrapper import task_executor_pool_async
            print(f"Using ASYNC executor: {cfg.max_workers} workers (non-blocking, per-agent serial)")
            if cfg.real_audio_task_ids is not None:
                print(f"Real audio generation enabled for task_id(s): {cfg.real_audio_task_ids}")
            p3 = multiprocessing.Process(
                target=task_executor_pool_async,
                args=(result_queues, cfg.agent_num, cfg.request_list_path,
                      global_stop_signal, cfg.run_mode, cfg.comm_time,
                      cfg.max_workers, False, cfg.real_audio_task_ids, cfg.tts_engine)
            )
        else:
            from executor.virtual_wrapper import task_executor
            p3 = multiprocessing.Process(
                target=task_executor,
                args=(result_queues, cfg.agent_num, cfg.request_list_path,
                      global_stop_signal, cfg.run_mode, cfg.comm_time,
                      False, cfg.real_audio_task_ids, cfg.tts_engine)
            )
    elif cfg.executor_mode == 'real':
        from executor.realworld_wrapper import task_executor
        robot_info_path = "robot_info.json"
        p3 = multiprocessing.Process(
            target=task_executor,
            args=(result_queues, cfg.agent_num, global_stop_signal,
                  robot_info_path, cfg.comm_time)
        )
    else:
        raise ValueError(f"Unknown executor mode: {cfg.executor_mode}")

    p1.start()
    p2.start()
    p3.start()

    # Phase 1: Wait for request generator to finish sending all tasks.
    # Phase 2: Wait for scheduler (p2) and executor (p3) to drain all queues.
    # Use run_duration as an overall timeout.
    deadline = time.time() + cfg.run_duration
    while p1.is_alive() and time.time() < deadline:
        time.sleep(1)

    if p1.is_alive():
        print(f"Timeout ({cfg.run_duration}s) reached. Stopping.")
    else:
        print("All requests sent. Waiting for scheduler & executor to finish...")
        # Poll until task_queue and all result_queues stay empty,
        # meaning all work has been processed.
        idle_checks = 0
        while time.time() < deadline:
            queues_empty = task_queue.empty() and all(q.empty() for q in result_queues.values())
            if queues_empty:
                idle_checks += 1
                if idle_checks >= 3:
                    print("All queues drained. Experiment complete.")
                    break
            else:
                idle_checks = 0
            time.sleep(10)

    # Sending termination signals
    global_stop_signal.set()

    # Wait for all processes to finish
    p1.join(timeout=30)
    p2.join(timeout=30)
    p3.join(timeout=30)

    # Stop memory monitoring
    if memory_monitor:
        memory_monitor.stop_monitoring()
