import heapq
import time
import random
import re
import json
import os
from util.log_config import logger
from queue import PriorityQueue
from dataclasses import dataclass, field
from util.util import Task
import numpy as np
from pathlib import Path
    
class RequestGenerator:
    def __init__(self, task_queue, agent_num, request_list_path, stop_signal, run_mode, seg_exe, comm_time, robot_type, robot_system, real_audio_task_ids=None, whisper_device="cpu", engine_ready=None):
        self.engine_ready = engine_ready
        self.task_queue = task_queue
        self.stop_signal = stop_signal
        self.agent_num = agent_num
        self.tasks = []
        self.run_mode =run_mode
        self.seg_exe = seg_exe
        self.comm = comm_time
        self.robot_type = robot_type
        self.robot_system = robot_system
        self.epsilon = 1e-25
        self.real_audio_task_ids = real_audio_task_ids if real_audio_task_ids is not None else []
        self.whisper_device = whisper_device
        self.audio_file_path = Path(__file__).parent.parent / "dataset-chatbot" / "audio_human.wav"
        self.read_task_list(request_list_path)

    def transcribe_audio(
        self,
        audio_path: Path,
        model_name: str = "base",
        language: str | None = None,
        temperature: float = 0.0,
        device: str = "cpu",  
    ) -> str:
        """Transcribe audio file using Whisper model."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # If using CPU, disable CUDA before importing torch/whisper to avoid CUDA init in forked process
        if device == "cpu":
            # Save original environment variable
            original_cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            # Set to empty so this process cannot see any GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            logger.info(f"Disabled CUDA for Whisper transcription (using CPU only)")
        
        try:
            # Import whisper after setting env var to ensure whisper cannot see GPU
            import whisper
            
            logger.info(f"Loading Whisper model: {model_name} on device: {device}")
            model = whisper.load_model(model_name, device=device)
            logger.info(f"Transcribing audio file: {audio_path}")
            result = model.transcribe(
                str(audio_path),
                language=language,
                temperature=temperature,
            )
            transcribed_text = result.get("text", "").strip()
            logger.info(f"Transcription completed: {transcribed_text}...")
            return transcribed_text
        finally:
            # Restore original environment variable if needed
            if device == "cpu":
                if original_cuda_visible_devices is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible_devices
                elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                    del os.environ['CUDA_VISIBLE_DEVICES']

    def read_task_list(self, filename):
        """Read task time and description from a JSON file and store them in a list."""
        with open(filename, 'r') as file:
            tasks = json.load(file)

        for task in tasks:
            details = self.parse_task_details(task)
            heapq.heappush(self.tasks, (details['task_arrival_time'], details))

    def parse_task_details(self, task):
        """Parse details from a task dictionary."""        
        alpha = task['tuf'][0]/(task['tuf'][1]-task['tuf'][2])
        exe_end_pre = task['tuf'][1]
        
        task_input = task['task_input']
        needs_audio_transcription = task['job_id'] in self.real_audio_task_ids
        
        # output_len is only needed in chatbot mode, non-chatbot mode uses default 200
        output_len = task.get('output_len', 200)
        
        details = {
            'task_arrival_time': task['trigger time'],
            'task_id': task['job_id'],
            'agent_id': task['agent_id'],
            "robot_type": task.get('robot_type', self.robot_type),
            "robot_system": task.get('robot_system', self.robot_system),
            'task_input': task_input,
            'needs_audio_transcription': needs_audio_transcription,  # Flag for audio transcription
            'cur_plan': str(""),
            'task_tuf': task['tuf'], # util_max, ect, uct
            'task_alpha': alpha,
            'exe_end_pre': exe_end_pre,
            'output_length': output_len,
            'remaining_length': output_len
        }
        return details

    def _init_priority_calculate(self, tuf_set, task_alpha, exe_end_pre):
        beta = tuf_set[0]
        delta_w = time.time()+self.seg_exe+self.comm-exe_end_pre
        priority = beta * np.exp(delta_w) + self.epsilon * (-task_alpha)
        return -round(priority,30)

    def generate_requests(self):
        """Generate requests from predefined tasks and add them to the task queue until all tasks are processed or stop signal is received."""
        # disable CUDA for whisper in this process to avoid issues with forked processes and GPU memory
        if self.whisper_device == "cpu":
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

        # The trace is replayed against wall-clock: task N is sent at init_time +
        # its trigger time. Without this wait, init_time is set while the engine
        # is still loading the model, so the early part of the trace queues up
        # behind startup rather than being served. That is invisible when loading
        # takes ~25s, but under an MPS thread-percentage cap loading is several
        # times slower, and the measured latency becomes mostly load time.
        if self.engine_ready is not None:
            print("Waiting for engine before starting the trace clock...")
            wait_start = time.time()
            while not self.engine_ready.is_set():
                if self.stop_signal.is_set():
                    print("Stopped while waiting for the engine.")
                    return
                self.engine_ready.wait(timeout=1.0)
            print(f"Engine ready after {time.time() - wait_start:.1f}s; starting trace.")

        init_time = time.time()
        total_tasks = len(self.tasks)
        print(f"Init time for sending request: {init_time}")
        print(f"Total tasks to generate: {total_tasks}")
        tasks_sent = 0
        while not self.stop_signal.is_set():
            if self.tasks: 
                task_arrival_time, task_details = heapq.heappop(self.tasks)
                # update relative arrival time to absolute arrival time
                task_details['task_arrival_time'] += init_time
                task_details['exe_end_pre'] += task_details['task_arrival_time']
                time_pass = time.time()-init_time
                # print(f'time_pass: {time_pass}')
                if time_pass > 0:
                    time_diff = task_arrival_time-time_pass
                    if time_diff > 0:
                        time.sleep(task_arrival_time-time_pass)  # Simulate delay based on the task time

                # Perform audio transcription in subprocess (if needed)
                if task_details.get('needs_audio_transcription', False):
                    logger.info(f"Task {task_details['task_id']} is a real audio task, transcribing audio...")
                    try:
                        transcribed_text = self.transcribe_audio(self.audio_file_path, device=self.whisper_device)
                        task_details['task_input'] = transcribed_text
                        logger.info(f"Transcription completed for task {task_details['task_id']}: {transcribed_text[:50]}...")
                    except Exception as e:
                        logger.error(f"Failed to transcribe audio for task {task_details['task_id']}: {e}")
                        logger.info("Using original task_input")

                # task_description = f"{task_details['task_input']}"
                # task = f"Task {task_details['task_id']} at time {task_details['task_ddl']}: {task_description}"
                if self.run_mode == 'rtllm' or self.run_mode == 'rtllm-fixed':
                    init_priority = self._init_priority_calculate(task_details['task_tuf'], task_details['task_alpha'], task_details['exe_end_pre'])
                elif self.run_mode == 'vllm' or self.run_mode == 'vllm-fixed' or self.run_mode == 'rtllm-fcfs-fixed' or self.run_mode == 'vllm-stream':
                    init_priority = task_details['task_arrival_time']
                elif self.run_mode == 'rtllm-chatbot' or self.run_mode == 'vllm-chatbot-stream':
                    init_priority = task_details['task_arrival_time']
                elif self.run_mode == 'vllm-edf' or self.run_mode == 'rtllm-edf' or 'chatbot' in self.run_mode or self.run_mode == 'rtllm-edf-fixed':
                    init_priority = task_details['task_tuf'][1] + task_details['task_arrival_time']
                # self.task_queue.put((task_details['task_ddl']+ddl, task_details))
                print(f"init_priority for task {task_details['task_id']}: {init_priority}")
                # print(f"task_details: {task_details}")
                # self.update_priority()
                
                # Use non-blocking put to avoid getting stuck when queue is full
                while not self.stop_signal.is_set():
                    try:
                        self.task_queue.put(Task(init_priority, task_details), block=True, timeout=1.0)
                        logger.info(f"Generating request from task {task_details['task_id']}, time: {time.time()}")
                        tasks_sent += 1
                        if tasks_sent % 100 == 0:
                            print(f"Progress: {tasks_sent}/{total_tasks} tasks sent to queue")
                        break
                    except:
                        # Queue is full, wait and retry
                        print(f"Queue full, waiting to add task {task_details['task_id']}...")
                        time.sleep(0.1)
            else:
                # All tasks have been sent, exit the loop
                print(f"All tasks sent ({tasks_sent}/{total_tasks}). Request generator finishing.")
                break

        print(f"Request generator shutting down. Total tasks sent: {tasks_sent}/{total_tasks}")
