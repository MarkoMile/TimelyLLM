import time
import random
import threading
import queue
import re
import json
from util.log_config import logger
from executor.speech_wrapper import SpeechGenerator


class AgentTaskCache:
    def __init__(self, request_list_path):
        self.taskdata = {}
        with open(request_list_path, 'r') as file:
            self.task_database = json.load(file)
        
    def read_exe_time(self, task_id):
        if task_id in self.taskdata:
            self.taskdata[task_id]['cur_step'] += 1
        else:
            self._register(task_id)
        step_index = self.taskdata[task_id]['cur_step']
        exe_time = self.taskdata[task_id]['task_item'][step_index][1]
        if step_index == self.taskdata[task_id]['step_sum']-1:
            self._abort(task_id)
        return exe_time

    def _abort(self, task_id):
        return self.taskdata.pop(task_id, None) 

    def _register(self, task_id):
        task_details = self._get_task_by_job_id(task_id)
        self.taskdata[task_id] = {}
        self.taskdata[task_id]['task_item'] = task_details['exe_time_detail']
        self.taskdata[task_id]['step_sum'] = len(task_details['exe_time_detail'])
        self.taskdata[task_id]['cur_step'] = 0
    
    def _get_task_by_job_id(self, job_id):
        for task_item in self.task_database:
            if task_item['job_id'] == job_id:
                return task_item
        return None


class ActionDelay:
    def __init__(self, request_list_path):
        self.agentcache = AgentTaskCache(request_list_path)      
        self.last_audio_end_time = {}
    # -------------------------------- robot ------------------------------------------ #
    def delay_map(self, command, task_id):
        delay = 0
        keywords = [
            r'\bmf\b',
            r'\bmb\b',
            r'\bml\b',
            r'\bmr\b',
            r'\bmu\b',
            r'\bmd\b',
            r'\btc\b',
            r'\btu\b',
            r'\box\b',
            r'\boy\b',
            r'\bow\b',
            r'\boh\b',
            r'\bod\b',
            r'\biv\b',
            r'(?<![tr])\bp\b',
            r'(?<!m)\bl\b',
            r'(?<!m)\bd\b', 
            r'\btp\b',
            r'\brp\b',
            r'\bg\(',
            r'\bs\(',
            r'\bsa\(',
            # robot arm
            r'\bgt\b',  # goto
            r'\bpi\b', # pick
            r'\bdr\b', # drop
            r'\bsc\b',  # scan
            # robot arm fltrnn
            r'\bmove_to\b',  # move_to
            r'\btake\b', # take
            r'\bdrop\b', # drop
            r'\bsd\b', # sound
        ]

        sorted_matches = sorted(
            (match.start(), match.group(0)) 
            for key in keywords
            for match in re.finditer(key, command)
        )

        for _, matched_text in sorted_matches:
            delay += self.agentcache.read_exe_time(task_id)

        return round(delay, 6)


    def execute_task(self, agent_id, result_queue, stop_signal, comm_time):
        """Thread target function to execute tasks from result queue."""
        while not stop_signal.is_set():
            try:
                task_id, result, timein, final_flag = result_queue.get(timeout=1)
                cur_time = time.time()
                if cur_time-timein < comm_time:
                    time.sleep(cur_time-timein)
                    # print("delay due to comm")
                start_time = time.time()
                logger.info(f"Start executing task {task_id} for agent {agent_id} on time {start_time} with plan {result}")
                # time.sleep(random.randint(10, 30))  # Simulate processing delay, replace this part with collected time trace 
                delaytime = self.delay_map(result, task_id)
                time.sleep(delaytime)
                end_time = time.time()
                logger.info(f"Finish executing task {task_id} for agent {agent_id} on time {end_time} with plan {result}")
            except queue.Empty:
                continue
        print("Task execution thread shutting down.")

    # -------------------------------- chatbot ------------------------------------------ #
    def chat_delay_map(self, command, task_id):
        # x= command.split()
        # print(f"command split:{command.split()}")
        word_count = len(command.split())
        reading_speed = 5 # 5 words per second
        # print(f"task id: {task_id} word_count: {word_count}")
        reading_time = word_count / reading_speed
        return round(reading_time, 6)
    
    def execute_chat(self, agent_id, result_queue, stop_signal):
        """Thread target function to execute tasks from result queue."""
        while not stop_signal.is_set():
            try:
                task_id, result = result_queue.get(timeout=1)
                start_time = time.time()
                logger.info(f"Start executing task {task_id} for agent {agent_id} on time {start_time} with plan {task_id}")
                # time.sleep(random.randint(10, 30))  # Simulate processing delay, replace this part with collected time trace 
                delaytime = self.chat_delay_map(result, task_id)
                # print(f"task_id:{task_id}, delaytime: {delaytime}")
                time.sleep(delaytime)
                end_time = time.time()
                logger.info(f"Finish executing task {task_id} for agent {agent_id} on time {end_time} with plan {task_id}")
            except queue.Empty:
                continue
        print("Task execution thread shutting down.")

    def execute_speech(self, agent_id, result_queue, stop_signal, speech_generator):
        """Thread target function to execute tasks from result queue."""
        print(f"[Agent {agent_id}] Speech execution thread started")
        while not stop_signal.is_set():
            try:
                # Get task data: task_id, result, timein (LLM completion time), final_flag, task_arrival_time
                task_data = result_queue.get(timeout=0.01)
                if len(task_data) == 5:
                    task_id, result, timein, final_flag, task_arrival_time = task_data
                else:
                    # Backward compatibility: if task_arrival_time is not provided, use timein as fallback
                    task_id, result, timein, final_flag = task_data
                    task_arrival_time = timein
                
                current_time = time.time()

                agent_filename = f"./speech_output/speech_output_{agent_id}.wav"

                # Calculate response time (from request arrival to LLM completion)
                response_time = timein - task_arrival_time
                last_audio_end = virtualdelay.last_audio_end_time.get(agent_id, 0)
                silence_to_insert = 0
                if last_audio_end == 0 and response_time > 0.01:
                    # First task: insert response time silence at the beginning
                    silence_to_insert = response_time
                elif timein > last_audio_end:
                    # Gap between last audio and request arrival + response time
                    silence_to_insert = timein - last_audio_end

                start_time = time.time()
                logger.info(f"Start executing task {task_id} for agent {agent_id} on time {start_time} with plan {task_id}")
                print(f"[Agent {agent_id}] Processing task {task_id}, text length: {len(result)} chars")
                
                delaytime = speech_generator.speech_generate(result, filename=agent_filename, silence_duration=silence_to_insert, task_id=task_id)
                time.sleep(delaytime)
                end_time = time.time()
                logger.info(f"Finish executing task {task_id} for agent {agent_id} on time {end_time} with plan {task_id}")

                self.last_audio_end_time[agent_id] = end_time

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Agent {agent_id}] Error processing task: {e}")
                import traceback
                traceback.print_exc()
        print(f"[Agent {agent_id}] Task execution thread shutting down.")


def task_executor(result_queues, agent_num, request_list_path, stop_signal, run_mode, comm_time = 0, enable_speech_output=True, real_audio_task_ids=None, tts_engine="auto"):
    """Process 3: Manage threads that execute tasks from the result queue."""
    def execute_task_wrapper(agent_id, result_queue, stop_signal, request_list_path, run_mode, enable_speech_output, real_audio_task_ids, tts_engine):
        virtualdelay = ActionDelay(request_list_path)
        # dry_run=True, no wav file
        # tts_engine: "auto", "piper", "espeak-ng", "flite"
        speech_generator = SpeechGenerator(dry_run=not enable_speech_output, real_audio_task_ids=real_audio_task_ids, engine=tts_engine)
        if 'chatbot' in run_mode:
            # virtualdelay.execute_chat(agent_id, result_queue, stop_signal)
            virtualdelay.execute_speech(agent_id, result_queue, stop_signal, speech_generator)
        else:
            virtualdelay.execute_task(agent_id, result_queue, stop_signal, comm_time)
    
    agents = [threading.Thread(target=execute_task_wrapper, args=(i, result_queues[i],stop_signal,request_list_path, run_mode, enable_speech_output, real_audio_task_ids, tts_engine)) for i in range(agent_num)]
    for agent in agents:
        agent.start()
    
    for agent in agents:
        agent.join()  # Wait for all threads to finish

    print("Task executor process shutting down.")



def task_executor_pool_async(result_queues, agent_num, request_list_path, stop_signal, run_mode, comm_time=0, max_workers=8, enable_speech_output=True, real_audio_task_ids=None, tts_engine="auto"):
    """
    Async non-blocking version: Workers don't block during delays.
    Uses a scheduler to track task completion times, allowing workers to process other tasks during delays.
    """
    import heapq
    import queue
    
    # Create shared components
    virtualdelay = ActionDelay(request_list_path)
    # dry_run=True means no real WAV files are generated, only simulating delays
    # real_audio_task_ids specifies which task_ids should generate real audio
    # tts_engine: "auto", "piper", "espeak-ng", "flite"
    speech_generator = SpeechGenerator(dry_run=not enable_speech_output, real_audio_task_ids=real_audio_task_ids, engine=tts_engine)
    
    # Priority queue for scheduled task completions: (completion_time, agent_id, task_id, result, start_time)
    completion_queue = []
    completion_lock = threading.Lock()
    
    # Per-agent execution control: ensures tasks from same agent execute serially
    agent_executing = {i: False for i in range(agent_num)}
    agent_lock = threading.Lock()
    
    # Track last segment finish time for each agent
    agent_last_finish_time = {i: 0 for i in range(agent_num)}
    
    def worker(worker_id, stop_signal):
        """Non-blocking worker that schedules task completion instead of sleeping."""
        print(f"Async Worker {worker_id} started")
        while not stop_signal.is_set():
            processed = False
            
            # Try to get new tasks from agent queues
            for agent_id in range(agent_num):
                # Atomic check-and-set: check if agent is free AND get task
                task_data = None
                with agent_lock:
                    if agent_executing[agent_id]:
                        # Skip this agent, it's busy
                        continue
                    
                    # Try to get task (non-blocking)
                    try:
                        task_data = result_queues[agent_id].get(timeout=0.01)
                        # Successfully got task, mark agent as executing
                        agent_executing[agent_id] = True
                    except queue.Empty:
                        continue  # No task, try next agent
                
                # Process the task (outside lock)
                if task_data:
                    # Get task data: task_id, result, timein (LLM completion time), final_flag, task_arrival_time
                    if len(task_data) == 5:
                        task_id, result, timein, final_flag, task_arrival_time = task_data
                    else:
                        # Backward compatibility: if task_arrival_time is not provided, use timein as fallback
                        task_id, result, timein, final_flag = task_data
                        task_arrival_time = timein
                    processed = True
                    
                    try:
                        # Use max of timein and last_audio_end as start_time
                        # This ensures: if LLM finishes before last audio ends, wait for it;
                        # if LLM finishes after last audio ends, start immediately at LLM completion time
                        last_audio_end = virtualdelay.last_audio_end_time.get(agent_id, 0)
                        start_time = max(timein, last_audio_end)
                        logger.info(f"Start executing task {task_id} for agent {agent_id} on time {start_time} with plan {task_id}")
                        
                        # Calculate delay time but DON'T sleep
                        if 'chatbot' in run_mode:
                            agent_filename = f"./speech_output/speech_output_{agent_id}.wav"
                            
                            # Calculate response time (from request arrival to LLM completion)
                            response_time = timein - task_arrival_time
                            silence_to_insert = 0
                            if last_audio_end == 0 and response_time > 0.01:
                                # First task: insert response time silence at the beginning
                                silence_to_insert = response_time
                            elif timein > last_audio_end:
                                # Gap between last audio and request arrival + response time
                                silence_to_insert = timein - last_audio_end
                            
                            delaytime = speech_generator.speech_generate(result, filename=agent_filename, silence_duration=silence_to_insert, task_id=task_id)
                        else:
                            delaytime = virtualdelay.delay_map(result, task_id)
                        
                        # Schedule completion instead of blocking
                        completion_time = start_time + delaytime
                        with completion_lock:
                            heapq.heappush(completion_queue, 
                                          (completion_time, agent_id, task_id, result, start_time, delaytime))
                        
                        print(f"Async Worker {worker_id}: Task {task_id} scheduled, will complete at {completion_time:.2f} (delay={delaytime:.2f}s)")
                    except Exception as e:
                        # If any error occurs during processing, release the agent
                        logger.error(f"Error processing task {task_id} for agent {agent_id}: {e}")
                        with agent_lock:
                            agent_executing[agent_id] = False
                    
                    # Worker is now FREE to process other tasks!
                    break
            
            if not processed:
                # No new tasks, brief sleep
                time.sleep(0.01)
        
        print(f"Async Worker {worker_id} shutting down")
    
    def completion_handler(stop_signal):
        """Separate thread to handle task completions."""
        print("Completion handler started")
        while not stop_signal.is_set():
            current_time = time.time()
            
            with completion_lock:
                # Check if any tasks have completed
                while completion_queue and completion_queue[0][0] <= current_time:
                    completion_time, agent_id, task_id, result, start_time, delaytime = heapq.heappop(completion_queue)
                    
                    # Update last audio end time for chatbot
                    if 'chatbot' in run_mode:
                        virtualdelay.last_audio_end_time[agent_id] = completion_time
                    
                    # Update last segment finish time for this agent
                    agent_last_finish_time[agent_id] = completion_time
                    
                    logger.info(f"Finish executing task {task_id} for agent {agent_id} on time {completion_time} with plan {task_id}")
                    
                    # Release this agent - allow it to execute next task
                    with agent_lock:
                        agent_executing[agent_id] = False
                        print(f"Agent {agent_id} released, can accept new tasks")
            
            time.sleep(0.01)  # Check every 10ms
        
        print("Completion handler shutting down")
    
    # Start worker threads
    workers = []
    for i in range(max_workers):
        worker_thread = threading.Thread(target=worker, args=(i, stop_signal))
        worker_thread.start()
        workers.append(worker_thread)
    
    # Start completion handler
    completion_thread = threading.Thread(target=completion_handler, args=(stop_signal,))
    completion_thread.start()
    
    # Wait for all workers to finish
    for worker_thread in workers:
        worker_thread.join()
    
    completion_thread.join()
    
    print("Task executor (async) process shutting down.")



if __name__ == "__main__":
    request_list_path = './dataset/data_sample2.json'
    virtualdelay = ActionDelay(request_list_path)
    delay = virtualdelay.delay_map("?_1=sa('Any edible target here?');?_1!=False{g(_1);->True}->?_2=sa('Any drinkable target here?');?_2!=False{g(_2)};",11)
    print(delay)