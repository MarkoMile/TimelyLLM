from threading import Thread
from functools import reduce
import torch
import time
import os
import asyncio
from queue import Empty
# from vllm_lib_usage import RTAsyncLLMEngine
from vllm.engine.arg_utils import EngineArgs
from rtengine.backend import make_backend
# torch.multiprocessing.set_start_method('spawn')
from vllm.sampling_params import SamplingParams
from vllm import SamplingParams
# from executor.virtual_wrapper import ActionDelay
from rtengine.exe_worst_case import ExeWorstEst
# model_path = "/home/neiwen/hf_modelzoo/models--meta-llama--Meta-Llama-3-8B-instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa"
# model_path = "/home/neiwen/hfmodelzoo/models--Qwen--Qwen2-7B-Instruct/snapshots/f2826a00ceef68f0f2b946d945ecc0477ce4450c"
# model_path = "/home/neiwen/hfmodelzoo/models--microsoft--Phi-3-mini-128k-instruct/snapshots/a90b62ae09941edff87a90ced39ba5807e6b2ade"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from util.log_config import logger
from util.util import Task
from queue import PriorityQueue
from rtengine.stop_rule import MiniSpecProgram
import numpy as np


class TaskDetails:
    def __init__(self, comm_time, lmax, max_token_history=2000):
        self.taskdata = {}
        self.comm_time = comm_time
        self.lmax = lmax
        self.max_token_history = max_token_history  # Limit token history to prevent memory explosion

    def _update_add(self, task_id, key, value):
        if task_id in self.taskdata:
            self.taskdata[task_id][key] += value
        else:
            raise KeyError("Task ID not found.")
        
    def _update_replace(self, task_id, key, value):
        if task_id in self.taskdata:
            self.taskdata[task_id][key] = value
        else:
            raise KeyError("Task ID not found.")
        
    def _update_input(self, task_id, output):
        if task_id in self.taskdata:
            task_input_cur = self.taskdata[task_id]['task_input']
            if isinstance(task_input_cur, str):
                prompt_token_id = output.prompt_token_ids
                res_token_id = output.outputs[0].token_ids
                self.taskdata[task_id]['task_input'] = prompt_token_id + list(res_token_id)
            else:
                res_token_id = output.outputs[0].token_ids
                self.taskdata[task_id]['task_input'] += list(res_token_id)
            
        else:
            raise KeyError("Task ID not found.")
        
    
    def _read_para(self, task_id, key):
        if task_id in self.taskdata:
            return self.taskdata[task_id][key]
        else:
            # print(task_id)
            raise KeyError("Task ID not found.")
        
    def _read_task(self, task_id):
        if task_id in self.taskdata:
            return self.taskdata[task_id]
        else:
            raise KeyError("Task ID not found.")

    def _abort(self, task_id):
        """Remove task and explicitly clean up large objects to free memory."""
        task = self.taskdata.pop(task_id, None)
        if task:
            # Explicitly delete large objects (token IDs) to help garbage collector
            if 'task_input' in task and not isinstance(task['task_input'], str):
                del task['task_input']
            task.clear()  # Clear all references
        return task 

    def _register(self, task_details):
        task_id = task_details['task_id']
        if task_id not in self.taskdata:
            self.taskdata[task_id] = {}
        self.taskdata[task_id]['task_arrival_time'] = task_details['task_arrival_time']
        self.taskdata[task_id]['task_id'] = task_details['task_id']
        self.taskdata[task_id]['agent_id'] = task_details['agent_id']
        self.taskdata[task_id]['task_input'] = task_details['task_input']
        self.taskdata[task_id]['cur_plan'] = task_details['cur_plan']
        self.taskdata[task_id]['task_tuf'] = task_details['task_tuf']
        self.taskdata[task_id]['reg_time'] = time.time()
        self.taskdata[task_id]['task_alpha'] = task_details['task_alpha']
        self.taskdata[task_id]['exe_end_pre'] = task_details['exe_end_pre']
        self.taskdata[task_id]['robot_type'] = task_details['robot_type']
        self.taskdata[task_id]['robot_system'] = task_details['robot_system']
        self.taskdata[task_id]['finished_token_num'] = 0
        self.taskdata[task_id]['output_length'] = task_details['output_length']
        self.taskdata[task_id]['remaining_length'] = task_details['remaining_length']

    def get_earliest_registered_task(self):
        if not self.taskdata:
            return None  # Return None if no tasks are registered
        earliest_task = min(self.taskdata.values(), key=lambda x: x['reg_time'])
        return earliest_task['task_id']
    
    def get_urgent_registered_task(self, token_speed=0):
        if not self.taskdata:
            return None  # Return None if no tasks are registered

        if token_speed == 0:
            # search the urgentest task based on exe_end_pre
            urgent_task = min(self.taskdata.values(), key=lambda x: x['exe_end_pre'])
        else:
            # search the urgentest task based on remaining_workload - remaing_slack_time
            urgent_task = max(self.taskdata.values(), key=lambda x: self._time_diff_calcualte(x['finished_token_num'], x['exe_end_pre'], token_speed))
        return urgent_task['task_id']

    def _time_diff_calcualte(self, finished_token_num, exe_end_pre, token_speed):
        slack_time = -(time.time()+self.comm_time-exe_end_pre)
        remaining_workload = (self.lmax-finished_token_num)/token_speed
        time_diff = remaining_workload - slack_time
        # print(f"finished_token_num:{finished_token_num}, token_speed:{token_speed}, time_diff:{time_diff}")
        return time_diff
    
class RequestScheduler:
    def __init__(self, prompt_path : str, global_stop_signal, agent_num: int, seg_exe: float, lmax : int,
                              run_mode : str = "rtllm", use_cache : bool =True, comm_time: float = 0, robot_system: str = "typefly",
                              real_audio_task_ids=None, prompt_speech_path=None, model_path=None):
        # gpu parameter
        gpu_memory_utilization=0.8
        max_model_len=4000 #4000
        self.max_num_seqs=8

        # Use passed model_path, or fall back to module-level default
        _model_path = model_path if model_path is not None else globals().get('model_path')
        engine_args = EngineArgs(
            model=_model_path,
            tensor_parallel_size=1,
            dtype="half",
            # tensor_parallel_size=2,
            # dtype = "float32",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=self.max_num_seqs, 
            enable_prefix_caching=True
            # disable_sliding_window=True
        )
        self.backend = make_backend(
            engine_args,
            robot_system=robot_system,
            segment_stop=("rtllm" in run_mode),
        )
        self.llm_engine = self.backend.engine
        if "rtllm" in run_mode:
            print("initilize stop checker")
        
        self.sampling_params = SamplingParams(temperature=0,skip_special_tokens=True,
            max_tokens=200)

        self.global_stop_signal= global_stop_signal
        self.run_mode = run_mode
        self.use_cache = use_cache
        self.virtualdelay = ExeWorstEst()
        self.taskData = TaskDetails(comm_time, lmax)
        self.prompt_path = prompt_path
        self.seg_exe = seg_exe
        self.lmax = lmax
        self.program_interpreter = MiniSpecProgram()
        self.comm_time = comm_time
        self.epsilon = 1e-25
        
        # Support for speech-specific prompts
        # real_audio_task_ids: specifies which task_ids need real audio file generation (others only simulate timing)
        # Can be a single task_id (int), a list, or None (don't use speech prompt)
        if real_audio_task_ids is not None:
            if isinstance(real_audio_task_ids, int):
                self.real_audio_task_ids = {real_audio_task_ids}
            elif isinstance(real_audio_task_ids, (list, set)):
                self.real_audio_task_ids = set(real_audio_task_ids)
            else:
                self.real_audio_task_ids = set()
        else:
            self.real_audio_task_ids = None
        
        # prompt_speech_path: prompt path for audio output tasks
        self.prompt_speech_path = prompt_speech_path
        
    def _prompt_read(self, prompt_path=None):
        # print("read prompts")
        # prompt_list = ['0','1','2']
        # prompt_list = [str(element) for element in prompt_list]
        # prompt = prompt_list[self.agent_id]
        current_directory = os.getcwd()
        path_to_use = prompt_path if prompt_path is not None else self.prompt_path
        with open(current_directory + path_to_use, 'r', encoding='utf-8') as file:
            prompt = file.read()
        return prompt
    
    def _input_gen(self, task_id):
        # Decide which prompt to use based on whether task_id is in real_audio_task_ids
        if self.real_audio_task_ids is not None and task_id in self.real_audio_task_ids:
            prompt = self._prompt_read(self.prompt_speech_path or None)
        else:
            # Use default prompt
            prompt = self._prompt_read()
        
        task_input = self.taskData._read_para(task_id,'task_input')
        prompt_sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + prompt + "<|eot_id|>\n\n"
        # description = "Here is the 'scene description': [chair_3 x:0.34 y:0.3 width:0.09 height:0.1], Here is the 'task description': [A] Go and take a picture of the chair. Here is the 'execution history'(has value if replanning): None."
        command = "Please only generate the response with a single sentence of MiniSpec program. 'Response':"
        if 'chatbot' in self.run_mode:
            prompt_usr = "<|start_header_id|>user<|end_header_id|>\n\n" + task_input + "<|eot_id|>\n\n"
        else:
            prompt_usr = "<|start_header_id|>user<|end_header_id|>\n\n" + task_input + command + "<|eot_id|>\n\n"
        prompt_ass= "<|start_header_id|>assistant<|end_header_id|>\n\n" 

        cur_plan = self.taskData._read_para(task_id,'cur_plan')
        input_prompt = prompt_sys + prompt_usr + prompt_ass + cur_plan
        return input_prompt
    
    
    def _utility_calcualte(self, tuf, alpha, exe_end_pre):
        delta_w =  time.time() - exe_end_pre
        utility = alpha * np.maximum(0, delta_w) + tuf[0]
        return utility

    
    def _priority_calculate(self, tuf, alpha, exe_end_pre):
        # print(f"act_time:{act_time}")
        # tuf_value = self.tuf(act_time, tuf_set)
        # priority = tuf_value/(tuf_set[1]-act_time)
        # priority = self._tuf(act_time+self.seg_exe, tuf_set)/self.seg_exe
        # priority = priority/(tuf_set[1]-act_time)
        delta_w = time.time()+self.seg_exe+self.comm_time-exe_end_pre
        # print(f'delta:{delta_w}')
        priority = (alpha * np.maximum(0, delta_w) + tuf[0]) * np.exp(np.minimum(0, delta_w)) + self.epsilon * (-alpha)
        # print(f'priority:{priority}')
        return -round(priority,30)

    def _update_queue_priority(self):
        temp_queue = PriorityQueue()
        cur_time = time.time()
        while not self.task_queue.empty():
            task_item = self.task_queue.get()
            task = task_item.task_detail
            new_priority = self._priority_calculate(task['task_tuf'], task['task_alpha'], task['exe_end_pre'])
            # print(f"task: {task['task_id']}", f"new_priority: {new_priority}")
            temp_queue.put(Task(new_priority, task))
        
        while not temp_queue.empty():
            task_item = temp_queue.get()
            priority = task_item.priority
            task = task_item.task_detail
            # print(f"priority: {priority}")
            # self.task_queue.put(Task(priority,task))
            utility = self._utility_calcualte(task['task_tuf'], task['task_alpha'], task['exe_end_pre'])
            if utility > 0:
                self.task_queue.put(Task(priority,task))
            else:
                self.taskData._abort(task['task_id'])
                # print(f"task {task['task_id']} has negative utility, not added to queue")

    def _update_task_time_details(self, task_id, exe_time):
        gen_end = time.time() + self.comm_time
        exe_end_pre_old = self.taskData._read_para(task_id, 'exe_end_pre')
        task_cur_plan = self.taskData._read_para(task_id, 'cur_plan')
        if task_cur_plan == "" :
            # for the first segment
            exe_end_pre = gen_end + exe_time
        else:
            exe_end_pre = max(gen_end,exe_end_pre_old) + exe_time
        self.taskData._update_replace(task_id, 'exe_end_pre', exe_end_pre)
        self.taskData._update_replace(task_id, 'finished_token_num', 0)
        return 
    

    def _admit_memory_ok(self):
        '''
        Memory-side admission gate.

        Upstream compared free KV blocks against a threshold of 0, or 1376 for
        "sequential" mode. On the RTX 4090 it was tuned for, 1376 is close to the
        total block count, so that test meant "the cache is empty", i.e. run one
        request at a time. Both branches are restated here in terms that do not
        depend on GPU size, so they carry over to the GH200 unchanged.
        See PORT_PLAN.md D1.
        '''
        if self.run_mode == "sequential":
            return self.backend.num_running() == 0
        return self.backend.kv_has_free()

    def get_task(self):
        '''
        Get task from task queue
        For methods schedule_timely, schedule_timely_fixed_batch
        '''
        time1 = time.time()
        self._update_queue_priority()
        try:
            task_item = self.task_queue.get_nowait()
        except Empty:
            return None
        if task_item is None:
            return None
        task_details = task_item.task_detail

        agent_id = task_details['agent_id']
        task_id = task_details['task_id']
        self.taskData._register(task_details)
        task_input = task_details['task_input']
        if isinstance(task_input, str):
            prompt_input = self._input_gen(task_id)
        else:
            prompt_input = {"prompt_token_ids": task_input}
        print(f"Starting to add task {task_id}")
        time2 = time.time()
        # print(f"Get task overhead:{time2-time1} ")
        return task_id, agent_id, prompt_input


    def get_task_edf(self):
        '''
        Get task from task queue
        For methods schedule_edf, schedule_other_fixed_batch, schedule_chatbot
        '''
        time1 = time.time()
        try:
            task_item = self.task_queue.get_nowait()
        except Empty:
            return None
        if task_item is None:
            return None
        task_details = task_item.task_detail

        agent_id = task_details['agent_id']
        task_id = task_details['task_id']
        self.taskData._register(task_details)
        task_input = task_details['task_input']
        if isinstance(task_input, str):
            prompt_input = self._input_gen(task_id)
        else:
            prompt_input = {"prompt_token_ids": task_input}
        print(f"Starting to add task {task_id}")
        time2 = time.time()
        # print(f"Get task overhead:{time2-time1} ")
        return task_id, agent_id, prompt_input

    def get_task_normal(self):
        '''
        Get task from task queue
        For methods based on vLLM nature
        '''
        try:
            task_item = self.task_queue.get_nowait()
        except Empty:
            return None
        if task_item is None:
            return None
        task_details = task_item.task_detail

        agent_id = task_details['agent_id']
        task_id = task_details['task_id']
        self.taskData._register(task_details)
        prompt_input = self._input_gen(task_id)
        print(f"Starting to add task {task_id}")
        return task_id, agent_id, prompt_input
    
    def _process_seg_output(self, output, task_id, result_gen):
        robot_type = self.taskData._read_para(task_id, 'robot_type')
        exe_time =  self.virtualdelay.delay_wc_map(result_gen,robot_type)
        self._update_task_time_details(task_id, exe_time)
        self.taskData._update_add(task_id, 'cur_plan', result_gen)
        # update task input, store token id in task input
        self.taskData._update_input(task_id, output)
        print("read request to task queue")
        task_details = self.taskData._read_task(task_id)
        # init_priority = round(task_details['task_tuf'][0]/task_details['task_tuf'][1],6)
        # print(f"init_priority: {init_priority}")
        init_priority = self._priority_calculate(task_details['task_tuf'], task_details['task_alpha'], task_details['exe_end_pre'])
        self.task_queue.put(Task(init_priority, task_details))
    
    def _process_chat_seg_output(self, output, task_id, result_gen):
        exe_time =  self.virtualdelay.chat_delay_map(result_gen)
        self._update_task_time_details(task_id, exe_time)
        self.taskData._update_add(task_id, 'cur_plan', result_gen)
        # update task input, store token id in task input
        self.taskData._update_input(task_id, output)
        
        remaining_length_old = self.taskData._read_para(task_id, 'remaining_length')
        cur_output_length = len(output.outputs[0].token_ids)
        remaining_length = remaining_length_old - cur_output_length - 1
        if remaining_length <= 0:
            remaining_length = 1
        # 1 for stop token
        self.taskData._update_replace(task_id, 'remaining_length', remaining_length)

        # print("readd request to task queue")
        task_details = self.taskData._read_task(task_id)
        init_priority = task_details['exe_end_pre']
        self.task_queue.put(Task(init_priority, task_details))

    def _process_chat_seg_output_timely(self, output, task_id, result_gen):
        exe_time =  self.virtualdelay.chat_delay_map(result_gen)
        self._update_task_time_details(task_id, exe_time)
        self.taskData._update_add(task_id, 'cur_plan', result_gen)
        # update task input, store token id in task input
        self.taskData._update_input(task_id, output)
        
        remaining_length_old = self.taskData._read_para(task_id, 'remaining_length')
        cur_output_length = len(output.outputs[0].token_ids)
        remaining_length = remaining_length_old - cur_output_length - 1
        if remaining_length <= 0:
            remaining_length = 1
        # 1 for stop token
        self.taskData._update_replace(task_id, 'remaining_length', remaining_length)

        # print("readd request to task queue")
        task_details = self.taskData._read_task(task_id)
        # init_priority = round(task_details['task_tuf'][0]/task_details['task_tuf'][1],6)
        # print(f"init_priority: {init_priority}")
        init_priority = self._priority_calculate(task_details['task_tuf'], task_details['task_alpha'], task_details['exe_end_pre'])
        self.task_queue.put(Task(init_priority, task_details))
    # -------------------------------- Methods based on timelyLLM------------------------------------------ #
    def schedule_timely(self, task_queue, result_queues, robot_system='typefly'):
            lmax = self.lmax # max token length of one segment, 10 for TypeFly, 20 for FLTRNN
            self.task_queue = task_queue
            id_urgent_gen_len = 0
            running_num = 0
            # block_size_in_bytes=self.llm_engine.scheduler[0].block_manager.block_size
            batch_cur_size = 0
            wait_flag = False
            token_speed = 0 # initial token speed
            while not self.global_stop_signal.is_set():
                # wait until get new requests
                # num_free_gpu = sum(
                #     scheduler.block_manager.get_num_free_gpu_blocks()
                #     for scheduler in self.llm_engine.scheduler)
                # print(f"when no task, the free gpu is {num_free_gpu}")
                while True:
                    task_info = self.get_task()
                    if task_info:
                        task_id, agent_id, prompt_input = task_info
                        id_urgent = self.taskData.get_urgent_registered_task(token_speed)
                        running_num += 1
                        self.llm_engine.add_request(task_id, prompt_input, self.sampling_params)
                        logger.info(f"Added task {task_id}, time: {time.time()}")
                        batch_cur_size += 1
                        # print(f"batch size: {batch_cur_size}")
                        break

                while self.llm_engine.has_unfinished_requests():
                    # process exsiting request
                    step_outputs = self.llm_engine.step()

                    # decide add more request or not
                    
                    # running_num = len(self.llm_engine.scheduler[0].running)
                    # print(f"current batch size: {batch_cur_size} running request number: {running_num} ")

                    # waiting_num = self.llm_engine.get_num_unfinished_requests()
                    # print(f"current batch size: {batch_cur_size} unifinished request number: {waiting_num} ")
                    if not wait_flag:
                        # memory constraint
                        if self._admit_memory_ok():
                        # if token_budget > lmax:
                            # print("pass memory constraint")
                            # latency constraint
                            exe_end_pre = self.taskData._read_para(id_urgent, 'exe_end_pre')
                            slack_time = -(time.time()+self.comm_time-exe_end_pre)
                            reg_time = self.taskData._read_para(id_urgent, 'reg_time')
                            # id_urgent = self.taskData.get_earliest_registered_task

                            id_urgent_gen_len = self.taskData._read_para(id_urgent, 'finished_token_num')
                            token_speed = (id_urgent_gen_len)/(time.time()-reg_time)
                            # print(f"id_urgent_gen_len: {id_urgent_gen_len}")
                            # print(f"token_speed: {token_speed}")
                            # print(f"slack_time: {slack_time}")
                            # print(f"token_speed: {token_speed}")
                            if token_speed == 0:
                                add_flag = True # avoid adding request when it quals to zero
                                # print(f"add task due to tokenspeed 0")
                            elif (lmax-id_urgent_gen_len)/token_speed < slack_time:
                                add_flag = True
                            elif slack_time < 0:
                                pass  # print(f"not add task due to slack time {slack_time}")
                            else:
                                pass  # print(f"not add token_speed: {token_speed}")
                                # a = (lmax-id_urgent_gen_len)/token_speed
                                # b = slack_time
                                # print(f"remain_exe_time:{a}")
                                # print(f"ddl:{b}")
                                # print(f"add task due to pass time test")

                            if add_flag:
                                add_flag = False
                                # print("pass time constraint")
                                task_info = self.get_task()
                                id_urgent = self.taskData.get_urgent_registered_task(token_speed)
                                if task_info:
                                    task_id, agent_id, prompt_input = task_info
                                    running_num += 1
                                    self.llm_engine.add_request(task_id, prompt_input, self.sampling_params)
                                    logger.info(f"Added task {task_id}, time: {time.time()}")
                                    batch_cur_size += 1
                                    # print(f"batch size: {batch_cur_size}")
                                # if added task is urgent than the current urgent task, update urgent task
                                # new_task_urgency = self.taskData._read_para(task_id,'exe_end_pre')
                                # cur_urgency = self.taskData._read_para(id_urgent,'exe_end_pre')
                                # if new_task_urgency < cur_urgency:
                                #     id_urgent = task_id
                                #     id_urgent_gen_len = 0
                            else:
                                # print("not add task due to time limitation")
                                wait_flag = True
                        else:
                            # print("not add task due to memory limitation")
                            wait_flag = True
                    
                    # process output
                    for output in step_outputs:
                        task_id = output.request_id
                        # if task_id == id_urgent:
                        #     id_urgent_gen_len = len(output.outputs[0].token_ids)
                        if output.finished:
                            wait_flag = False
                            running_num -= 1
                            task_id = output.request_id
                            result_gen = output.outputs[0].text
                            # task_id = self.taskData._read_para(agent_id,'task_id')
                            print(f"output for task:{task_id}")
                            batch_cur_size -= 1
                            # print(f"batch size: {batch_cur_size}")
                            logger.info(f"Output for task {task_id}: {result_gen}, time: {time.time()}")
                            agent_id = self.taskData._read_para(task_id,'agent_id')
                            if output.outputs[0].stop_reason == 'stop by checker':
                                final_flag = False
                                result_queues[agent_id].put((task_id, result_gen, time.time(), final_flag))
                                self._process_seg_output(output, task_id, result_gen)
                            else:
                                final_flag = True
                                result_queues[agent_id].put((task_id, result_gen, time.time(), final_flag))
                                self.taskData._abort(task_id)
                            if task_id == id_urgent:
                                id_urgent = self.taskData.get_urgent_registered_task(token_speed)
                                id_urgent_gen_len = 0
                        else: 
                            self.taskData._update_replace(task_id, 'finished_token_num', len(output.outputs[0].token_ids))


    def schedule_timely_fixed_batch(self, task_queue, result_queues, robot_system='typefly', batch_size=1):
        self.task_queue = task_queue
        batch_cur_size = 0
        while not self.global_stop_signal.is_set():
            # wait until get new requests
            # num_free_gpu = sum(
            #     scheduler.block_manager.get_num_free_gpu_blocks()
            #     for scheduler in self.llm_engine.scheduler)
            # print(f"when no task, the free gpu is {num_free_gpu}")
            while True:
                task_info = self.get_task()
                if task_info:
                    task_id, agent_id, prompt_input = task_info
                    self.llm_engine.add_request(task_id, prompt_input, self.sampling_params)
                    logger.info(f"Added task {task_id}, time: {time.time()}")
                    batch_cur_size += 1
                    break

            while self.llm_engine.has_unfinished_requests():
                # process exsiting request
                step_outputs = self.llm_engine.step()

                # decide add more request or not based on gpu util
                # num_free_gpu = sum(
                #         scheduler.block_manager.get_num_free_gpu_blocks()
                #         for scheduler in self.llm_engine.scheduler)
                # gpu_free_threhold = 1376 if self.run_mode == "sequential" else 0
                # if num_free_gpu > gpu_free_threhold:

                # decide add more request or not based on batch size
                if batch_cur_size < batch_size:
                    task_info = self.get_task()
                    if task_info:
                        task_id, agent_id, prompt_input = task_info
                        self.llm_engine.add_request(task_id, prompt_input, self.sampling_params)
                        logger.info(f"Added task {task_id}, time: {time.time()}")
                        batch_cur_size += 1
                
                # process output
                for output in step_outputs:
                    if output.finished:
                        batch_cur_size -= 1
                        task_id = output.request_id
                        result_gen = output.outputs[0].text
                        print(f"output for task:{task_id}")
                        logger.info(f"Output for task {task_id}: {result_gen}, time: {time.time()}")
                        agent_id = self.taskData._read_para(task_id,'agent_id')
                        if output.outputs[0].stop_reason == 'stop by checker':
                            final_flag = False
                            result_queues[agent_id].put((task_id, result_gen, time.time(), final_flag))
                            self._process_seg_output(output, task_id, result_gen)
                        else:
                            final_flag = True
                            print(f"final output for task:{task_id}")
                            result_queues[agent_id].put((task_id, result_gen, time.time(), final_flag))
                            self.taskData._abort(task_id)

    def schedule_edf(self, task_queue, result_queues, run_mode, batch_size=1):
        '''
        Abort this funciton
        '''
        lmax = 20
        self.task_queue = task_queue
        id_urgent_gen_len = 0
        running_num = 0
        batch_cur_size = 0
        wait_flag = False
        while not self.global_stop_signal.is_set():
            # wait until get new requests
            # num_free_gpu = sum(
            #     scheduler.block_manager.get_num_free_gpu_blocks()
            #     for scheduler in self.llm_engine.scheduler)
            # print(f"when no task, the free gpu is {num_free_gpu}")
            while True:
                task_info = self.get_task_edf()
                if task_info:
                    task_id, agent_id, prompt_input = task_info
                    id_urgent = self.taskData.get_earliest_registered_task()
                    running_num += 1
                    self.llm_engine.add_request(task_id, prompt_input, self.sampling_params)
                    logger.info(f"Added task {task_id}, time: {time.time()}")
                    batch_cur_size += 1
                    # print(f"batch size: {batch_cur_size}")
                    break

            while self.llm_engine.has_unfinished_requests():
                # process exsiting request
                step_outputs = self.llm_engine.step()

                # decide add more request or not
                if not wait_flag:
                    # memory constraint
                    if self._admit_memory_ok():
                    # if token_budget > lmax:
                        # print("pass memory constraint")
                        # latency constraint
                        tuf_urgent = self.taskData._read_para(id_urgent, 'task_tuf')
                        task_arrival_time = self.taskData._read_para(id_urgent, 'task_arrival_time')
                        ddl_urgent = tuf_urgent[1]-time.time()
                        slack_time = ddl_urgent-(time.time()-task_arrival_time)
                        reg_time = self.taskData._read_para(id_urgent, 'reg_time')
                        # id_urgent = self.taskData.get_earliest_registered_task
                        token_speed = (id_urgent_gen_len)/(time.time()-reg_time)
                        if token_speed == 0:
                            add_flag = True
                            # print(f"add task due to tokenspeed 0")
                        elif (lmax-id_urgent_gen_len)/token_speed < slack_time:
                            add_flag = True
                        if add_flag:
                            add_flag = False
                            # print("pass time constraint")
                            task_info = self.get_task_edf()
                            if task_info:
                                task_id, agent_id, prompt_input = task_info
                                running_num += 1
                                self.llm_engine.add_request(task_id, prompt_input, self.sampling_params)
                                logger.info(f"Added task {task_id}, time: {time.time()}")
                                batch_cur_size += 1
                                # print(f"batch size: {batch_cur_size}")
                        else:
                            # print("not add task due to time limitation")
                            wait_flag = True
                    else:
                        # print("not add task due to memory limitation")
                        wait_flag = True
                
                # process output
                for output in step_outputs:
                    task_id = output.request_id
                    if task_id == id_urgent:
                        id_urgent_gen_len += len(output.outputs[0].token_ids)
                    if output.finished:
                        wait_flag = False
                        running_num -= 1
                        task_id = output.request_id
                        result_gen = output.outputs[0].text
                        # task_id = self.taskData._read_para(agent_id,'task_id')
                        print(f"output for task:{task_id}")
                        batch_cur_size -= 1
                        # print(f"batch size: {batch_cur_size}")
                        logger.info(f"Output for task {task_id}: {result_gen}, time: {time.time()}")
                        agent_id = self.taskData._read_para(task_id,'agent_id')
                        result_queues[agent_id].put((task_id, result_gen, time.time(), True))
                        if output.outputs[0].stop_reason == 'stop by checker':
                            exe_time =  self.virtualdelay.delay_wc_map(result_gen)
                            task_tuf = self.taskData._read_para(task_id, 'task_tuf')
                            # self._update_segment_priority(task_id, exe_time)
                            task_cur_plan = self.taskData._read_para(task_id, 'cur_plan')
                            task_arrival_time = self.taskData._read_para(task_id, 'task_arrival_time')
                            # if task_cur_plan == "" :
                            #     seg_ddl = time.time()-task_arrival_time + exe_time
                            # else:
                            #     seg_ddl =  task_tuf[1]+exe_time
                            # task_tuf[1] = seg_ddl
                            if 'edf' in run_mode:
                                priority = task_arrival_time + task_tuf[1]
                            else:
                                priority = time.time()
                            # priority = task_arrival_time + seg_ddl
                            self.taskData._update_add(task_id, 'cur_plan', result_gen)
                            # update task input, store token id in task input
                            self.taskData._update_input(task_id, output)
                            print("readd request to task queue")
                            # self.taskData._update_replace(task_id, 'task_tuf', task_tuf)
                            task_details = self.taskData._read_task(task_id)
                            # print(f"priority for task id {task_id}: {priority}")
                            self.task_queue.put(Task(priority, task_details))
                        else:
                            self.taskData._abort(task_id)
                        if task_id == id_urgent:
                            id_urgent = self.taskData.get_earliest_registered_task()
                            id_urgent_gen_len = 0
    
    def schedule_other_fixed_batch(self, task_queue, result_queues, batch_size, run_mode):
        self.task_queue = task_queue
        batch_cur_size = 0
        while not self.global_stop_signal.is_set():
            while True:
                task_info = self.get_task_edf()
                if task_info:
                    task_id, agent_id, prompt_input = task_info
                    self.llm_engine.add_request(task_id, prompt_input, self.sampling_params)
                    logger.info(f"Added task {task_id}, time: {time.time()}")
                    batch_cur_size += 1
                    break

            while self.llm_engine.has_unfinished_requests():
                # process exsiting request
                step_outputs = self.llm_engine.step()

                if batch_cur_size < batch_size:
                    task_info = self.get_task_edf()
                    if task_info:
                        task_id, agent_id, prompt_input = task_info
                        self.llm_engine.add_request(task_id, prompt_input, self.sampling_params)
                        logger.info(f"Added task {task_id}, time: {time.time()}")
                        batch_cur_size += 1
                
                # process output
                for output in step_outputs:
                    if output.finished:
                        batch_cur_size -= 1
                        task_id = output.request_id
                        result_gen = output.outputs[0].text
                        print(f"output for task:{task_id}")
                        logger.info(f"Output for task {task_id}: {result_gen}, time: {time.time()}")
                        agent_id = self.taskData._read_para(task_id,'agent_id')
                        result_queues[agent_id].put((task_id, result_gen, time.time(), True))
                        if output.outputs[0].stop_reason == 'stop by checker':
                            task_tuf = self.taskData._read_para(task_id, 'task_tuf')
                            task_arrival_time = self.taskData._read_para(task_id, 'task_arrival_time')
                            if 'edf' in run_mode:
                                priority = task_arrival_time + task_tuf[1]
                            else:
                                priority = time.time()
                            # print(f"priority{priority}")
                            self.taskData._update_add(task_id, 'cur_plan', result_gen)
                            # update task input, store token id in task input
                            self.taskData._update_input(task_id, output)
                            print("readd request to task queue")
                            self.taskData._update_replace(task_id, 'task_tuf', task_tuf)
                            task_details = self.taskData._read_task(task_id)
                            # print(f"priority for task id {task_id}: {priority}")
                            self.task_queue.put(Task(priority, task_details))
                        else:
                            self.taskData._abort(task_id)
    
    def schedule_chatbot(self, task_queue, result_queues, batch_size):
        self.task_queue = task_queue
        batch_cur_size = 0
        while not self.global_stop_signal.is_set():
            # wait until get new requests
            # num_free_gpu = sum(
            #     scheduler.block_manager.get_num_free_gpu_blocks()
            #     for scheduler in self.llm_engine.scheduler)
            # print(f"when no task, the free gpu is {num_free_gpu}")
            while True:
                task_info = self.get_task_edf()
                if task_info:
                    task_id, agent_id, prompt_input = task_info
                    sampling_params = SamplingParams(temperature=0, skip_special_tokens=True, max_tokens=self.taskData._read_para(task_id, 'remaining_length'))
                    self.llm_engine.add_request(task_id, prompt_input, sampling_params)
                    logger.info(f"Added task {task_id}, time: {time.time()}")
                    batch_cur_size += 1
                    break

            while self.llm_engine.has_unfinished_requests():
                # process exsiting request
                step_outputs = self.llm_engine.step()

                # decide add more request or not based on gpu util
                # num_free_gpu = sum(
                #         scheduler.block_manager.get_num_free_gpu_blocks()
                #         for scheduler in self.llm_engine.scheduler)
                # gpu_free_threhold = 1376 if self.run_mode == "sequential" else 0
                # if num_free_gpu > gpu_free_threhold:

                # decide add more request or not based on batch size
                if batch_cur_size < batch_size:
                    task_info = self.get_task_edf()
                    if task_info:
                        task_id, agent_id, prompt_input = task_info
                        sampling_params = SamplingParams(temperature=0, skip_special_tokens=True, max_tokens=self.taskData._read_para(task_id, 'remaining_length'))
                        self.llm_engine.add_request(task_id, prompt_input, sampling_params)
                        logger.info(f"Added task {task_id}, time: {time.time()}")
                        batch_cur_size += 1
                
                # process output
                for output in step_outputs:
                    if output.finished:
                        batch_cur_size -= 1
                        task_id = output.request_id
                        result_gen = output.outputs[0].text
                        print(f"output for task:{task_id}")
                        logger.info(f"Output for task {task_id}: {result_gen}, time: {time.time()}")
                        agent_id = self.taskData._read_para(task_id,'agent_id')
                        task_arrival_time = self.taskData._read_para(task_id, 'task_arrival_time')
                        print(f"Putting result for task {task_id} (agent {agent_id}) into queue: {result_gen}") # Added for debugging
                        result_queues[agent_id].put((task_id, result_gen, time.time(), True, task_arrival_time))
                        if output.outputs[0].stop_reason == 'stop by checker':
                            # print(f"stop reason: {output.outputs[0].stop_reason}")
                            self._process_chat_seg_output(output, task_id, result_gen)
                        else:
                            self.taskData._abort(task_id)

    def schedule_timely_chatbot(self, task_queue, result_queues, robot_system='chatbot'):
        """
        Timely scheduler for chatbot application with TUF-based priority scheduling
        Combines the advantages of schedule_timely (TUF-based scheduling with _update_queue_priority) 
        and schedule_chatbot (chatbot-specific output format and segment processing)
        """
        lmax = self.lmax # max token length of one segment
        self.task_queue = task_queue
        id_urgent_gen_len = 0
        running_num = 0
        batch_cur_size = 0
        wait_flag = False
        token_speed = 0 # initial token speed
        
        while not self.global_stop_signal.is_set():
            # wait until get new requests
            while True:
                task_info = self.get_task()  # Use get_task() to call _update_queue_priority()
                if task_info:
                    task_id, agent_id, prompt_input = task_info
                    id_urgent = self.taskData.get_urgent_registered_task(token_speed)
                    running_num += 1
                    # Use dynamic sampling params based on remaining_length
                    sampling_params = SamplingParams(temperature=0, skip_special_tokens=True, max_tokens=self.taskData._read_para(task_id, 'remaining_length'))
                    self.llm_engine.add_request(task_id, prompt_input, sampling_params)
                    logger.info(f"Added task {task_id}, time: {time.time()}")
                    batch_cur_size += 1
                    # print(f"batch size: {batch_cur_size}")
                    break

            while self.llm_engine.has_unfinished_requests():
                # process existing request
                step_outputs = self.llm_engine.step()

                # decide add more request or not
                if not wait_flag:
                    # memory constraint
                    if self._admit_memory_ok():
                        # latency constraint
                        exe_end_pre = self.taskData._read_para(id_urgent, 'exe_end_pre')
                        slack_time = -(time.time()+self.comm_time-exe_end_pre)
                        reg_time = self.taskData._read_para(id_urgent, 'reg_time')

                        id_urgent_gen_len = self.taskData._read_para(id_urgent, 'finished_token_num')
                        token_speed = (id_urgent_gen_len)/(time.time()-reg_time)
                        
                        if token_speed == 0:
                            add_flag = True
                        elif (lmax-id_urgent_gen_len)/token_speed < slack_time:
                            add_flag = True
                        elif slack_time < 0:
                            print(f"not add task due to slack time {slack_time}")
                            add_flag = False
                        else:
                            print(f"not add token_speed: {token_speed}")
                            add_flag = False

                        if add_flag:
                            add_flag = False
                            task_info = self.get_task()  # Use get_task() for TUF-based scheduling
                            id_urgent = self.taskData.get_urgent_registered_task(token_speed)
                            if task_info:
                                task_id, agent_id, prompt_input = task_info
                                running_num += 1
                                # Use dynamic sampling params
                                sampling_params = SamplingParams(temperature=0, skip_special_tokens=True,max_tokens=self.taskData._read_para(task_id, 'remaining_length'))
                                self.llm_engine.add_request(task_id, prompt_input, sampling_params)
                                logger.info(f"Added task {task_id}, time: {time.time()}")
                                batch_cur_size += 1
                                # print(f"batch size: {batch_cur_size}")
                        else:
                            wait_flag = True
                    else:
                        wait_flag = True
                
                # process output
                for output in step_outputs:
                    task_id = output.request_id
                    if output.finished:
                        wait_flag = False
                        running_num -= 1
                        task_id = output.request_id
                        result_gen = output.outputs[0].text
                        # print(f"output for task:{task_id}")
                        batch_cur_size -= 1
                        # print(f"batch size: {batch_cur_size}")
                        logger.info(f"Output for task {task_id}: {result_gen}, time: {time.time()}")
                        agent_id = self.taskData._read_para(task_id,'agent_id')
                        task_arrival_time = self.taskData._read_para(task_id, 'task_arrival_time')
                        
                        # Use 5-element output format for chatbot
                        if output.outputs[0].stop_reason == 'stop by checker':
                            final_flag = False
                            result_queues[agent_id].put((task_id, result_gen, time.time(), final_flag, task_arrival_time))
                            # Use chat-specific segment processing
                            self._process_chat_seg_output_timely(output, task_id, result_gen)
                        else:
                            final_flag = True
                            result_queues[agent_id].put((task_id, result_gen, time.time(), final_flag, task_arrival_time))
                            self.taskData._abort(task_id)
                        
                        if task_id == id_urgent:
                            id_urgent = self.taskData.get_urgent_registered_task(token_speed)
                            id_urgent_gen_len = 0
                    else: 
                        self.taskData._update_replace(task_id, 'finished_token_num', len(output.outputs[0].token_ids))

    # --------------------------------  Methods based on vLLM ------------------------------------------ #

    def vllm_schedule(self, task_queue, result_queues):
        # baseline for continous batching in vllm
        self.task_queue = task_queue
        batch_cur_size = 0
        while not self.global_stop_signal.is_set():
            # wait until get new requests
            # num_free_gpu = sum(
            #     scheduler.block_manager.get_num_free_gpu_blocks()
            #     for scheduler in self.llm_engine.scheduler)
            # print(f"when no task, the free gpu is {num_free_gpu}")
            while True:
                task_info = self.get_task_normal()
                if task_info:
                    task_id, agent_id, prompt_input = task_info
                    self.llm_engine.add_request(task_id, prompt_input, self.sampling_params)
                    logger.info(f"Added task {task_id}, time: {time.time()}")
                    batch_cur_size += 1
                    # print(f"batch size: {batch_cur_size}")
                    break

            while self.llm_engine.has_unfinished_requests():
                # process exsiting request
                step_outputs = self.llm_engine.step()
                # running_num = len(self.llm_engine.scheduler[0].running)
                # print(f"current batch size: {batch_cur_size} running request number: {running_num} ")

                # waiting_num = self.llm_engine.get_num_unfinished_requests()
                # print(f"current batch size: {batch_cur_size} unifinished request number: {waiting_num} ")

                task_info = self.get_task_normal()
                if task_info:
                    task_id, agent_id, prompt_input = task_info
                    self.llm_engine.add_request(task_id, prompt_input, self.sampling_params)
                    logger.info(f"Added task {task_id}, time: {time.time()}")
                    batch_cur_size += 1
                    # print(f"batch size: {batch_cur_size}")
                
                # process output
                for output in step_outputs:
                    if output.finished:
                        task_id = output.request_id
                        result_gen = output.outputs[0].text
                        # task_id = self.taskData._read_para(agent_id,'task_id')
                        print(f"output for task:{task_id}")
                        logger.info(f"Output for task {task_id}: {result_gen}, time: {time.time()}")
                        agent_id = self.taskData._read_para(task_id,'agent_id')
                        final_flag = True
                        result_queues[agent_id].put((task_id, result_gen, time.time(), final_flag))
                        self.taskData._abort(task_id)
                        batch_cur_size -= 1
                        # print(f"batch size: {batch_cur_size}")

    def vllm_schedule_fixed_batch(self, task_queue, result_queues, batch_size):
        # baseline for continous batching in vllm
        self.task_queue = task_queue
        batch_cur_size = 0
        while not self.global_stop_signal.is_set():
            # wait until get new requests
            # num_free_gpu = sum(
            #     scheduler.block_manager.get_num_free_gpu_blocks()
            #     for scheduler in self.llm_engine.scheduler)
            # print(f"when no task, the free gpu is {num_free_gpu}")
            while True:
                task_info = self.get_task_normal()
                if task_info:
                    task_id, agent_id, prompt_input = task_info
                    self.llm_engine.add_request(task_id, prompt_input, self.sampling_params)
                    logger.info(f"Added task {task_id}, time: {time.time()}")
                    batch_cur_size += 1
                    # print(f"batch size: {batch_cur_size}")
                    break

            while self.llm_engine.has_unfinished_requests():
                # process exsiting request
                step_outputs = self.llm_engine.step()

                if batch_cur_size < batch_size:
                    task_info = self.get_task_normal()
                    if task_info:
                        task_id, agent_id, prompt_input = task_info
                        self.llm_engine.add_request(task_id, prompt_input, self.sampling_params)
                        logger.info(f"Added task {task_id}, time: {time.time()}")
                        batch_cur_size += 1
                        # print(f"batch size: {batch_cur_size}")
                
                # process output
                for output in step_outputs:
                    if output.finished:
                        task_id = output.request_id
                        result_gen = output.outputs[0].text
                        # task_id = self.taskData._read_para(agent_id,'task_id')
                        print(f"output for task:{task_id}")
                        logger.info(f"Output for task {task_id}: {result_gen}, time: {time.time()}")
                        batch_cur_size -= 1
                        # print(f"batch size: {batch_cur_size}")
                        agent_id = self.taskData._read_para(task_id,'agent_id')
                        final_flag = True
                        result_queues[agent_id].put((task_id, result_gen, time.time(), final_flag))
                        self.taskData._abort(task_id)
      
    def vllm_schedule_stream(self, task_queue, result_queues, robot_system='typefly'):
        # baseline for continous batching in vllm
        self.task_queue = task_queue
        batch_cur_size = 0
        while not self.global_stop_signal.is_set():
            # wait until get new requests
            # num_free_gpu = sum(
            #     scheduler.block_manager.get_num_free_gpu_blocks()
            #     for scheduler in self.llm_engine.scheduler)
            # print(f"when no task, the free gpu is {num_free_gpu}")
            while True:
                task_info = self.get_task_normal()
                if task_info:
                    task_id, agent_id, prompt_input = task_info
                    sampling_params = SamplingParams(temperature=0, skip_special_tokens=True, max_tokens=self.taskData._read_para(task_id, 'output_length'))
                    self.llm_engine.add_request(task_id, prompt_input, sampling_params)
                    logger.info(f"Added task {task_id}, time: {time.time()}")
                    batch_cur_size += 1
                    # print(f"batch size: {batch_cur_size}")
                    break

            while self.llm_engine.has_unfinished_requests():
                # process exsiting request
                step_outputs = self.llm_engine.step()

                task_info = self.get_task_normal()
                if task_info:
                    task_id, agent_id, prompt_input = task_info
                    sampling_params = SamplingParams(temperature=0, skip_special_tokens=True,max_tokens=self.taskData._read_para(task_id, 'output_length'))
                    self.llm_engine.add_request(task_id, prompt_input, sampling_params)
                    logger.info(f"Added task {task_id}, time: {time.time()}")
                    batch_cur_size += 1
                    # print(f"batch size: {batch_cur_size}")
                
                # process output
                for output in step_outputs:
                    result_gen = output.outputs[0].text
                    task_id = output.request_id
                    task_cur_plan = self.taskData._read_para(task_id, 'cur_plan')
                    # new_plan = result_gen - task_cur_plan
                    new_plan = result_gen.replace(task_cur_plan, "")

                    if robot_system == 'typefly':
                        flag_executable = self.program_interpreter.parse(new_plan, True)
                    elif robot_system == 'fltrnn':
                        if ';' in new_plan:
                            flag_executable =  True
                        else: 
                            flag_executable = False
                    elif robot_system == 'chatbot':
                        if '.' in new_plan:
                            flag_executable =  True
                        else: 
                            flag_executable = False
                    else:
                        print("Add support for new robot system")

                    # print(f"new plan {new_plan}, flag_executable: {flag_executable}")
                    if flag_executable:
                        print(f"output for task{task_id}: {new_plan}")
                        logger.info(f"Output for task {task_id}: {new_plan}, time: {time.time()}")
                        agent_id = self.taskData._read_para(task_id,'agent_id')
                        task_arrival_time = self.taskData._read_para(task_id, 'task_arrival_time')
                        result_queues[agent_id].put((task_id, new_plan, time.time(), False, task_arrival_time))
                        self.taskData._update_add(task_id, 'cur_plan', new_plan)
                    if output.finished:
                        # result_gen = output.outputs[0].text
                        # task_id = self.taskData._read_para(agent_id,'task_id')
                        print(f"output for task:{task_id}")
                        logger.info(f"Output for task {task_id}: {new_plan}, time: {time.time()}")
                        batch_cur_size -= 1
                        # print(f"batch size: {batch_cur_size}")
                        agent_id = self.taskData._read_para(task_id,'agent_id')
                        task_arrival_time = self.taskData._read_para(task_id, 'task_arrival_time')
                        result_queues[agent_id].put((task_id, new_plan, time.time(), False, task_arrival_time))
                        self.taskData._abort(task_id)

    def vllm_chatbot(self, task_queue, result_queues):
        # baseline for continous batching in vllm
        self.task_queue = task_queue
        batch_cur_size = 0
        while not self.global_stop_signal.is_set():
            # wait until get new requests
            # num_free_gpu = sum(
            #     scheduler.block_manager.get_num_free_gpu_blocks()
            #     for scheduler in self.llm_engine.scheduler)
            # print(f"when no task, the free gpu is {num_free_gpu}")
            while True:
                task_info = self.get_task_normal()
                if task_info:
                    task_id, agent_id, prompt_input = task_info
                    self.llm_engine.add_request(task_id, prompt_input, self.sampling_params)
                    logger.info(f"Added task {task_id}, time: {time.time()}")
                    batch_cur_size += 1
                    # print(f"batch size: {batch_cur_size}")
                    break

            while self.llm_engine.has_unfinished_requests():
                # process exsiting request
                step_outputs = self.llm_engine.step()

                task_info = self.get_task_normal()
                if task_info:
                    task_id, agent_id, prompt_input = task_info
                    self.llm_engine.add_request(task_id, prompt_input, self.sampling_params)
                    logger.info(f"Added task {task_id}, time: {time.time()}")
                    batch_cur_size += 1
                    # print(f"batch size: {batch_cur_size}")
                
                # process output
                for output in step_outputs:
                    if output.finished:
                        task_id = output.request_id
                        result_gen = output.outputs[0].text
                        # task_id = self.taskData._read_para(agent_id,'task_id')
                        print(f"output for task:{task_id}")
                        logger.info(f"Output for task {task_id}: {task_id}, time: {time.time()}")
                        agent_id = self.taskData._read_para(task_id,'agent_id')
                        task_arrival_time = self.taskData._read_para(task_id, 'task_arrival_time')
                        result_queues[agent_id].put((task_id, result_gen,time.time(), True, task_arrival_time))
                        self.taskData._abort(task_id)
                        batch_cur_size -= 1
                        # print(f"batch size: {batch_cur_size}")
    
    def vllm_chatbot_stream(self, task_queue, result_queues):
        # streaming chatbot version - outputs intermediate results like vllm_schedule_stream
        # but without executable checking like vllm_chatbot
        self.task_queue = task_queue
        batch_cur_size = 0
        while not self.global_stop_signal.is_set():
            # wait until get new requests
            while True:
                task_info = self.get_task_normal()
                if task_info:
                    task_id, agent_id, prompt_input = task_info
                    self.llm_engine.add_request(task_id, prompt_input, self.sampling_params)
                    logger.info(f"Added task {task_id}, time: {time.time()}")
                    batch_cur_size += 1
                    # print(f"batch size: {batch_cur_size}")
                    break

            while self.llm_engine.has_unfinished_requests():
                # process exsiting request
                step_outputs = self.llm_engine.step()

                task_info = self.get_task_normal()
                if task_info:
                    task_id, agent_id, prompt_input = task_info
                    self.llm_engine.add_request(task_id, prompt_input, self.sampling_params)
                    logger.info(f"Added task {task_id}, time: {time.time()}")
                    batch_cur_size += 1
                    # print(f"batch size: {batch_cur_size}")
                
                # process output
                for output in step_outputs:
                    result_gen = output.outputs[0].text
                    task_id = output.request_id
                    task_cur_text = self.taskData._read_para(task_id, 'cur_plan')
                    # compute the incremental new text
                    new_text = result_gen.replace(task_cur_text, "")

                    # stream output for chatbot - no executable checking needed
                    if new_text:  # only send if there's new content
                        # print(f"streaming output for task:{task_id}")
                        logger.info(f"Output for task {task_id}: {new_text}, time: {time.time()}")
                        agent_id = self.taskData._read_para(task_id,'agent_id')
                        task_arrival_time = self.taskData._read_para(task_id, 'task_arrival_time')
                        result_queues[agent_id].put((task_id, new_text, time.time(), False, task_arrival_time))
                        self.taskData._update_add(task_id, 'cur_plan', new_text)
                    
                    if output.finished:
                        # send final signal when task is complete
                        print(f"final output for task:{task_id}")
                        logger.info(f"Output for task {task_id}: {new_text}, time: {time.time()}")
                        batch_cur_size -= 1
                        # print(f"batch size: {batch_cur_size}")
                        agent_id = self.taskData._read_para(task_id,'agent_id')
                        task_arrival_time = self.taskData._read_para(task_id, 'task_arrival_time')
                        result_queues[agent_id].put((task_id, new_text, time.time(), True, task_arrival_time))
                        self.taskData._abort(task_id)



def infer_start(task_queue, result_queues, prompt_path, agent_num, global_stop_signal, run_mode, robot_system='typefly', batchsize=1, seg_exe=0.09, lmax =10, comm_time=0, real_audio_task_ids=None, prompt_speech_path=None, model_path_override=None):
    # llm_device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # for normal generation
    # plan_gen = SerialPlanner(plan_rec_path, global_stop_signal, agent_num,
    #                          run_mode = "normal", use_cache=False)

    # Use override if provided, otherwise fall back to module-level default
    effective_model_path = model_path_override if model_path_override is not None else model_path

    # for preemptive generation
    scheduler = RequestScheduler(prompt_path, global_stop_signal, agent_num, seg_exe, lmax,
                              run_mode= run_mode, use_cache=False, comm_time=comm_time, robot_system=robot_system,
                              real_audio_task_ids=real_audio_task_ids, prompt_speech_path=prompt_speech_path,
                              model_path=effective_model_path)
    if run_mode =='vllm' or run_mode =='vllm-edf':
        scheduler.vllm_schedule(task_queue, result_queues)
    elif run_mode =='vllm-fixed':
        scheduler.vllm_schedule_fixed_batch(task_queue, result_queues, batch_size=batchsize)
    elif run_mode == 'vllm-stream':
        scheduler.vllm_schedule_stream(task_queue, result_queues, robot_system)
    elif run_mode == 'vllm-chatbot':
        scheduler.vllm_chatbot(task_queue, result_queues)
    elif run_mode == 'vllm-chatbot-stream':
        scheduler.vllm_schedule_stream(task_queue, result_queues, robot_system)

    elif run_mode =='rtllm':
        scheduler.schedule_timely(task_queue, result_queues,robot_system)
        print("start schedule_timely")
    elif run_mode == 'rtllm-fixed':
        scheduler.schedule_timely_fixed_batch(task_queue, result_queues, robot_system, batch_size=batchsize)
        print("start schedule_timely_fixed_batch")
    elif run_mode == 'rtllm-edf':
        print("start schedule_edf")
        scheduler.schedule_edf(task_queue, result_queues, run_mode, batch_size=batchsize)
    elif run_mode == 'rtllm-edf-fixed' or run_mode == 'rtllm-fcfs-fixed':
        scheduler.schedule_other_fixed_batch(task_queue, result_queues, batch_size=batchsize, run_mode=run_mode)
    elif run_mode == 'rtllm-chatbot':
        scheduler.schedule_chatbot(task_queue, result_queues, batch_size=batchsize)
    elif run_mode == 'rtllm-chatbot-timely':
        scheduler.schedule_timely_chatbot(task_queue, result_queues, robot_system)
        print("start schedule_timely_chatbot")

if __name__ == "__main__":
    task_queue=1
    result_queues=2
    plan_rec_path=3
    agent_num=2
    global_stop_signal=False
    infer_start(task_queue, result_queues, plan_rec_path, agent_num, global_stop_signal)