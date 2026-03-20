import time
import random
import threading
import queue
import re
import json
import os
from util.log_config import logger
from executor.minispec_executor.llm_controller import LLMController
from executor.minispec_executor.robot_info import RobotInfo

# from log_config import logger
# from minispec_executor.llm_controller import LLMController
# from minispec_executor.robot_info import RobotInfo

class ActionPerform:
    def __init__(self, robot_info_path: str):  
        print("ActionPerform initialized.")
        self.robot_info_path = robot_info_path
    
    def read_robot_info(self, agent_id: int) -> list[RobotInfo]:
        """Read robot information from a JSON file and return a list of RobotInfo, excluding agent_id."""
        file_path = os.path.join(os.path.dirname(__file__), self.robot_info_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Robot info file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            robot_info_list = json.load(f)
        
        # Filter robots based on agent_id
        filtered_robot_dicts = [info for info in robot_info_list if info['agent_id'] == agent_id]
        
        if not filtered_robot_dicts:
            raise ValueError(f"No robot information found for agent_id {agent_id}")
        
        # Remove 'agent_id' field and convert to RobotInfo
        robot_infos = []
        for info in filtered_robot_dicts:
            info.pop('agent_id', None)
            robot_infos.append(RobotInfo.from_dict(info))
        
        return robot_infos

    def execute_task(self, agent_id:int, result_queue, stop_signal, comm_time):
        """Thread target function to execute tasks from result queue."""
        robot_info_list = self.read_robot_info(agent_id)
        self.llm_controller = LLMController(robot_info_list, message_queue=None)
        # build connection with agent
        self.llm_controller.start_controller()
        past_task_id = []
        end_flag = False
        while not stop_signal.is_set():
            try:
                task_id, result, _, end_flag = result_queue.get(timeout=1)
                if task_id not in past_task_id:
                    past_task_id.append(task_id)
                    eval_flag = True
                    key = f"<plan, robot{agent_id}>"
                    plan_input = {
                        key: result
                    }
                    # key = f"<plan, robot{agent_id}>:"
                    plan_input = json.dumps(plan_input)
                    plan_input = plan_input.removesuffix('"}')
                    print(f"Agent {agent_id} received task {task_id} with plan: {plan_input}")
                else:
                    eval_flag = False
                    plan_input = result
                    print(f"Agent {agent_id} received task {task_id} with plan: {plan_input}")

                if end_flag:
                    plan_input = plan_input + '"}'
                    print(f"Agent {agent_id} received task {task_id} with plan: {plan_input}")

                # send action plan to agent
                start_time = time.time()
                logger.info(f"Start send plan for task {task_id} for agent {agent_id} on time {start_time} with plan {result}")
                # print(f"Agent {agent_id} executing task {task_id} with plan: {result}")
                self.llm_controller.execute_task(plan_input, eval_flag, task_id)
                end_time = time.time()
                logger.info(f"Finish send plan for task {task_id} for agent {agent_id} on time {end_time} with plan {result}")
            except queue.Empty:
                continue
        self.llm_controller.stop_controller()
        print("Task execution thread shutting down.")



def task_executor(result_queues, agent_num:int, stop_signal, robot_info_path:str, comm_time = 0):
    """Process 3: Manage threads that execute tasks from the result queue."""
    def execute_task_wrapper(agent_id, result_queue, stop_signal):
        actionperform = ActionPerform(robot_info_path)
        actionperform.execute_task(agent_id, result_queue, stop_signal, comm_time)
    
    agents = [threading.Thread(target=execute_task_wrapper, args=(i, result_queues[i],stop_signal)) for i in range(agent_num)]
    for agent in agents:
        agent.start()
    
    for agent in agents:
        agent.join()  # Wait for all threads to finish

    print("Task executor process shutting down.")



if __name__ == "__main__":
    agent_num = 2
    robot_info_path = "robot_info.json"
    result_queues = {i: queue.Queue() for i in range(agent_num)}


    input_dog = "?s('sports ball')==True{g('sports ball')};"
    input_car1 = "?scan('car')==True{move(30,0)};"
    input_car2 = "?scan('sports ball')==True{move(30,0)};"
    # input_car1 = "?scan('car')==True{goto('car');d(1)}->False;"
    # input_car2 = "?scan('mouse')==True{goto('mouse');d(1);scan('toy')}->False;"
    result_queues[0].put((0, input_car1, time.time()))
    result_queues[1].put((1, input_car2, time.time()))
    # result_queues[2].put((2, input_dog, time.time()))
    # input_car1_2 = "{g('orange')};"
    # result_queues[1].put((1, input_car1_2, time.time()))

    stop_signal = threading.Event()
    def stop_later():
        time.sleep(600)  # Wait for 10 minutes before stopping
        stop_signal.set()
    threading.Thread(target=stop_later).start()

    task_executor(result_queues=result_queues, agent_num=agent_num, stop_signal=stop_signal, robot_info_path=robot_info_path)
