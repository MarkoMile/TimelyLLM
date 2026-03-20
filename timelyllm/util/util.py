from dataclasses import dataclass, field

# @dataclass(order=True)
# class Task:
#     priority: int
#     task_detail: dict = field(compare=False) 
class Task:
    def __init__(self, priority, task_detail):
        self.priority = priority
        self.task_detail = task_detail
    
    def __lt__(self, other):
        return self.priority < other.priority