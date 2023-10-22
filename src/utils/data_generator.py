import random

import pandas as pd


class DataGenerator:
    def __init__(self,
                 num_pods: int = 1000,
                 num_nodes: int = 10,
                 cpu_usage_min: float = 1,
                 cpu_usage_max: float = 100,
                 memory_usage_min: float = 1,
                 memory_usage_max: float = 6000):
        self.num_pods = num_pods
        self.num_nodes = num_nodes
        self.cpu_usage_min = cpu_usage_min
        self.cpu_usage_max = cpu_usage_max
        self.memory_usage_min = memory_usage_min
        self.memory_usage_max = memory_usage_max

    def generate(self) -> pd.DataFrame:
        nodes = [f"node_{i}" for i in range(1, self.num_nodes + 1)]
        pods = [f"pod_{i}" for i in range(1, self.num_pods + 1)]

        data = []
        for i in range(self.num_pods):
            node = random.choice(nodes)
            pod = pods[i]
            cpu_usage = round(random.uniform(self.cpu_usage_min, self.cpu_usage_max), 2)
            memory_usage = round(random.uniform(self.memory_usage_min, self.memory_usage_max), 2)
            data.append([node, pod, cpu_usage, memory_usage])

        return pd.DataFrame(data, columns=["node_name", "pod_name", "cpu_usage", "memory_usage"])
