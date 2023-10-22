import numpy as np
import pandas as pd
import logging


class BaseOptimizer:

    METRIC_MULTI = 'multi'
    METRIC_MEMORY = 'memory'
    METRIC_CPU = 'cpu'

    def __init__(self, dataframe: pd.DataFrame):
        if dataframe.empty:
            raise ValueError('Dataframe must not be empty')
        if dataframe.columns.tolist() != [
            'node_name',
            'pod_name',
            'cpu_usage',
            'memory_usage',
            'cpu_class',
            'memory_class'
        ]:
            raise ValueError('Dataframe must have the following columns: '
                             'node_name, pod_name, cpu_usage, memory_usage, cpu_class, memory_class')
        self.dataframe = dataframe.sort_values(by=['pod_name'])
        self.nodes = self.dataframe['node_name'].unique()
        self.nodes.sort()
        self.pods = self.dataframe['pod_name'].unique()
        self.pods.sort()
        self.amount_nodes = len(self.nodes)
        self.amount_pods = len(self.pods)
        self.pods_memory = self.dataframe.groupby('pod_name')['memory_usage'].sum().values
        self.pods_cpu = self.dataframe.groupby('pod_name')['cpu_usage'].sum().values
        self.pods_memory_classification = self.dataframe.groupby('pod_name')['memory_class'].sum().values
        self.pods_cpu_classification = self.dataframe.groupby('pod_name')['cpu_class'].sum().values
        self.allocation_matrix = np.zeros([self.amount_pods, self.amount_nodes])
        for i, pod in enumerate(self.pods):
            allocated_node = self.dataframe[self.dataframe['pod_name'] == pod]['node_name'].values[0]
            self.allocation_matrix[i, self.nodes == allocated_node] = 1

    def optimization_to_dataframe(self, allocation_suggestion: np.array) -> pd.DataFrame:
        optimized_list = []
        for pod_index in range(len(self.pods)):
            node_index = np.where(allocation_suggestion[pod_index, :] == 1)[0][0]
            optimized_list.append({
                'node_name': self.nodes[node_index],
                'pod_name': self.pods[pod_index],
                'cpu_usage': self.pods_cpu[pod_index],
                'memory_usage': self.pods_memory[pod_index],
                'cpu_class': self.pods_cpu_classification[pod_index],
                'memory_class': self.pods_memory_classification[pod_index],
            })
        optimized_dataframe = pd.DataFrame(optimized_list, columns=self.dataframe.columns)
        return optimized_dataframe

    @staticmethod
    def add_log(msg: str):
        logging.info(f'>> {msg}')

    def calc_fitness(self, allocation_suggestion: np.array, metric: str = METRIC_MULTI) -> float:
        if metric == self.METRIC_MULTI:
            mem_consumption_by_nodes = (self.pods_memory.reshape(-1, 1) * allocation_suggestion).sum(axis=0)
            cpu_consumption_by_nodes = (self.pods_cpu.reshape(-1, 1) * allocation_suggestion).sum(axis=0)

            prop_memory_consumption_by_nodes = np.array([i / sum(mem_consumption_by_nodes) for i in mem_consumption_by_nodes])
            prop_cpu_consumption_by_nodes = np.array([i / sum(cpu_consumption_by_nodes) for i in cpu_consumption_by_nodes])

            perc_amount_class_per_node_mem = []
            for node in range(len(self.nodes)):
                list_reqs_allocated = np.where(allocation_suggestion[:, node] == 1)[0]
                amount_class = np.array([0 for _ in range(max(self.pods_memory_classification) + 1)])
                for req in list_reqs_allocated:
                    class_index = int(self.pods_memory_classification[req])
                    amount_class[class_index] = amount_class[class_index] + 1
                perc = np.divide(amount_class, sum(amount_class)) if sum(amount_class) > 0 else 0
                perc_amount_class_per_node_mem.append(perc)

            perc_amount_class_per_node_cpu = []
            for node in range(len(self.nodes)):
                list_reqs_allocated = np.where(allocation_suggestion[:, node] == 1)[0]
                amount_class = np.array([0 for _ in range(max(self.pods_cpu_classification) + 1)])
                for req in list_reqs_allocated:
                    class_index = int(self.pods_cpu_classification[req])
                    amount_class[class_index] = amount_class[class_index] + 1

                perc = np.divide(amount_class, sum(amount_class)) if sum(amount_class) > 0 else 0
                perc_amount_class_per_node_cpu.append(perc)

            cpu_class_prop = np.array([
                np.bincount(self.pods_cpu_classification)[i] / sum(np.bincount(self.pods_cpu_classification))
                for i in range(max(self.pods_cpu_classification) + 1)
            ])

            memory_class_prop = np.array([
                np.bincount(self.pods_memory_classification)[i] / sum(np.bincount(self.pods_memory_classification))
                for i in range(max(self.pods_memory_classification) + 1)
            ])

            mem_class_dist = np.mean(
                [np.sqrt(sum((s - memory_class_prop) ** 2)) for s in perc_amount_class_per_node_mem])
            cpu_class_dist = np.mean([np.sqrt(sum((s - cpu_class_prop) ** 2)) for s in perc_amount_class_per_node_cpu])
            mem_cons_kpi = sum(
                -prop_memory_consumption_by_nodes * [np.log(i) if i > 0 else 0 for i in
                                                     prop_memory_consumption_by_nodes]
            ) / np.log(len(self.nodes))
            cpu_cons_kpi = sum(
                -prop_cpu_consumption_by_nodes * [np.log(i) if i > 0 else 0 for i in prop_cpu_consumption_by_nodes]
            ) / np.log(len(self.nodes))

            return (1 / mem_cons_kpi) + (1 / cpu_cons_kpi) + mem_class_dist + cpu_class_dist
        elif metric == self.METRIC_MEMORY:
            mem_consumption_by_nodes = (self.pods_memory.reshape(-1, 1) * allocation_suggestion).sum(axis=0)
            prop_memory_consumption_by_nodes = np.array(
                [i / sum(mem_consumption_by_nodes) for i in mem_consumption_by_nodes])

            perc_amount_class_per_node_mem = []
            for node in range(len(self.nodes)):
                list_reqs_allocated = np.where(allocation_suggestion[:, node] == 1)[0]
                amount_class = np.array([0 for _ in range(max(self.pods_memory_classification) + 1)])
                for req in list_reqs_allocated:
                    class_index = int(self.pods_memory_classification[req])
                    amount_class[class_index] = amount_class[class_index] + 1
                perc = np.divide(amount_class, sum(amount_class)) if sum(amount_class) > 0 else 0
                perc_amount_class_per_node_mem.append(perc)

            memory_class_prop = np.array([
                np.bincount(self.pods_memory_classification)[i] / sum(np.bincount(self.pods_memory_classification))
                for i in range(max(self.pods_memory_classification) + 1)
            ])

            mem_class_dist = np.mean(
                [np.sqrt(sum((s - memory_class_prop) ** 2)) for s in perc_amount_class_per_node_mem])
            mem_cons_kpi = sum(
                -prop_memory_consumption_by_nodes * [np.log(i) if i > 0 else 0 for i in
                                                     prop_memory_consumption_by_nodes]
            ) / np.log(len(self.nodes))

            return (1 / mem_cons_kpi) + mem_class_dist
        else:
            cpu_consumption_by_nodes = (self.pods_cpu.reshape(-1, 1) * allocation_suggestion).sum(axis=0)
            prop_cpu_consumption_by_nodes = np.array(
                [i / sum(cpu_consumption_by_nodes) for i in cpu_consumption_by_nodes])

            perc_amount_class_per_node_cpu = []
            for node in range(len(self.nodes)):
                list_reqs_allocated = np.where(allocation_suggestion[:, node] == 1)[0]
                amount_class = np.array([0 for _ in range(max(self.pods_cpu_classification) + 1)])
                for req in list_reqs_allocated:
                    class_index = int(self.pods_cpu_classification[req])
                    amount_class[class_index] = amount_class[class_index] + 1

                perc = np.divide(amount_class, sum(amount_class)) if sum(amount_class) > 0 else 0
                perc_amount_class_per_node_cpu.append(perc)

            cpu_class_prop = np.array([
                np.bincount(self.pods_cpu_classification)[i] / sum(np.bincount(self.pods_cpu_classification))
                for i in range(max(self.pods_cpu_classification) + 1)
            ])

            cpu_class_dist = np.mean([np.sqrt(sum((s - cpu_class_prop) ** 2)) for s in perc_amount_class_per_node_cpu])
            cpu_cons_kpi = sum(
                -prop_cpu_consumption_by_nodes * [np.log(i) if i > 0 else 0 for i in prop_cpu_consumption_by_nodes]
            ) / np.log(len(self.nodes))

            return (1 / cpu_cons_kpi) + cpu_class_dist
