import pandas as pd
import numpy as np
import random


class BaseOptimizer:
    def __init__(self,
                 n_requests,
                 n_nodes,
                 memory_consumption,
                 cpu_consumption,
                 memory_classification,
                 cpu_classification
                 ):

        self.n_requests = n_requests
        self.n_nodes = n_nodes
        self.memory_consumption = memory_consumption
        self.memory_classification = memory_classification
        self.cpu_consumption = cpu_consumption
        self.cpu_classification = cpu_classification

    def _suggestion2alloc(self, individual):
        alloc = np.zeros([self.n_requests, self.n_nodes])
        for pod, node in enumerate(individual):
            alloc[pod, node] = 1

        return alloc

    def calc_fitness(self, alloc_suggestion, is_suggestion_in_mat_format=False):
        if not is_suggestion_in_mat_format:
            alloc = self._suggestion2alloc(alloc_suggestion)
        else:
            alloc = alloc_suggestion

        mem_consumption_by_nodes = (self.memory_consumption.reshape(-1, 1) * alloc).sum(axis=0)
        cpu_consumption_by_nodes = (self.cpu_consumption.reshape(-1, 1) * alloc).sum(axis=0)

        prop_memory_consumption_by_nodes = np.array([i / sum(mem_consumption_by_nodes) for i in mem_consumption_by_nodes])
        prop_cpu_consumption_by_nodes = np.array([i / sum(cpu_consumption_by_nodes) for i in cpu_consumption_by_nodes])

        perc_amount_class_per_node_mem = []
        for node in range(self.n_nodes):
            list_reqs_allocated = np.where(alloc[:, node] == 1)[0]
            amount_class = np.array([0 for _ in range(max(self.memory_classification) + 1)])
            for req in list_reqs_allocated:
                class_index = int(self.memory_classification[req])
                amount_class[class_index] = amount_class[class_index] + 1
            perc = np.divide(amount_class, sum(amount_class)) if sum(amount_class) > 0 else 0
            perc_amount_class_per_node_mem.append(perc)

        perc_amount_class_per_node_cpu = []
        for node in range(self.n_nodes):
            list_reqs_allocated = np.where(alloc[:, node] == 1)[0]
            amount_class = np.array([0 for _ in range(max(self.cpu_classification) + 1)])
            for req in list_reqs_allocated:
                class_index = int(self.cpu_classification[req])
                amount_class[class_index] = amount_class[class_index] + 1

            perc = np.divide(amount_class, sum(amount_class)) if sum(amount_class) > 0 else 0
            perc_amount_class_per_node_cpu.append(perc)

        cpu_class_prop = np.array([
            np.bincount(self.cpu_classification)[i] / sum(np.bincount(self.cpu_classification))
            for i in range(max(self.cpu_classification) + 1)
        ])

        memory_class_prop = np.array([
            np.bincount(self.memory_classification)[i] / sum(np.bincount(self.memory_classification))
            for i in range(max(self.memory_classification) + 1)
        ])

        mem_class_dist = np.mean([np.sqrt(sum((s - memory_class_prop) ** 2)) for s in perc_amount_class_per_node_mem])
        cpu_class_dist = np.mean([np.sqrt(sum((s - cpu_class_prop) ** 2)) for s in perc_amount_class_per_node_cpu])
        mem_cons_kpi = sum(
            -prop_memory_consumption_by_nodes * [np.log(i) if i > 0 else 0 for i in prop_memory_consumption_by_nodes]
        ) / np.log(self.n_nodes)
        cpu_cons_kpi = sum(
            -prop_cpu_consumption_by_nodes * [np.log(i) if i > 0 else 0 for i in prop_cpu_consumption_by_nodes]
        ) / np.log(self.n_nodes)

        return \
            (1 / mem_cons_kpi) +\
            (1 / cpu_cons_kpi) + \
            mem_class_dist + \
            cpu_class_dist
