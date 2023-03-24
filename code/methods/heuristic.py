import numpy as np

from methods.base import BaseOptimizer


class HeuristicOptimizer(BaseOptimizer):

    def __init__(self,
                 n_requests,
                 n_nodes,
                 memory_consumption,
                 cpu_consumption,
                 memory_classification,
                 cpu_classification):
        self.n_requests = n_requests
        self.n_nodes = n_nodes
        super().__init__(
            self.n_requests,
            self.n_nodes,
            memory_consumption,
            cpu_consumption,
            memory_classification,
            cpu_classification
        )

        self.memory_consumption = memory_consumption
        self.memory_classification = memory_classification

        self.cpu_consumption = cpu_consumption
        self.cpu_classification = cpu_classification

    def optimize(self):
        solution = np.zeros((self.n_requests, self.n_nodes))

        node_memory_consumption = [0] * self.n_nodes
        node_memory_labels = [[]] * self.n_nodes

        node_cpu_consumption = [0] * self.n_nodes
        node_cpu_labels = [[]] * self.n_nodes

        for pod_idx in range(0, self.n_requests):
            request_memory_label = self.memory_classification[pod_idx]
            request_memory_consumption = self.memory_consumption[pod_idx][0]

            node_memory_labels_std = [
                np.std(np.bincount(np.array(np.concatenate([j, [request_memory_label]]), dtype=int)))
                for j in node_memory_labels
            ]

            request_cpu_label = self.cpu_classification[pod_idx]
            request_cpu_consumption = self.cpu_consumption[pod_idx][0]

            node_cpu_labels_std = [
                np.std(np.bincount(np.array(np.concatenate([j, [request_cpu_label]]), dtype=int)))
                for j in node_cpu_labels
            ]

            # heuristic
            heuristic = sum(node_memory_labels_std, node_cpu_labels_std) + \
                        (node_memory_consumption + request_memory_consumption) + \
                        (node_cpu_consumption + request_cpu_consumption)

            best_node = np.argsort(heuristic)[0]

            solution[pod_idx, best_node] = 1

            node_memory_labels[best_node] = np.concatenate([node_memory_labels[best_node], [request_memory_label]])
            node_memory_consumption[best_node] += request_memory_consumption

            node_cpu_labels[best_node] = np.concatenate([node_cpu_labels[best_node], [request_cpu_label]])
            node_cpu_consumption[best_node] += request_cpu_consumption

        print(f'heuristic fitness: {self.calc_fitness(solution, is_suggestion_in_mat_format=True)}')
        return solution
