import pandas as pd
from ortools.linear_solver import pywraplp
import numpy as np

from methods.base import BaseOptimizer


class IntegerLinearSolver(BaseOptimizer):

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

        self.n_labels_memory = len(np.bincount(memory_classification))
        self.n_labels_cpu = len(np.bincount(cpu_classification))

        self.memory_classification = memory_classification
        self.cpu_classification = cpu_classification

        self.memory_consumption = memory_consumption
        self.cpu_consumption = cpu_consumption

        self.data = {}
        self.solver = pywraplp.Solver.CreateSolver('CBC')
        self.coefficients = {}

        self.memory_class_count = np.bincount(memory_classification)
        self.upper_memory_limit = None

        self.cpu_class_count = np.bincount(cpu_classification)
        self.upper_cpu_limit = None

    def optimize(self, solver_max_timeout_ms):
        solver_max_timeout_ms = solver_max_timeout_ms

        self._data_workload()
        self._variables_workload()
        # Minimize objective function
        self.solver.Minimize(self.upper_memory_limit + self.upper_cpu_limit)

        # objective.SetMinimization()
        self.solver.EnableOutput()
        self.solver.SetTimeLimit(solver_max_timeout_ms)
        _ = self.solver.Solve()

        # Objective function
        objective = self.solver.Objective()
        print('Total Packed Value:', objective.Value())

        solution_values = []
        for i in self.data['nodes']:
            row = []
            for j in range(self.n_requests):
                row.append(self.coefficients[j, i].solution_value())
            solution_values.append(row)

        solution = np.transpose(solution_values)

        return solution

    def _data_workload(self):

        # PREPARING DATA TO OPTIMIZER
        dummies_memory = np.zeros((self.n_requests, self.n_labels_memory), dtype=int)
        dummies_memory = pd.DataFrame(dummies_memory, columns=range(self.n_labels_memory))

        dummies_cpu = np.zeros((self.n_requests, self.n_labels_cpu), dtype=int)
        dummies_cpu = pd.DataFrame(dummies_cpu, columns=range(self.n_labels_cpu))

        for i in range(0, self.n_requests):
            dummies_memory.iloc[i, int(self.memory_classification[i])] = 1

        for i in range(0, self.n_requests):
            dummies_cpu.iloc[i, int(self.cpu_classification[i])] = 1

        # value of memory for each of the requests
        self.data['request_memory'] = self.memory_consumption
        # value of cpu for each of the requests
        self.data['request_cpu'] = self.cpu_consumption
        # group memory labels
        self.data['groups_memory'] = [i for i in range(self.n_labels_memory)]
        # group cpu labels
        self.data['groups_cpu'] = [i for i in range(self.n_labels_cpu)]

        # memory group per request
        self.data['group_memory_request'] = dummies_memory.values
        # cpu group per request
        self.data['group_cpu_request'] = dummies_cpu.values

        # requests labels for optimization (0, ..., n)
        self.data['requests'] = range(self.n_requests)
        self.data['num_requests'] = self.n_requests

        # node labels for optimization (0,..., m)
        self.data['nodes'] = range(0, self.n_nodes)

    def _variables_workload(self):

        # PREPARING VARIABLE TO OPTIMIZER
        for i in self.data['requests']:
            for j in self.data['nodes']:
                self.coefficients[(i, j)] = self.solver.IntVar(0, 1, 'x_%i_%i' % (i, j))

        self.upper_memory_limit = self.solver.NumVar(0.0, self.solver.infinity(), 'W')
        self.upper_cpu_limit = self.solver.NumVar(0.0, self.solver.infinity(), 'Y')

        z_memory_vector = self.memory_class_count / self.n_nodes
        z_memory_vector = np.ceil(z_memory_vector)

        z_cpu_vector = self.cpu_class_count / self.n_nodes
        z_cpu_vector = np.ceil(z_cpu_vector)

        z_memory = {}
        for i in self.data['groups_memory']:
            z_memory[i] = self.solver.NumVar(0.0, z_memory_vector[i], 'z_memory_%i' % i)

        z_cpu = {}
        for i in self.data['groups_cpu']:
            z_cpu[i] = self.solver.NumVar(0.0, z_cpu_vector[i], 'z_cpu_%i' % i)

        # Constraint uniform memory distribution
        for i in self.data['nodes']:
            self.solver.Add(

                sum(self.data['request_memory'][j] * self.coefficients[j, i] for j in self.data['requests'])[0]
                <= self.upper_memory_limit
            )

        # Constraint uniform cpu distribution
        for i in self.data['nodes']:
            self.solver.Add(

                sum(self.data['request_cpu'][j] * self.coefficients[j, i] for j in self.data['requests'])[0]
                <= self.upper_cpu_limit
            )

        # Constraint one request in a node
        for i in self.data['requests']:
            self.solver.Add(sum(self.coefficients[i, j] for j in self.data['nodes']) == 1)

        # Constraint classification distribution
        # TO DO - add offset in function memory usage
        for k in self.data['groups_memory']:
            for n in self.data['nodes']:
                self.solver.Add(
                    sum(self.data['group_memory_request'][i][k] * self.coefficients[i, n] for i in self.data['requests']) <= z_memory[k])

        for k in self.data['groups_cpu']:
            for n in self.data['nodes']:
                self.solver.Add(
                    sum(self.data['group_cpu_request'][i][k] * self.coefficients[i, n] for i in self.data['requests']) <= z_cpu[k])
