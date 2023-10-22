import unittest

import numpy as np
import pandas as pd

from src.optimization.base import BaseOptimizer


class TestBase(unittest.TestCase):
    def test_init(self):
        # create an instance of the BaseOptimizer class
        base = BaseOptimizer(pd.DataFrame(
            data=[['node_1', 'pod_1', 10, 256, 0, 0]],
            columns=["node_name", "pod_name", "cpu_usage", "memory_usage", "cpu_class", "memory_class"]
        ))
        # check if the BaseOptimizer class has the correct attributes
        assert hasattr(base, 'dataframe')
        assert hasattr(base, 'nodes')
        assert hasattr(base, 'pods')
        assert hasattr(base, 'amount_nodes')
        assert hasattr(base, 'amount_pods')
        assert hasattr(base, 'pods_memory')
        assert hasattr(base, 'pods_cpu')
        # check if the BaseOptimizer class has the correct attribute types
        assert isinstance(base.dataframe, pd.DataFrame)
        assert isinstance(base.nodes, np.ndarray)
        assert isinstance(base.pods, np.ndarray)
        assert isinstance(base.amount_nodes, int)
        assert isinstance(base.amount_pods, int)
        assert isinstance(base.pods_memory, np.ndarray)
        assert isinstance(base.pods_cpu, np.ndarray)
        # check if the BaseOptimizer class has the correct default attribute values
        assert base.dataframe.shape[0] == 1
        assert base.dataframe.shape[1] == 6
        assert base.nodes.shape[0] == 1
        assert base.pods.shape[0] == 1
        assert base.amount_nodes == 1
        assert base.amount_pods == 1
        assert base.pods_memory.shape[0] == 1
        assert base.pods_cpu.shape[0] == 1

    def test_init_with_empty_dataframe(self):
        # create an instance of the BaseOptimizer class with an empty dataframe
        with self.assertRaises(ValueError):
            BaseOptimizer(pd.DataFrame())

    def test_init_with_dataframe_without_memory_usage_column(self):
        # create an instance of the BaseOptimizer class with a dataframe without a memory_usage column
        with self.assertRaises(ValueError):
            BaseOptimizer(pd.DataFrame(
                data=[['node_1', 'pod_1', 10]],
                columns=["node_name", "pod_name", "cpu_usage"]
            ))

    def test_init_with_dataframe_without_cpu_usage_column(self):
        # create an instance of the BaseOptimizer class with a dataframe without a cpu_usage column
        with self.assertRaises(ValueError):
            BaseOptimizer(pd.DataFrame(
                data=[['node_1', 'pod_1', 256]],
                columns=["node_name", "pod_name", "memory_usage"]
            ))

    def test_init_with_dataframe_without_node_name_column(self):
        # create an instance of the BaseOptimizer class with a dataframe without a node_name column
        with self.assertRaises(ValueError):
            BaseOptimizer(pd.DataFrame(
                data=[['pod_1', 10, 256]],
                columns=["pod_name", "cpu_usage", "memory_usage"]
            ))

    def test_init_with_dataframe_without_pod_name_column(self):
        # create an instance of the BaseOptimizer class with a dataframe without a pod_name column
        with self.assertRaises(ValueError):
            BaseOptimizer(pd.DataFrame(
                data=[['node_1', 10, 256]],
                columns=["node_name", "cpu_usage", "memory_usage"]
            ))

    def test_init_with_dataframe_with_invalid_columns(self):
        # create an instance of the BaseOptimizer class with a dataframe with invalid columns
        with self.assertRaises(ValueError):
            BaseOptimizer(pd.DataFrame(
                data=[['node_1', 'pod_1', 10, 256, 0, 0]],
                columns=["node_name", "pod_name", "cpu_usage", "memory_usage", "cpu_class", "memory_class", "invalid"]
            ))

    def test_calc_fitness_multi(self):
        # create an instance of the BaseOptimizer class
        base = BaseOptimizer(pd.DataFrame(
            data=[
                ['node_1', 'pod_1', 3, 67, 0, 0],
                ['node_2', 'pod_2', 10, 1400, 1, 1],
                ['node_1', 'pod_3', 6, 857, 1, 1],
                ['node_1', 'pod_4', 1, 280, 0, 0],
            ],
            columns=["node_name", "pod_name", "cpu_usage", "memory_usage", "cpu_class", "memory_class"]
        ))
        # check if the calc_fitness method returns the correct value
        fitness = base.calc_fitness(np.array([
            [0, 1],
            [0, 1],
            [0, 1],
            [1, 0],
        ]))
        assert 6 < fitness < 7

        # create an instance of the BaseOptimizer class
        base = BaseOptimizer(pd.DataFrame(
            data=[
                ['node_1', 'pod_1', 10, 100, 0, 0],
                ['node_2', 'pod_2', 10, 100, 0, 0],
                ['node_1', 'pod_3', 10, 100, 0, 0],
                ['node_1', 'pod_4', 10, 100, 0, 0],
            ],
            columns=["node_name", "pod_name", "cpu_usage", "memory_usage", "cpu_class", "memory_class"]
        ))
        # check if the calc_fitness method returns the correct value
        fitness = base.calc_fitness(np.array([
            [0, 1],
            [0, 1],
            [1, 0],
            [1, 0],
        ]))
        assert fitness == 2

    def test_calc_fitness_memory(self):
        # create an instance of the BaseOptimizer class
        base = BaseOptimizer(pd.DataFrame(
            data=[
                ['node_1', 'pod_1', 3, 67, 0, 0],
                ['node_2', 'pod_2', 10, 1400, 1, 1],
                ['node_1', 'pod_3', 6, 857, 1, 1],
                ['node_1', 'pod_4', 1, 280, 0, 0],
            ],
            columns=["node_name", "pod_name", "cpu_usage", "memory_usage", "cpu_class", "memory_class"]
        ))
        # check if the calc_fitness method returns the correct value
        fitness = base.calc_fitness(np.array([
            [0, 1],
            [0, 1],
            [0, 1],
            [1, 0],
        ]), metric=BaseOptimizer.METRIC_MEMORY)
        assert 2 < fitness < 3

        # create an instance of the BaseOptimizer class
        base = BaseOptimizer(pd.DataFrame(
            data=[
                ['node_1', 'pod_1', 10, 100, 0, 0],
                ['node_2', 'pod_2', 10, 100, 0, 0],
                ['node_1', 'pod_3', 10, 100, 0, 0],
                ['node_1', 'pod_4', 10, 100, 0, 0],
            ],
            columns=["node_name", "pod_name", "cpu_usage", "memory_usage", "cpu_class", "memory_class"]
        ))
        # check if the calc_fitness method returns the correct value
        fitness = base.calc_fitness(np.array([
            [0, 1],
            [0, 1],
            [1, 0],
            [1, 0],
        ]), metric=BaseOptimizer.METRIC_MEMORY)
        assert fitness == 1

    def test_calc_fitness_cpu(self):
        # create an instance of the BaseOptimizer class
        base = BaseOptimizer(pd.DataFrame(
            data=[
                ['node_1', 'pod_1', 3, 67, 0, 0],
                ['node_2', 'pod_2', 10, 1400, 1, 1],
                ['node_1', 'pod_3', 6, 857, 1, 1],
                ['node_1', 'pod_4', 1, 280, 0, 0],
            ],
            columns=["node_name", "pod_name", "cpu_usage", "memory_usage", "cpu_class", "memory_class"]
        ))
        # check if the calc_fitness method returns the correct value
        fitness = base.calc_fitness(np.array([
            [0, 1],
            [0, 1],
            [0, 1],
            [1, 0],
        ]), metric=BaseOptimizer.METRIC_CPU)
        assert 3 < fitness < 4

        # create an instance of the BaseOptimizer class
        base = BaseOptimizer(pd.DataFrame(
            data=[
                ['node_1', 'pod_1', 10, 100, 0, 0],
                ['node_2', 'pod_2', 10, 100, 0, 0],
                ['node_1', 'pod_3', 10, 100, 0, 0],
                ['node_1', 'pod_4', 10, 100, 0, 0],
            ],
            columns=["node_name", "pod_name", "cpu_usage", "memory_usage", "cpu_class", "memory_class"]
        ))
        # check if the calc_fitness method returns the correct value
        fitness = base.calc_fitness(np.array([
            [0, 1],
            [0, 1],
            [1, 0],
            [1, 0],
        ]), metric=BaseOptimizer.METRIC_CPU)
        assert fitness == 1

    def test_optimization_to_dataframe(self):
        initial_allocation = pd.DataFrame(
            data=[
                ['node_1', 'pod_1', 3, 67, 0, 0],
                ['node_2', 'pod_2', 10, 1400, 1, 1],
                ['node_1', 'pod_3', 6, 857, 1, 1],
                ['node_1', 'pod_4', 1, 280, 0, 0],
            ],
            columns=["node_name", "pod_name", "cpu_usage", "memory_usage", "cpu_class", "memory_class"]
        )
        base = BaseOptimizer(initial_allocation)
        allocation_suggestion = np.array([
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
        ])
        optimized_dataframe = base.optimization_to_dataframe(allocation_suggestion)
        assert optimized_dataframe.shape[0] == 4
        assert optimized_dataframe.shape[1] == 6
        assert optimized_dataframe['node_name'].tolist() == ['node_1', 'node_1', 'node_2', 'node_2']
        assert optimized_dataframe['pod_name'].tolist() == ['pod_1', 'pod_2', 'pod_3', 'pod_4']
        assert optimized_dataframe['cpu_usage'].tolist() == [3, 10, 6, 1]
        assert optimized_dataframe['memory_usage'].tolist() == [67, 1400, 857, 280]
        assert optimized_dataframe['cpu_class'].tolist() == [0, 1, 1, 0]
        assert optimized_dataframe['memory_class'].tolist() == [0, 1, 1, 0]
