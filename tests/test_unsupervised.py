import unittest

import pandas as pd

from src.classification.unsupervised import SOM


class TestSOM(unittest.TestCase):
    def test_init(self):
        # create an instance of the SOM class
        som = SOM(pd.DataFrame(
            data=[['node_1', 'pod_1', 10, 256]],
            columns=["node_name", "pod_name", "cpu_usage", "memory_usage"]
        ))
        # check if the SOM class has the correct attributes
        assert hasattr(som, 'model_memory')
        assert hasattr(som, 'model_cpu')
        assert hasattr(som, 'dataframe')
        assert hasattr(som, 'width')
        assert hasattr(som, 'height')
        assert hasattr(som, 'random_state')
        # check if the SOM class has the correct attribute types
        assert isinstance(som.model_memory, type(None))
        assert isinstance(som.model_cpu, type(None))
        assert isinstance(som.dataframe, pd.DataFrame)
        assert isinstance(som.width, int)
        assert isinstance(som.height, int)
        assert isinstance(som.random_state, int)
        # check if the SOM class has the correct default attribute values
        assert som.model_memory is None
        assert som.model_cpu is None
        assert som.dataframe.shape[0] == 1
        assert som.dataframe.shape[1] == 4
        assert som.width == 4
        assert som.height == 4
        assert som.random_state == 0

    def test_init_with_empty_dataframe(self):
        # create an instance of the SOM class with an empty dataframe
        with self.assertRaises(ValueError):
            SOM(pd.DataFrame())

    def test_init_with_dataframe_without_memory_usage_column(self):
        # create an instance of the SOM class with a dataframe without a memory_usage column
        with self.assertRaises(ValueError):
            SOM(pd.DataFrame(
                data=[['node_1', 'pod_1', 10]],
                columns=["node_name", "pod_name", "cpu_usage"]
            ))

    def test_init_with_dataframe_without_cpu_usage_column(self):
        # create an instance of the SOM class with a dataframe without a cpu_usage column
        with self.assertRaises(ValueError):
            SOM(pd.DataFrame(
                data=[['node_1', 'pod_1', 256]],
                columns=["node_name", "pod_name", "memory_usage"]
            ))

    def test_init_with_dataframe_without_node_name_column(self):
        # create an instance of the SOM class with a dataframe without a node_name column
        with self.assertRaises(ValueError):
            SOM(pd.DataFrame(
                data=[['pod_1', 10, 256]],
                columns=["pod_name", "cpu_usage", "memory_usage"]
            ))

    def test_init_with_dataframe_without_pod_name_column(self):
        # create an instance of the SOM class with a dataframe without a pod_name column
        with self.assertRaises(ValueError):
            SOM(pd.DataFrame(
                data=[['node_1', 10, 256]],
                columns=["node_name", "cpu_usage", "memory_usage"]
            ))

    def test_fit_memory(self):
        # create an instance of the SOM class
        som = SOM(pd.DataFrame(
            data=[
                ['node_1', 'pod_1', 10, 256],
                ['node_2', 'pod_2', 20, 512],
                ['node_3', 'pod_3', 30, 1024],
            ],
            columns=["node_name", "pod_name", "cpu_usage", "memory_usage"]
        ))
        # fit the SOM model for memory usage
        som.fit_memory()
        # check if the SOM model for memory usage has been fitted
        assert isinstance(som.model_memory, type(None)) is False
        # check if the SOM model for memory usage has the correct attribute values
        assert som.model_memory.m == 4
        assert som.model_memory.n == 4
        assert som.model_memory.dim == 1
        assert som.model_memory.random_state == 0

    def test_fit_cpu(self):
        # create an instance of the SOM class
        som = SOM(pd.DataFrame(
            data=[
                ['node_1', 'pod_1', 10, 256],
                ['node_2', 'pod_2', 20, 512],
                ['node_3', 'pod_3', 30, 1024],
            ],
            columns=["node_name", "pod_name", "cpu_usage", "memory_usage"]
        ))
        # fit the SOM model for cpu usage
        som.fit_cpu()
        # check if the SOM model for cpu usage has been fitted
        assert isinstance(som.model_cpu, type(None)) is False
        # check if the SOM model for cpu usage has the correct attribute values
        assert som.model_cpu.m == 4
        assert som.model_cpu.n == 4
        assert som.model_cpu.dim == 1
        assert som.model_cpu.random_state == 0

    def test_predict_memory(self):
        # create an instance of the SOM class
        som = SOM(pd.DataFrame(
            data=[
                ['node_1', 'pod_1', 10, 256],
                ['node_2', 'pod_2', 20, 512],
                ['node_3', 'pod_3', 30, 1024],
            ],
            columns=["node_name", "pod_name", "cpu_usage", "memory_usage"]
        ))
        # fit the SOM model for memory usage
        som.fit_memory()
        # predict memory usage
        memory_class = som.predict_memory(pd.DataFrame(
            data=[
                [1],
            ],
            columns=["memory_usage"]
        ).to_numpy())
        # check if the memory usage has been predicted correctly
        assert len(memory_class) == 1
        assert 0 <= memory_class[0] <= 15

    def test_predict_cpu(self):
        # create an instance of the SOM class
        som = SOM(pd.DataFrame(
            data=[
                ['node_1', 'pod_1', 10, 256],
                ['node_2', 'pod_2', 20, 512],
                ['node_3', 'pod_3', 30, 1024],
            ],
            columns=["node_name", "pod_name", "cpu_usage", "memory_usage"]
        ))
        # fit the SOM model for cpu usage
        som.fit_cpu()
        # predict cpu usage
        cpu_class = som.predict_cpu(pd.DataFrame(
            data=[
                [1],
            ],
            columns=["cpu_usage"]
        ).to_numpy())
        # check if the cpu usage has been predicted correctly
        assert len(cpu_class) == 1
        assert 0 <= cpu_class[0] <= 15

