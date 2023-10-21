import unittest

import pandas as pd

from src.utils.data_generator import DataGenerator


class TestDataGenerator(unittest.TestCase):
    def test_generate(self):
        # create an instance of the data generator class
        data_generator = DataGenerator()
        # call the generate method of the data generator class
        data = data_generator.generate()
        # check if the data is a pandas dataframe
        assert isinstance(data, pd.DataFrame)
        # check if the data has the correct number of rows
        assert data.shape[0] == 1000
        # check if the data has the correct number of columns
        assert data.shape[1] == 4
        # check if the data has the correct column names
        assert list(data.columns) == ["node", "pod", "cpu_usage", "memory_usage"]
        # check if the data has the correct data types
        assert data.dtypes.tolist() == [object, object, float, float]
        # check if the data has the correct number of unique nodes
        assert len(data["node"].unique()) == 10
        # check if the data has the correct number of unique pods
        assert len(data["pod"].unique()) == 1000
        # check if the data has the correct minimum cpu usage value
        assert data["cpu_usage"].min() >= 1
        # check if the data has the correct maximum cpu usage value
        assert data["cpu_usage"].max() <= 100
        # check if the data has the correct minimum memory usage value
        assert data["memory_usage"].min() >= 1
        # check if the data has the correct maximum memory usage value
        assert data["memory_usage"].max() <= 6000
        # check if the data has the correct number of unique node-pod combinations
        assert len(data.groupby(["node", "pod"]).size()) == 1000
