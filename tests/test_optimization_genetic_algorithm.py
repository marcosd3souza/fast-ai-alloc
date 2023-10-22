import unittest

import pandas as pd

from src.optimization.genetic_algorithm import GeneticAlgorithm


class TestGeneticAlgorithm(unittest.TestCase):
    def setUp(self) -> None:
        self.dataframe = pd.DataFrame(
            [
                ['node1', 'pod1', 0.1, 0.2, 0, 0],
                ['node1', 'pod2', 0.2, 0.3, 0, 0],
                ['node1', 'pod3', 0.3, 0.4, 0, 0],
                ['node1', 'pod4', 0.4, 0.5, 0, 0],
                ['node1', 'pod5', 0.5, 0.6, 0, 0],
                ['node1', 'pod6', 0.6, 0.7, 0, 0],
                ['node1', 'pod7', 0.7, 0.8, 0, 0],
                ['node1', 'pod8', 0.8, 0.9, 0, 0],
                ['node1', 'pod9', 0.9, 0.1, 0, 0],
                ['node1', 'pod10', 0.1, 0.2, 0, 0],
                ['node1', 'pod11', 0.2, 0.3, 0, 0],
                ['node1', 'pod12', 0.3, 0.4, 0, 0],
                ['node1', 'pod13', 0.4, 0.5, 0, 0],
                ['node1', 'pod14', 0.5, 0.6, 0, 0],
                ['node1', 'pod15', 0.6, 0.7, 0, 0],
                ['node1', 'pod16', 0.7, 0.8, 0, 0],
                ['node2', 'pod17', 0.8, 0.9, 0, 0],
                ['node2', 'pod18', 0.9, 0.1, 0, 0],
                ['node2', 'pod19', 0.1, 0.2, 0, 0],
            ], columns=['node_name', 'pod_name', 'cpu_usage', 'memory_usage', 'cpu_class', 'memory_class'])

    def test_init(self):
        with self.assertRaises(ValueError):
            GeneticAlgorithm(pd.DataFrame())

    def test_init2(self):
        with self.assertRaises(ValueError):
            GeneticAlgorithm(pd.DataFrame(columns=['node_name', 'pod_name', 'cpu_usage', 'memory_usage', 'cpu_class']))

    def test_init3(self):
        with self.assertRaises(ValueError):
            GeneticAlgorithm(pd.DataFrame(columns=[
                'node_name', 'pod_name', 'cpu_usage', 'memory_usage', 'cpu_class', 'memory_class', 'extra_column']))

    def test_initial_population(self):
        ga = GeneticAlgorithm(self.dataframe)
        self.assertEqual(len(ga.init_population), 300)

    def test_optimize(self):
        ga = GeneticAlgorithm(self.dataframe)
        solution = ga.optimize(n_generations=1)
        self.assertEqual(len(solution), self.dataframe.shape[0])
