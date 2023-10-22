import logging

import pandas as pd

from classification.algorithms import ClassificationAlgorithm
from optimization.algorithms import OptimizationAlgorithm

logging.basicConfig(level=logging.INFO)


class FastAIAllocator:
    def __init__(self,
                 dataframe: pd.DataFrame,
                 optimization_algorithm: OptimizationAlgorithm = OptimizationAlgorithm.GENETIC_ALGORITHM,
                 classification_algorithm: ClassificationAlgorithm = ClassificationAlgorithm.SELF_ORGANIZING_MAPS,
                 **kwargs):
        self.dataframe = dataframe
        self.optimization_algorithm = optimization_algorithm
        self.classification_algorithm = classification_algorithm

        if self.optimization_algorithm not in [
            OptimizationAlgorithm.GENETIC_ALGORITHM,
            OptimizationAlgorithm.HEURISTIC,
            OptimizationAlgorithm.PSO,
        ]:
            raise ValueError('Invalid optimization algorithm')
        if self.classification_algorithm not in [
            ClassificationAlgorithm.SELF_ORGANIZING_MAPS,
        ]:
            raise ValueError('Invalid classification algorithm')
        if 'memory_usage' not in self.dataframe.columns:
            raise ValueError('Dataframe must have a column named "memory"')
        if 'cpu_usage' not in self.dataframe.columns:
            raise ValueError('Dataframe must have a column named "cpu"')
        if 'node_name' not in self.dataframe.columns:
            raise ValueError('Dataframe must have a column named "node_name"')
        if 'pod_name' not in self.dataframe.columns:
            raise ValueError('Dataframe must have a column named "pod_name"')

        self.__classify_pods(**kwargs)
        self.optimized_dataframe = pd.DataFrame(columns=[
            'node_name',
            'pod_name',
            'cpu_usage',
            'memory_usage',
            'cpu_class',
            'memory_class'
        ])

    def __classify_pods(self, **kwargs):
        if self.classification_algorithm == ClassificationAlgorithm.SELF_ORGANIZING_MAPS:
            from classification.unsupervised import SOM
            som = SOM(self.dataframe, **kwargs)
            som.fit_memory()
            som.fit_cpu()

            self.dataframe['cpu_class'] = som.model_cpu.predict(self.dataframe['cpu_usage'].values.reshape(-1, 1))
            self.dataframe['memory_class'] = som.model_memory.predict(self.dataframe['memory_usage'].values.reshape(-1, 1))

    def optimize_allocation(self, **kwargs) -> pd.DataFrame:
        if self.optimization_algorithm == OptimizationAlgorithm.GENETIC_ALGORITHM:
            from optimization.genetic_algorithm import GeneticAlgorithm
            ga = GeneticAlgorithm(self.dataframe)
            self.optimized_dataframe = ga.optimization_to_dataframe(ga.optimize(**kwargs))
        elif self.optimization_algorithm == OptimizationAlgorithm.PSO:
            from optimization.pso import PSO
            return PSO(self.dataframe).allocate()
        elif self.optimization_algorithm == OptimizationAlgorithm.HEURISTIC:
            from optimization.heuristic import Heuristic
            return Heuristic(self.dataframe).allocate()
        return self.optimized_dataframe


initial_allocation = pd.read_csv('../data_sample/kubernetes_sample_data.csv')
allocator = FastAIAllocator(
    dataframe=initial_allocation,
    optimization_algorithm=OptimizationAlgorithm.GENETIC_ALGORITHM,
    classification_algorithm=ClassificationAlgorithm.SELF_ORGANIZING_MAPS
)
optimized = allocator.optimize_allocation(n_generations=30)
optimized.to_csv('../data_sample/kubernetes_sample_data_optimized_genetic_algorithm.csv', index=False)
