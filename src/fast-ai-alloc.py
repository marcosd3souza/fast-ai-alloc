import pandas as pd
from optimization.algorithms import OptimizationAlgorithms
from classification.algorithms import ClassificationAlgorithms


class FastAIAllocator:
    def __init__(self,
                 dataframe: pd.DataFrame,
                 optimization_algorithm: str = OptimizationAlgorithms.GENETIC_ALGORITHM,
                 classification_algorithm: str = ClassificationAlgorithms.SELF_ORGANIZING_MAPS,
                 **kwargs):
        self.dataframe = dataframe
        self.optimization_algorithm = optimization_algorithm
        self.classification_algorithm = classification_algorithm

        if self.optimization_algorithm not in [
            OptimizationAlgorithms.GENETIC_ALGORITHM,
            OptimizationAlgorithms.HEURISTIC,
            OptimizationAlgorithms.PSO,
        ]:
            raise ValueError('Invalid optimization algorithm')
        if self.classification_algorithm not in [
            ClassificationAlgorithms.SELF_ORGANIZING_MAPS,
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

    def __classify_pods(self, **kwargs):
        if self.classification_algorithm == ClassificationAlgorithms.SELF_ORGANIZING_MAPS:
            from classification.unsupervised import SOM
            som = SOM(self.dataframe, **kwargs)
            som.fit_memory()
            som.fit_cpu()

            self.dataframe['memory_class'] = som.model_memory.predict(self.dataframe['memory'].values.reshape(-1, 1))
            self.dataframe['cpu_class'] = som.model_cpu.predict(self.dataframe['cpu'].values.reshape(-1, 1))

    def allocate(self):
        if self.optimization_algorithm == OptimizationAlgorithms.GENETIC_ALGORITHM:
            from optimization.genetic_algorithm import GeneticAlgorithm
            return GeneticAlgorithm(self.dataframe).allocate()
        elif self.optimization_algorithm == OptimizationAlgorithms.PSO:
            from optimization.pso import PSO
            return PSO(self.dataframe).allocate()
        elif self.optimization_algorithm == OptimizationAlgorithms.HEURISTIC:
            from optimization.heuristic import Heuristic
            return Heuristic(self.dataframe).allocate()
