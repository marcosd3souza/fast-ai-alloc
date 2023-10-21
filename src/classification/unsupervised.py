import numpy as np
import pandas as pd
from sklearn_som.som import SOM as sklearn_som


class SOM:
    def __init__(self, dataframe: pd.DataFrame, width: int = 4, height: int = 4, random_state: int = 0):
        self.model_memory = None
        self.model_cpu = None
        self.dataframe = dataframe
        self.width = width
        self.height = height
        self.random_state = random_state
        if self.dataframe.empty:
            raise ValueError('Dataframe cannot be empty')
        if 'memory_usage' not in self.dataframe.columns:
            raise ValueError('Dataframe must have a column named "memory"')
        if 'cpu_usage' not in self.dataframe.columns:
            raise ValueError('Dataframe must have a column named "cpu"')
        if 'node_name' not in self.dataframe.columns:
            raise ValueError('Dataframe must have a column named "node_name"')
        if 'pod_name' not in self.dataframe.columns:
            raise ValueError('Dataframe must have a column named "pod_name"')

    def fit_memory(self):
        self.model_memory = sklearn_som(m=self.width, n=self.height, dim=1, random_state=self.random_state)
        self.model_memory.fit(self.dataframe['memory_usage'].values.reshape(-1, 1))
        print('Model memory fitted')

    def fit_cpu(self):
        self.model_cpu = sklearn_som(m=self.width, n=self.height, dim=1, random_state=self.random_state)
        self.model_cpu.fit(self.dataframe['cpu_usage'].values.reshape(-1, 1))

    def predict_memory(self, memory_data: np.array) -> np.array:
        return self.model_memory.predict(memory_data)

    def predict_cpu(self, cpu_data: np.array) -> np.array:
        return self.model_cpu.predict(cpu_data)
