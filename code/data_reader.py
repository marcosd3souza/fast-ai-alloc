import numpy as np
from sklearn_som.som import SOM

from data_sampling import DataSampling


class DataReader:

    def __init__(self):
        self.root_path = './data/input/'

        self.BINARY_FILE_EXTENSION = '.npz'
        self.CSV_FILE_EXTENSION = '.csv'
        self.COLON_SEP = ';'
        self.SLASH = '/'

    def read(self, data_path):
        path = f'{self.root_path}{data_path}DAS-2_1min_samples{self.BINARY_FILE_EXTENSION}'
        samples = np.load(path, allow_pickle=True)['arr_0']

        X_memory_train, X_cpu_train, test_samples = DataSampling(samples, train_size=1000).get_train_test_samples()

        model_memory = SOM(m=4, n=4, dim=1, random_state=0)
        model_memory.fit(X_memory_train)

        model_cpu = SOM(m=4, n=4, dim=1, random_state=0)
        model_cpu.fit(X_cpu_train)

        return test_samples, model_memory, model_cpu
