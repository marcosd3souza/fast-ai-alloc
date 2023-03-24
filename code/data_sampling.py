import pandas as pd
import numpy as np
import random


class DataSampling:
    def __init__(self, samples, train_size):
        self.samples = samples
        self.train_size = train_size

        self.n_samples = len(samples)

    def get_train_test_samples(self):
        train_ids, test_ids = self._sampling_idx()

        X_memory_train = [self.samples[i].UsedMemory.values for i in train_ids]
        X_memory_train = pd.DataFrame([memory for sublist in X_memory_train for memory in sublist]).values

        X_cpu_train = [self.samples[i].UsedCPUTime.values for i in train_ids]
        X_cpu_train = pd.DataFrame([cpu for sublist in X_cpu_train for cpu in sublist]).values

        return X_memory_train, X_cpu_train, self.samples[test_ids]

    def _sampling_idx(self):
        train_ids = random.sample(range(0, self.n_samples), self.train_size)
        train_ids = list(np.sort(train_ids))

        s = set(train_ids)
        x = (i for i in range(0, self.n_samples) if i not in s)
        test_ids = list(x)

        return train_ids, test_ids
