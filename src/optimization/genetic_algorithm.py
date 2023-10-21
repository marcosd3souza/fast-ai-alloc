import pandas as pd


class GeneticAlgorithm:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.nodes = self.dataframe['node_name'].unique().sort()
        self.pods = self.dataframe['pod_name'].unique().sort()
        self.amount_nodes = len(self.nodes)
        self.amount_pods = len(self.pods)
        self.nodes_total_memory = self.dataframe.groupby('node_name')['memory'].sum()
        self.nodes_total_cpu = self.dataframe.groupby('node_name')['cpu'].sum()

