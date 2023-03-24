from data_reader import DataReader
from enum import Enum
from methods.genetic import GeneticAlgorithm
from methods.heuristic import HeuristicOptimizer
from methods.integer_linear_solver import IntegerLinearSolver
from methods.pso import ParticleSwarmAlgorithm
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from multiprocessing import Process


class Sample(Enum):
    DAS_2 = 'DAS-2/'
    MON = 'MON_2022-04-18-04-25-32'
    TUE = 'TUE_2022-04-19-11-30-24'
    WED = 'WED_2022-04-20-11-32-46'
    THU = 'THU_2022-04-21-05-37-53'
    FRI = 'FRI_2022-04-22-11-33-35'
    SAT = 'SAT_2022-04-23-11-36-09'
    SUN = 'SUN_2022-04-17-11-28-02'


def _consumption_chart(df_allocation, classification, consumption, method, metric):
    cpm = df_allocation.copy().mul(consumption)
    cpm['label'] = classification.astype('str')

    sum_by_group = cpm.groupby(by='label').sum()
    groups = sum_by_group.index.values
    amount_class_per_node = sum_by_group.values.T
    nodes = []
    classes = []
    count = []
    group_label = list(np.unique(classification.astype('str')))

    for node in range(df_allocation.shape[1]):
        for group in range(len(group_label)):
            nodes.append(node)
            classes.append(group_label[group])
            if group_label[group] in groups:
                index = list(groups).index(group_label[group])
                count.append(amount_class_per_node[node][index])
            else:
                count.append(0)
    dictionary = {
        'node': nodes,
        'class': classes,
        'consumption': count,
    }
    y_axis_range = int(max(count) * 5)
    return px.bar(pd.DataFrame(dictionary), x="node", y="consumption", color="class",
                  title=f"{metric} consumption per node - {method}", range_y=[0, y_axis_range])


def _generate_initial_allocations(initial_allocation_path, nodes_size, requests_size):
    number_of_initial_allocations = 500

    allocations = []
    for _ in range(number_of_initial_allocations):
        node = np.random.randint(0, nodes_size)
        mu, sigma = node, nodes_size - 1  # mean and standard deviation

        s = np.random.normal(mu, sigma, size=requests_size)
        alloc = s.astype(int)
        alloc[alloc < 0] = 0
        alloc[alloc > nodes_size - 1] = nodes_size - 1

        allocations.append(alloc)

    filepath = Path(initial_allocation_path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(allocations).to_csv(filepath, sep=';')


def _get_initial_allocations(initial_allocation_path):
    allocations_df = pd.read_csv(initial_allocation_path, sep=';')
    allocations_df.drop(allocations_df.columns[[0]], axis=1, inplace=True)

    allocations = allocations_df.to_numpy().tolist()

    return allocations


def run_ga(n_nodes, test_samples, model_memory, model_cpu):
    for i, sample in enumerate(test_samples):
        memory_consumption = sample['UsedMemory'].values.reshape(-1, 1)
        cpu_consumption_time = sample['UsedCPUTime'].values.reshape(-1, 1)
        cpu_consumption_processors = sample['ReqNProcs'].values.reshape(-1, 1)

        n_requests = memory_consumption.shape[0]
        memory_classification = model_memory.predict(memory_consumption)
        cpu_classification = model_cpu.predict(cpu_consumption_time)

        init_alloc_path = f'./data/input/DAS-2/init_alloc/nodes_{n_nodes}_sample_{i}.csv'
        population = _get_initial_allocations(init_alloc_path)

        # GENETIC ALGORITHM SOLUTION
        ga_solution = GeneticAlgorithm(
            n_requests,
            n_nodes,
            memory_consumption,
            cpu_consumption_processors,
            memory_classification,
            cpu_classification,
            population
        ).optimize(n_generations=30)

        ga_solution = pd.DataFrame(ga_solution)

        filepath = Path('./data/output/DAS-2/ga/alloc_' + str(n_nodes) + '/ga_solution_sample_' + str(i) + '.csv')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        ga_solution.to_csv(filepath, sep=';')


def run_pso(n_nodes, test_samples, model_memory, model_cpu):
    for i, sample in enumerate(test_samples):
        memory_consumption = sample['UsedMemory'].values.reshape(-1, 1)
        cpu_consumption_time = sample['UsedCPUTime'].values.reshape(-1, 1)
        cpu_consumption_processors = sample['ReqNProcs'].values.reshape(-1, 1)

        n_requests = memory_consumption.shape[0]
        memory_classification = model_memory.predict(memory_consumption)
        cpu_classification = model_cpu.predict(cpu_consumption_time)

        init_alloc_path = f'./data/input/DAS-2/init_alloc_{n_nodes}/nodes_{n_nodes}_sample_{i}.csv'
        population = _get_initial_allocations(init_alloc_path)

        # PARTICLE-SWARM (PSO) ALGORITHM SOLUTION
        pso_solution = ParticleSwarmAlgorithm(
            n_requests,
            n_nodes,
            memory_consumption,
            cpu_consumption_processors,
            memory_classification,
            cpu_classification,
            population
        ).optimize(max_iter=30)

        pso_solution = pd.DataFrame(pso_solution)

        filepath = Path('./data/output/DAS-2/pso/alloc_' + str(n_nodes) + '/pso_solution_sample_' + str(i) + '.csv')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        pso_solution.to_csv(filepath, sep=';')


def run_linear(n_nodes, test_samples, model_memory, model_cpu):
    for i, sample in enumerate(test_samples):
        memory_consumption = sample['UsedMemory'].values.reshape(-1, 1)
        cpu_consumption_time = sample['UsedCPUTime'].values.reshape(-1, 1)
        cpu_consumption_processors = sample['ReqNProcs'].values.reshape(-1, 1)

        n_requests = memory_consumption.shape[0]
        memory_classification = model_memory.predict(memory_consumption)
        cpu_classification = model_cpu.predict(cpu_consumption_time)
        # LINEAR SOLUTION
        linear_solution = IntegerLinearSolver(
            n_requests,
            n_nodes,
            memory_consumption,
            cpu_consumption_processors,
            memory_classification,
            cpu_classification
        ).optimize(solver_max_timeout_ms=5000)

        linear_solution = pd.DataFrame(linear_solution)

        filepath = Path('./data/output/DAS-2/linear/alloc_' + str(n_nodes) + '/milp_solution_sample_' + str(i) + '.csv')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        linear_solution.to_csv(filepath, sep=';')


def run_heuristic(n_nodes, test_samples, model_memory, model_cpu):
    for i, sample in enumerate(test_samples):
        memory_consumption = sample['UsedMemory'].values.reshape(-1, 1)
        cpu_consumption_time = sample['UsedCPUTime'].values.reshape(-1, 1)
        cpu_consumption_processors = sample['ReqNProcs'].values.reshape(-1, 1)

        n_requests = memory_consumption.shape[0]
        memory_classification = model_memory.predict(memory_consumption)
        cpu_classification = model_cpu.predict(cpu_consumption_time)

        # HEURISTIC SOLUTION
        heuristic_solution = HeuristicOptimizer(
            n_requests,
            n_nodes,
            memory_consumption,
            cpu_consumption_processors,
            memory_classification,
            cpu_classification
        ).optimize()

        heuristic_solution = pd.DataFrame(heuristic_solution)

        filepath = Path(
            './data/output/DAS-2/heuristic/alloc_' + str(n_nodes) + '/heuristic_solution_sample_' + str(i) + '.csv')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        heuristic_solution.to_csv(filepath, sep=';')


def runInParallel(*fns, n_nodes, test_samples, model_memory, model_cpu):
    proc = []
    for fn in fns:
        p = Process(target=fn, args=(n_nodes, test_samples, model_memory, model_cpu))
        p.start()
        proc.append(p)
    for p in proc:
        p.join()


if __name__ == '__main__':
    reader = DataReader()
    test_samples, model_memory, model_cpu = reader.read(Sample.DAS_2.value)

    n_nodes_candi = [5, 10, 15]
    for i, sample in enumerate(test_samples):
        sample.reset_index(inplace=True)

        filepath = f'./data/input/DAS-2/test_samples/test_sample_{i}.csv'
        sample.to_csv(filepath, sep=';', index=False)

for n_nodes in n_nodes_candi:
        for i, sample in enumerate(test_samples):
            memory_consumption = sample['UsedMemory'].values.reshape(-1, 1)

            n_requests = memory_consumption.shape[0]

            init_alloc_path = f'./data/input/DAS-2/init_alloc/nodes_{n_nodes}_sample_{i}.csv'
            _generate_initial_allocations(init_alloc_path, n_nodes, n_requests)

        runInParallel(run_ga, run_pso, run_linear, run_heuristic, n_nodes=n_nodes, test_samples=test_samples,
                      model_memory=model_memory, model_cpu=model_cpu)



        # memory_fig = _consumption_chart(ga_solution, memory_classification, memory_consumption, 'GA', 'Memory')
        # memory_fig.write_image(f'./data/output/DAS-2/ga/report/ga_solution_sample_{i}_memory.png')
        #
        # cpu_fig = _consumption_chart(ga_solution, cpu_classification, cpu_consumption_processors, 'GA', 'CPU')
        # cpu_fig.write_image(f'./data/output/DAS-2/ga/report/ga_solution_sample_{i}_cpu.png')


        # memory_fig = _consumption_chart(pso_solution, memory_classification, memory_consumption, 'PSO', 'Memory')
        # memory_fig.write_image(f'./data/output/DAS-2/pso/report/pso_solution_sample_{i}_memory.png')

        # cpu_fig = _consumption_chart(pso_solution, cpu_classification, cpu_consumption_processors, 'PSO', 'CPU')
        # cpu_fig.write_image(f'./data/output/DAS-2/pso/report/pso_solution_sample_{i}_cpu.png')

        # memory_fig = _consumption_chart(linear_solution, memory_classification, memory_consumption, 'Linear', 'Memory')
        # memory_fig.write_image(f'./data/output/DAS-2/report/linear_solution_sample_{i}_memory.png')
        #
        # cpu_fig = _consumption_chart(linear_solution, cpu_classification, cpu_consumption_processors, 'Linear', 'CPU')
        # cpu_fig.write_image(f'./data/output/DAS-2/report/linear_solution_sample_{i}_cpu.png')


        #
        # memory_fig = _consumption_chart(heuristic_solution, memory_classification, memory_consumption, 'Heuristic', 'Memory')
        # memory_fig.write_image(f'./data/output/DAS-2/heuristic/report/heuristic_solution_sample_{i}_memory.png')
        #
        # cpu_fig = _consumption_chart(heuristic_solution, cpu_classification, cpu_consumption_processors, 'Heuristic', 'CPU')
        # cpu_fig.write_image(f'./data/output/DAS-2/heuristic/report/heuristic_solution_sample_{i}_cpu.png')
