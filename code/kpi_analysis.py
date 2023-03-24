from data_reader import DataReader
from enum import Enum
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import concurrent.futures
import more_itertools


class Sample(Enum):
    DAS_2 = 'DAS-2/'
    MON = 'MON_2022-04-18-04-25-32'
    TUE = 'TUE_2022-04-19-11-30-24'
    WED = 'WED_2022-04-20-11-32-46'
    THU = 'THU_2022-04-21-05-37-53'
    FRI = 'FRI_2022-04-22-11-33-35'
    SAT = 'SAT_2022-04-23-11-36-09'
    SUN = 'SUN_2022-04-17-11-28-02'


def get_consumption_kpi(n_nodes, mem_cons, solution):
    sol_cons = (mem_cons.reshape(-1, 1) * solution).sum(axis=0)
    solution = sol_cons / sol_cons.sum()
    non_zero_solution = np.where(solution != 0)[0]

    return np.sum(- solution[non_zero_solution] * np.log(solution[non_zero_solution])) / np.log(n_nodes)


def get_class_kpi(n_nodes, classification, solution):
    # TODO: Fix division by zero
    perc_amount_class_per_node_sol = []
    for node in range(n_nodes):
        list_reqs_allocated = np.where(solution[:, node] == 1)[0]
        amount_class = np.array([0 for _ in range(max(classification) + 1)])
        for req in list_reqs_allocated:
            class_index = int(classification[req])
            amount_class[class_index] = amount_class[class_index] + 1

        perc_amount_class_per_node_sol.append(np.divide(amount_class, sum(amount_class)))
    proportions = [np.bincount(classification)[i] / sum(np.bincount(classification)) for i in
                   range(max(classification) + 1)]

    return 1 - np.mean([np.sqrt(sum((s - proportions) ** 2)) for s in perc_amount_class_per_node_sol])


def get_df_results(sample_id, n_requests, n_nodes, memory_consumption, memory_classification, cpu_consumption,
                   cpu_classification, solution):
    results = []

    if solution is None:
        node = np.random.randint(0, n_nodes)
        mu, sigma = node, n_nodes - 1  # mean and standard deviation

        s = np.random.normal(mu, sigma, size=n_requests)
        rand_alloc = s.astype(int)
        rand_alloc[rand_alloc < 0] = 0
        rand_alloc[rand_alloc > n_nodes - 1] = n_nodes - 1

        alloc = np.zeros([n_requests, n_nodes])

        for req, node in enumerate(rand_alloc):
            alloc[req, node] = 1

        solution_local = alloc
    else:
        solution_local = solution

    kpi_memory_consumption = get_consumption_kpi(n_nodes, memory_consumption, solution_local)
    kpi_cpu_consumption = get_consumption_kpi(n_nodes, cpu_consumption, solution_local)

    kpi_memory_class = get_class_kpi(n_nodes, memory_classification, solution_local)
    kpi_cpu_class = get_class_kpi(n_nodes, cpu_classification, solution_local)

    results.append({
        'sample': sample_id,
        'kpi_memory_consumption': kpi_memory_consumption,
        'kpi_memory_class': kpi_memory_class,
        'kpi_cpu_consumption': kpi_cpu_consumption,
        'kpi_cpu_class': kpi_cpu_class
    })

    return pd.DataFrame(results)


def run_kpis_on_samples(group_files_init_alloc, model_memory, model_cpu, algorithm):
    results = pd.DataFrame()

    for file_name in group_files_init_alloc:
        if file_name is not None and file_name != '.gitignore':
            name_components = file_name.split("_")
            node = int(name_components[1])
            sample_id = int(name_components[3].split(".")[0])

            sample = pd.read_csv('./data/input/DAS-2/test_samples/test_sample_' + str(sample_id) + '.csv', sep=';')

            memory_consumption = sample['UsedMemory'].values.reshape(-1, 1)
            cpu_consumption_time = sample['UsedCPUTime'].values.reshape(-1, 1)
            cpu_consumption_processors = sample['ReqNProcs'].values.reshape(-1, 1)

            n_requests = memory_consumption.shape[0]
            memory_classification = model_memory.predict(memory_consumption)

            cpu_classification = model_cpu.predict(cpu_consumption_time)

            if algorithm == 'baseline':
                solution = None

            elif algorithm == 'linear':
                solution = pd.read_csv('./data/output/DAS-2/linear/alloc_' + str(node) + '/milp_solution_sample_' + str(
                    sample_id) + '.csv', sep=';').values[:, 1:]

            else:
                solution = pd.read_csv(
                    './data/output/DAS-2/' + algorithm + '/alloc_' + str(
                        node) + '/' + algorithm + '_solution_sample_' + str(sample_id) + '.csv',
                    sep=';').values[:, 1:]

            sample_results_df = get_df_results(
                sample_id,
                n_requests,
                node,
                memory_consumption,
                memory_classification,
                cpu_consumption_processors,
                cpu_classification,
                solution
            )

            results = pd.concat([results, sample_results_df])

    return results


def plot_histograms(kpi_name, n_nodes, linear_kpi, heuristic_kpi, ga_kpi, pso_kpi):
    fig, ax = plt.subplots(1, 4)

    plt.rcParams['figure.figsize'] = 18, 2
    plt.rcParams.update({'font.size': 12})

    df = pd.DataFrame({
        'Linear': linear_kpi,
        'Heuristic': heuristic_kpi,
        'GA': ga_kpi,
        'PSO': pso_kpi
    })

    c0 = sns.histplot(df['Linear'], kde=True, ax=ax[0])
    c1 = sns.histplot(df['Heuristic'], kde=True, ax=ax[1])
    c2 = sns.histplot(df['GA'], kde=True, ax=ax[2])
    c3 = sns.histplot(df['PSO'], kde=True, ax=ax[3])
    c0.set(ylabel=None, xlabel=None)
    c1.set(ylabel=None, xlabel=None)
    c2.set(ylabel=None, xlabel=None)
    c3.set(ylabel=None, xlabel=None)

    if kpi_name != "changes":
        c0.set_title('Linear', pad=-14, y=1.19, x=0.8)
        c1.set_title('Heuristic', pad=-14, y=1.19, x=0.8)
        c2.set_title('GA', pad=-14, y=1.19, x=0.8)
        c3.set_title('PSO', pad=-14, y=1.19, x=0.8)
    else:
        c0.set_title('Linear', pad=-14, y=1.10, x=0.8)
        c1.set_title('Heuristic', pad=-14, y=1.10, x=0.8)
        c2.set_title('GA', pad=-14, y=1.10, x=0.8)
        c3.set_title('PSO', pad=-14, y=1.10, x=0.8)

    plt.tight_layout()
    plt.savefig('./data/output/DAS-2/kpi_analysis/figs/' + str(kpi_name) + '_hists_nodes_' + str(n_nodes) + '.pdf',
                format='pdf')
    plt.close()
    print('KPI ' + str(kpi_name) + ': \n', df.corr())


def kpi_analysis(num_nodes):
    reader = DataReader()
    _, model_memory, model_cpu = reader.read(Sample.DAS_2.value)

    # Get a list of all initial allocation files
    files_init_alloc = os.listdir("./data/input/DAS-2/init_alloc_" + str(num_nodes) + "/")

    ############
    # Baseline #
    # ##########
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=None)
    futures = [executor.submit(run_kpis_on_samples, group_init_alloc, model_memory, model_cpu, 'baseline')
               for group_init_alloc in more_itertools.grouper(16, files_init_alloc)]
    concurrent.futures.wait(futures)

    results_baseline = pd.DataFrame()
    for i in range(len(futures)):
        results_baseline = pd.concat([results_baseline, futures[i].result()])
    print("KPIs for Baseline completed.")

    ##########
    # Linear #
    ##########
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=None)
    futures = [executor.submit(run_kpis_on_samples, group_init_alloc, model_memory, model_cpu, 'linear')
               for group_init_alloc in more_itertools.grouper(16, files_init_alloc)]
    concurrent.futures.wait(futures)

    results_linear = pd.DataFrame()
    for i in range(len(futures)):
        results_linear = pd.concat([results_linear, futures[i].result()])
    print("KPIs for Linear completed.")

    #############
    # Heuristic #
    #############
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=None)
    futures = [executor.submit(run_kpis_on_samples, group_init_alloc, model_memory, model_cpu, 'heuristic')
               for group_init_alloc in more_itertools.grouper(16, files_init_alloc)]
    concurrent.futures.wait(futures)

    results_heuristic = pd.DataFrame()
    for i in range(len(futures)):
        results_heuristic = pd.concat([results_heuristic, futures[i].result()])
    print("KPIs for Heuristic completed.")

    ######
    # GA #
    ######
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=None)
    futures = [executor.submit(run_kpis_on_samples, group_init_alloc, model_memory, model_cpu, 'ga')
               for group_init_alloc in more_itertools.grouper(16, files_init_alloc)]
    concurrent.futures.wait(futures)

    results_ga = pd.DataFrame()
    for i in range(len(futures)):
        results_ga = pd.concat([results_ga, futures[i].result()])
    print("KPIs for GA completed.")

    #######
    # PSO #
    #######
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=None)
    futures = [executor.submit(run_kpis_on_samples, group_init_alloc, model_memory, model_cpu, 'pso')
               for group_init_alloc in more_itertools.grouper(16, files_init_alloc)]
    concurrent.futures.wait(futures)

    results_pso = pd.DataFrame()
    for i in range(len(futures)):
        results_pso = pd.concat([results_pso, futures[i].result()])
    print("KPIs for PSO completed.")

    # Save dataframes as CSVs
    results_ga.to_csv('./data/output/DAS-2/kpi_analysis/nodes_' + str(num_nodes) + '_results_ga.csv', sep=';',
                      index=False)
    results_pso.to_csv('./data/output/DAS-2/kpi_analysis/nodes_' + str(num_nodes) + '_results_pso.csv', sep=';',
                       index=False)
    results_linear.to_csv('./data/output/DAS-2/kpi_analysis/nodes_' + str(num_nodes) + '_results_linear.csv', sep=';',
                          index=False)
    results_heuristic.to_csv('./data/output/DAS-2/kpi_analysis/nodes_' + str(num_nodes) + '_results_heuristic.csv',
                             sep=';', index=False)


    return results_baseline, results_linear, results_heuristic, results_ga, results_pso


def create_latex_table(number_nodes, results_baseline, results_linear, results_heuristic, results_ga, results_pso):

    consumption_kpi_latex = "Consumption KPI Latex Table:\n"
    class_kpi_latex = 'Class KPI Latex Table:\n'

    for i, nodes in enumerate(number_nodes):
        desc_baseline_results = results_baseline[i].describe()
        desc_linear_results = results_linear[i].describe()
        desc_heuristic_results = results_heuristic[i].describe()
        desc_ga_results = results_ga[i].describe()
        desc_pso_results = results_pso[i].describe()

        print("\n Baseline:")
        print(desc_baseline_results.to_string())
        print("\n Linear:")
        print(desc_linear_results.to_string())
        print("\n Heuristic:")
        print(desc_heuristic_results.to_string())
        print("\n GA:")
        print(desc_ga_results.to_string())
        print("\n PSO:")
        print(desc_pso_results.to_string())

        ###############################################################################################################
        consumption_kpi_latex += '\multirow{{2}}{{*}}{{{}}} & Min / Max & {:.3f} / {:.3f} & {:.3f} / {:.3f} & ' \
                                 '{:.3f} / {:.3f} & {:.3f} / {:.3f} & {:.3f} / {:.3f} & {:.3f} / {:.3f} & ' \
                                 '{:.3f} / {:.3f} & {:.3f} / {:.3f} & {:.3f} / {:.3f} & {:.3f} / {:.3f} \\\\\n'.format(
            nodes,
            desc_baseline_results['kpi_memory_consumption']['min'], desc_baseline_results['kpi_memory_consumption']['max'],
            desc_baseline_results['kpi_cpu_consumption']['min'], desc_baseline_results['kpi_cpu_consumption']['max'],
            desc_linear_results['kpi_memory_consumption']['min'], desc_linear_results['kpi_memory_consumption']['max'],
            desc_linear_results['kpi_cpu_consumption']['min'], desc_linear_results['kpi_cpu_consumption']['max'],
            desc_heuristic_results['kpi_memory_consumption']['min'], desc_heuristic_results['kpi_memory_consumption']['max'],
            desc_heuristic_results['kpi_cpu_consumption']['min'], desc_heuristic_results['kpi_cpu_consumption']['max'],
            desc_ga_results['kpi_memory_consumption']['min'], desc_ga_results['kpi_memory_consumption']['max'],
            desc_ga_results['kpi_cpu_consumption']['min'], desc_ga_results['kpi_cpu_consumption']['max'],
            desc_pso_results['kpi_memory_consumption']['min'], desc_pso_results['kpi_memory_consumption']['max'],
            desc_pso_results['kpi_cpu_consumption']['min'], desc_pso_results['kpi_cpu_consumption']['max'])

        consumption_kpi_latex += '~ & Mean $\\pm$ Std & {:.3f}$\\pm${:.3f} & {:.3f}$\\pm${:.3f} & ' \
                                 '{:.3f}$\\pm${:.3f} & {:.3f}$\\pm${:.3f} & {:.3f}$\\pm${:.3f} & ' \
                                 '{:.3f}$\\pm${:.3f} & {:.3f}$\\pm${:.3f} & {:.3f}$\\pm${:.3f} & ' \
                                 '{:.3f}$\\pm${:.3f} & {:.3f}$\\pm${:.3f} \\\\ \\hline\n'.format(
            desc_baseline_results['kpi_memory_consumption']['mean'], desc_baseline_results['kpi_memory_consumption']['std'],
            desc_baseline_results['kpi_cpu_consumption']['mean'], desc_baseline_results['kpi_cpu_consumption']['std'],
            desc_linear_results['kpi_memory_consumption']['mean'], desc_linear_results['kpi_memory_consumption']['std'],
            desc_linear_results['kpi_cpu_consumption']['mean'], desc_linear_results['kpi_cpu_consumption']['std'],
            desc_heuristic_results['kpi_memory_consumption']['mean'],
            desc_heuristic_results['kpi_memory_consumption']['std'],
            desc_heuristic_results['kpi_cpu_consumption']['mean'], desc_heuristic_results['kpi_cpu_consumption']['std'],
            desc_ga_results['kpi_memory_consumption']['mean'], desc_ga_results['kpi_memory_consumption']['std'],
            desc_ga_results['kpi_cpu_consumption']['mean'], desc_ga_results['kpi_cpu_consumption']['std'],
            desc_pso_results['kpi_memory_consumption']['mean'], desc_pso_results['kpi_memory_consumption']['std'],
            desc_pso_results['kpi_cpu_consumption']['mean'], desc_pso_results['kpi_cpu_consumption']['std'])

        ###############################################################################################################
        class_kpi_latex += '\multirow{{2}}{{*}}{{{}}} & Min / Max & {:.3f} / {:.3f} & {:.3f} / {:.3f} & ' \
                           '{:.3f} / {:.3f} & {:.3f} / {:.3f} & {:.3f} / {:.3f} & {:.3f} / {:.3f} & ' \
                           '{:.3f} / {:.3f} & {:.3f} / {:.3f} & {:.3f} / {:.3f} & {:.3f} / {:.3f} \\\\\n'.format(
            nodes,
            desc_baseline_results['kpi_memory_class']['min'], desc_baseline_results['kpi_memory_class']['max'],
            desc_baseline_results['kpi_cpu_class']['min'], desc_baseline_results['kpi_cpu_class']['max'],
            desc_linear_results['kpi_memory_class']['min'], desc_linear_results['kpi_memory_class']['max'],
            desc_linear_results['kpi_cpu_class']['min'], desc_linear_results['kpi_cpu_class']['max'],
            desc_heuristic_results['kpi_memory_class']['min'], desc_heuristic_results['kpi_memory_class']['max'],
            desc_heuristic_results['kpi_cpu_class']['min'], desc_heuristic_results['kpi_cpu_class']['max'],
            desc_ga_results['kpi_memory_class']['min'], desc_ga_results['kpi_memory_class']['max'],
            desc_ga_results['kpi_cpu_class']['min'], desc_ga_results['kpi_cpu_class']['max'],
            desc_pso_results['kpi_memory_class']['min'], desc_pso_results['kpi_memory_class']['max'],
            desc_pso_results['kpi_cpu_class']['min'], desc_pso_results['kpi_cpu_class']['max'])

        class_kpi_latex += '~ & Mean $\\pm$ Std & {:.3f}$\\pm${:.3f} & {:.3f}$\\pm${:.3f} & {:.3f}$\\pm${:.3f} & ' \
                           '{:.3f}$\\pm${:.3f} & {:.3f}$\\pm${:.3f} & {:.3f}$\\pm${:.3f} & {:.3f}$\\pm${:.3f} & ' \
                           '{:.3f}$\\pm${:.3f} & {:.3f}$\\pm${:.3f} & {:.3f}$\\pm${:.3f} \\\\ \\hline\n'.format(
            desc_baseline_results['kpi_memory_class']['mean'], desc_baseline_results['kpi_memory_class']['std'],
            desc_baseline_results['kpi_cpu_class']['mean'], desc_baseline_results['kpi_cpu_class']['std'],
            desc_linear_results['kpi_memory_class']['mean'], desc_linear_results['kpi_memory_class']['std'],
            desc_linear_results['kpi_cpu_class']['mean'], desc_linear_results['kpi_cpu_class']['std'],
            desc_heuristic_results['kpi_memory_class']['mean'],
            desc_heuristic_results['kpi_memory_class']['std'],
            desc_heuristic_results['kpi_cpu_class']['mean'], desc_heuristic_results['kpi_cpu_class']['std'],
            desc_ga_results['kpi_memory_class']['mean'], desc_ga_results['kpi_memory_class']['std'],
            desc_ga_results['kpi_cpu_class']['mean'], desc_ga_results['kpi_cpu_class']['std'],
            desc_pso_results['kpi_memory_class']['mean'], desc_pso_results['kpi_memory_class']['std'],
            desc_pso_results['kpi_cpu_class']['mean'], desc_pso_results['kpi_cpu_class']['std'])

        # For each KPI, plot a histogram for the four algorithms
        linear_kpi_memory_consumption = results_linear[i]['kpi_memory_consumption']
        linear_kpi_memory_class = results_linear[i]['kpi_memory_class']
        linear_kpi_cpu_consumption = results_linear[i]['kpi_cpu_consumption']
        linear_kpi_cpu_class = results_linear[i]['kpi_cpu_class']

        heuristic_kpi_memory_consumption = results_heuristic[i]['kpi_memory_consumption']
        heuristic_kpi_memory_class = results_heuristic[i]['kpi_memory_class']
        heuristic_kpi_cpu_consumption = results_heuristic[i]['kpi_cpu_consumption']
        heuristic_kpi_cpu_class = results_heuristic[i]['kpi_cpu_class']

        ga_kpi_memory_consumption = results_ga[i]['kpi_memory_consumption']
        ga_kpi_memory_class = results_ga[i]['kpi_memory_class']
        ga_kpi_cpu_consumption = results_ga[i]['kpi_cpu_consumption']
        ga_kpi_cpu_class = results_ga[i]['kpi_cpu_class']

        pso_kpi_memory_consumption = results_pso[i]['kpi_memory_consumption']
        pso_kpi_memory_class = results_pso[i]['kpi_memory_class']
        pso_kpi_cpu_consumption = results_pso[i]['kpi_cpu_consumption']
        pso_kpi_cpu_class = results_pso[i]['kpi_cpu_class']

        # Histogram for Memory Consumption KPI
        plot_histograms('memory_consumption', nodes, linear_kpi_memory_consumption, heuristic_kpi_memory_consumption,
                        ga_kpi_memory_consumption, pso_kpi_memory_consumption)

        # Histogram for Memory Class KPI
        plot_histograms('memory_class', nodes, linear_kpi_memory_class, heuristic_kpi_memory_class,
                        ga_kpi_memory_class, pso_kpi_memory_class)

        # Histogram for CPU Consumption KPI
        plot_histograms('cpu_consumption', nodes, linear_kpi_cpu_consumption, heuristic_kpi_cpu_consumption,
                        ga_kpi_cpu_consumption, pso_kpi_cpu_consumption)

        # Histogram for Memory Class KPI
        plot_histograms('cpu_class', nodes, linear_kpi_cpu_class, heuristic_kpi_cpu_class,
                        ga_kpi_cpu_class, pso_kpi_cpu_class)

    tex_output = consumption_kpi_latex + '\n' + class_kpi_latex

    with open('./data/output/DAS-2/kpi_analysis/latex_tables.txt', 'w') as f:
        print(tex_output, file=f)


def main():
    n_nodes = [5, 10, 15]

    list_results_baseline = [None] * len(n_nodes)
    list_results_linear = [None] * len(n_nodes)
    list_results_heuristic = [None] * len(n_nodes)
    list_results_ga = [None] * len(n_nodes)
    list_results_pso = [None] * len(n_nodes)

    for i in range(len(n_nodes)):
        list_results_baseline[i], list_results_linear[i], list_results_heuristic[i], list_results_ga[i], \
        list_results_pso[i] = kpi_analysis(n_nodes[i])

    # Create Latex Table
    create_latex_table(n_nodes, list_results_baseline, list_results_linear, list_results_heuristic, list_results_ga,
                       list_results_pso)


if __name__ == '__main__':
    main()
