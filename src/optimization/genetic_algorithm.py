import numpy as np
import pandas as pd

from src.optimization.base import BaseOptimizer


class GeneticAlgorithm(BaseOptimizer):
    N_PARENTS = 10
    N_CHILDREN = 300

    def __init__(self,
                 dataframe: pd.DataFrame,
                 metric: str = BaseOptimizer.METRIC_MULTI):
        self.add_log('Initializing Genetic Algorithm')
        super().__init__(dataframe)
        self.metric = metric
        self.init_population = self.__generate_initial_population()

    def __generate_initial_population(self):
        population = []
        for _ in range(self.N_CHILDREN):
            individual = np.zeros([self.amount_pods, self.amount_nodes])
            for i in range(self.amount_pods):
                individual[i, np.random.randint(self.amount_nodes)] = 1
            population.append(individual)
        return population

    def __crossover(self, individuals, n_solutions):
        crossover_point = int(len(self.pods) / 2)

        offspring = []
        for k in range(n_solutions):
            individual1_idx = k
            individual2_idx = (len(individuals) - k - 1)
            offspring.append(
                np.concatenate(
                    [
                        individuals[individual1_idx][0:crossover_point], individuals[individual2_idx][crossover_point:]
                    ]
                )
            )
        return offspring

    def __mutation(self, individuals):
        new_individuals = individuals.copy()
        for individual in range(len(new_individuals)):
            for pod in range(self.amount_pods):
                if np.random.rand() < 0.8:
                    new_individuals[individual][pod, :] = 0
                    new_individuals[individual][pod, np.random.randint(self.amount_nodes)] = 1
                else:
                    new_individuals[individual][pod, :] = individuals[individual][pod, :]

        return new_individuals

    def optimize(self, n_generations: int = 300):
        fitness = []
        new_population = self.init_population
        for g in range(n_generations):
            if g > 0:
                self.add_log(f'generation: {g+1}/{n_generations} - fitness: {fitness[np.argmin(fitness)]}')

            fitness = []
            for p in new_population:
                fitness.append(self.calc_fitness(p, metric=self.metric))

            parents = np.array(new_population)[np.argsort(fitness)[0: self.N_PARENTS]]
            children = np.array(new_population)[np.argsort(fitness)[self.N_PARENTS:self.N_CHILDREN]]

            crossover_parents = self.__crossover(parents, int(self.N_PARENTS / 2))
            crossover_children = self.__crossover(children, int(self.N_CHILDREN / 2))
            crossover_parents_children = self.__crossover(np.concatenate([parents, children]),
                                                          int((self.N_PARENTS + self.N_CHILDREN) / 2))

            mutation_parents = self.__mutation(parents)
            mutation_children = self.__mutation(children)

            mutation_crossover_parents = self.__mutation(crossover_parents)
            mutation_crossover_children = self.__mutation(crossover_children)
            mutation_crossover_parents_children = self.__mutation(crossover_parents_children)

            new_population = np.concatenate([
                np.array(parents),
                np.array(crossover_parents),
                np.array(crossover_children),
                np.array(crossover_parents_children),
                np.array(mutation_parents),
                np.array(mutation_children),
                np.array(mutation_crossover_parents),
                np.array(mutation_crossover_children),
                np.array(mutation_crossover_parents_children)
            ])
        idx = np.argmin(fitness)
        solution = new_population[idx]

        self.add_log((self.pods_memory.reshape(-1, 1) * solution).sum(axis=0))

        return solution
