import numpy as np
import random

from methods.base import BaseOptimizer


class GeneticAlgorithm(BaseOptimizer):
    N_PARENTS = 10
    N_CHILDREN = 300

    def __init__(self,
                 n_requests,
                 n_nodes,
                 memory_consumption,
                 cpu_consumption,
                 memory_classification,
                 cpu_classification,
                 initial_allocation_candi):

        self.n_requests = n_requests
        self.n_nodes = n_nodes
        super().__init__(
            self.n_requests,
            self.n_nodes,
            memory_consumption,
            cpu_consumption,
            memory_classification,
            cpu_classification
        )

        self.init_population = initial_allocation_candi

    def optimize(self, n_generations=300):

        fitness = []
        new_population = self.init_population
        for g in range(n_generations):
            # print(f'generation: {g}')
            if g > 0:
                print(f'ga fitness: {fitness[np.argmin(fitness)]}')

            fitness = []
            for p in new_population:
                fitness.append(self.calc_fitness(p))

            parents = np.array(new_population)[np.argsort(fitness)[0: self.N_PARENTS]]
            children = np.array(new_population)[np.argsort(fitness)[self.N_PARENTS:self.N_CHILDREN]]

            crossover_parents = self.crossover(parents, int(self.N_PARENTS / 2))
            crossover_children = self.crossover(children, int(self.N_CHILDREN / 2))
            crossover_parents_children = self.crossover(np.concatenate([parents, children]),
                                                        int((self.N_PARENTS + self.N_CHILDREN) / 2))

            mutation_parents = self.mutation(parents)
            mutation_children = self.mutation(children)

            mutation_crossover_parents = self.mutation(crossover_parents)
            mutation_crossover_children = self.mutation(crossover_children)
            mutation_crossover_parents_children = self.mutation(crossover_parents_children)

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
        solution = self._suggestion2alloc(new_population[idx])

        print((self.memory_consumption.reshape(-1, 1) * solution).sum(axis=0))

        return solution

    def crossover(self, individuals, n_solutions):
        crossover_point = int(self.n_requests / 2)

        offspring = []
        for k in range(n_solutions):
            # Index of the first parent to mate.
            individual1_idx = k
            # Index of the second parent to mate.
            individual2_idx = (len(individuals) - k - 1)

            offspring.append(
                np.concatenate(
                    [
                       individuals[individual1_idx][0:crossover_point], individuals[individual2_idx][crossover_point:]
                    ]
                )
            )
        return offspring

    def mutation(self, individuals):
        new_individuals = individuals.copy()
        for i in range(len(new_individuals)):
            for j in range(0, self.n_requests, 2):
                new_individuals[i][j] = random.randint(0, self.n_nodes-1)

        return new_individuals



