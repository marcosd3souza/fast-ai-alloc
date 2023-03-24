import numpy as np
import random

from methods.base import BaseOptimizer


class ParticleSwarmAlgorithm(BaseOptimizer):

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
        self.INIT_POPULATION_SIZE = len(initial_allocation_candi)

        self.memory_consumption = memory_consumption
        self.memory_classification = memory_classification

        self.cpu_consumption = cpu_consumption
        self.cpu_classification = cpu_classification

        # Initialize particle velocities using a uniform distribution
        self.velocities = np.random.randint(0, self.n_nodes-1, size=(len(self.init_population), self.n_requests))

        # Initialize the best positions
        self.g_best = np.array([
            int(np.where(self._suggestion2alloc(self.init_population[0])[i] == 1)[0]) for i in range(self.n_requests)
        ])
        self.p_best = self.init_population

    def optimize(self, max_iter=300):

        best_g_fitness = self.calc_fitness(self.g_best)
        fitness = [np.inf] * self.INIT_POPULATION_SIZE
        particles = self.init_population.copy()
        for _ in range(max_iter):
            for i in range(self.INIT_POPULATION_SIZE):
                x = particles[i]
                v = self.velocities[i]
                p_best = self.p_best[i]
                p_best_fitness = self.calc_fitness(self.p_best[i])
                self.velocities[i] = self._update_velocity(x, v, p_best, self.g_best)
                particles[i] = self._update_position(x, v)

                fitness[i] = self.calc_fitness(particles[i])

                # Update the best position for particle i
                if fitness[i] < p_best_fitness:
                    self.p_best[i] = particles[i]
                # Update the best position overall
                if fitness[i] < best_g_fitness:
                    self.g_best = particles[i]
                    best_g_fitness = fitness[i]
                    print(f'pso fitness: {best_g_fitness}')

        solution = self._suggestion2alloc(self.g_best)

        return solution

    def _update_position(self, x, v):
        """
          Update particle position.
          Args:
            x (array-like): particle current position.
            v (array-like): particle current velocity.
          Returns:
            The updated position (array-like).
        """
        x = np.array(x)
        v = np.array(v)
        new_x = x + v
        new_x = np.array([v - (int(v / self.n_nodes) * self.n_nodes) for v in new_x])
        return new_x

    def _update_velocity(self, x, v, p_best, g_best):
        """
          Update particle velocity.
          Args:
            x (array-like): particle current position.
            v (array-like): particle current velocity.
            p_best (array-like): the best position found so far for a particle.
            g_best (array-like): the best position regarding
                                 all the particles found so far.
            c0 (float): the cognitive scaling constant.
            c1 (float): the social scaling constant.
            w (float): the inertia weight
          Returns:
            The updated velocity (array-like).
        """
        x = np.array(x)
        v = np.array(v)
        assert x.shape == v.shape, 'Position and velocity must have same shape'
        # a random number between 0 and 1.
        r = random.randint(0, self.n_nodes - 1)
        p_best = np.array(p_best)
        g_best = np.array(g_best)

        new_v = v + (p_best - x) + (g_best - x) + r
        new_v = np.array([v - (int(v/self.n_nodes) * self.n_nodes) for v in new_v])
        return new_v
