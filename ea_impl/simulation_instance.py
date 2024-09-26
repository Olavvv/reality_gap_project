import numpy as np
import pygad
import matplotlib.pyplot as plt
from robot import Robot
import time

class SimulationInstance:
    """
    SimulationInstance class, takes in parameters for GA to initialize,
    it has methods for the fitness function for the GA.
    Idea is to initialize a random set of parameteres (within a certain range), then feed these into MJX to simulate.
    After simulation we feed the fitness (distance to current goal/objective) back into this class, and progress the GA
    based on the fitness of each robot.
    """

    def __init__(self, n_gen: int, n_parents_mating: int, sol_per_pop: int, keep_elitism: int, mutation_probability: float,
                 mutation_type: str, 
                 parent_selection_type: str, save_best_solutions: bool):
        
        self.SEED = 8523945
        
        self._on_gen = 0

        #gene_space = [{Amplitude}, {phi (offset)}*n_joints]
        self._gene_space = [{'low': -1.57,'high':1.57},{'low':-np.pi, 'high':np.pi},]*12 #Number of joints in a single robot.
        self._num_genes = len(self._gene_space)

        self.ga = pygad.GA(sol_per_pop=sol_per_pop, 
                           num_generations=n_gen, 
                           num_parents_mating=n_parents_mating,
                           num_genes=self._num_genes,
                           keep_elitism=keep_elitism,
                           mutation_probability=mutation_probability,
                           save_best_solutions=save_best_solutions,
                           suppress_warnings=False,
                           parent_selection_type=parent_selection_type,
                           mutation_type=mutation_type,
                           random_seed=self.SEED,
                           parallel_processing=2, 
                           fitness_func=self.fitness_func,
                           on_start=self.on_start,
                           on_generation=self.on_gen)
        

        self.rob_array = [Robot() for i in range(sol_per_pop)]

        #TODO: Initialize as many robot-objects as populations in GA.

    def on_start(self, ga_instance):
        #TODO: Assign starting parameters to each robot object.
        for rob in self.rob_array:
            pass
        pass

    def run_ga(self):
        #Starts the GA
        self.ga.run()

    def fitness_func(self, ga_instance, solution, solution_idx):
        #TODO
        """
        Get fitness from simulation instance result, then return it here.
        """
        #print(solution,solution_idx)
        #print(solution_idx)
        #print(ga_instance.population[0])
        return sum(solution)
    
    def on_gen(self, ga_instance):
        #TODO
        """
        This gets called when a new population is initialized,
        update parameters for each robot in population, then run another sim on every robot, pass the new fitness to each robot.
        Idea is to feed the new parameters into MJX, then simulate. (Sine-wave controller).
        """
        try: 
            # Get the best solution from the current generation
            best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()

            # Print the generation number, best fitness score, and best solution
            print(f"Generation: {ga_instance.generations_completed}, Best fitness: {best_solution_fitness}, Best solution: {best_solution}")

        except Exception as e:
            print(f"Error in on_generation function: {e}")
            raise e
    
    def start_ga(self):
        self.ga.run()

start = time.time()
sim_instance = SimulationInstance(n_gen=500, n_parents_mating=4, 
                                  sol_per_pop=10, keep_elitism=1, 
                                  mutation_probability=[.25,.1], parent_selection_type='sss',
                                  save_best_solutions=True, mutation_type="adaptive")

sim_instance.run_ga()

print(f"Time taken is: {time.time() - start}")


"""
print(sum(sim_instance.ga.best_solutions[-1]))
print(sim_instance.ga.population)
print(sim_instance.ga.population[0], len(sim_instance.ga.population[0]))
"""

