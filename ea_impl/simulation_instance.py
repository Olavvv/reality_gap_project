import numpy as np
import pygad


class Robot:
    def __init__(self):
        #Two params per joint, 12 joints in total.
        self.params = np.zeros((12,2))


    def update_parameters(self, new_parameters):
        self.params = new_parameters



class SimulationInstance:
    def __init__(self, n_gen: int, n_parents_mating: int, sol_per_pop: int, keep_elitism: int, mutation_probability: float, parent_selection_type: str, save_best_solutions: bool):
        #gene_space = [{Amplitude}, {phi (offset)}*n_joints]
        self._gene_space = [{'low': -1.57,'high':1.57},{None},]*12 #Number of joints in a single robot.
        self._num_genes = len(self._gene_space)

        self.ga = pygad.GA(sol_per_pop=sol_per_pop, 
                           num_generations=n_gen, 
                           num_parents_mating=n_parents_mating,
                           num_genes=self._num_genes,
                           keep_elitism=keep_elitism,
                           parent_selection_type=parent_selection_type,
                           mutation_probability=mutation_probability,
                           save_best_solutions=save_best_solutions,
                           suppress_warnings=True, 
                           fitness_func=self.fitness_func,
                           on_start=self.on_start,
                           on_generation=self.on_gen)
        

        self.rob_array = []
        for i in range(sol_per_pop):
            self.rob_array.append(Robot())

        #TODO: Initialize as many robot-objects as populations in GA.

    def on_start(self, ga_instance):
        #TODO: Assign starting parameters to each robot object.
        for rob in self.rob_array:
            pass
        pass

    def run_ga(self):
        self.ga.run()

    def fitness_func(self, ga_instance, solution, solution_idx):
        #TODO
        """
        Get fitness from simulation instance result, then return it here.
        """
        return 0
    
    def on_gen(self, ga_instance):
        #TODO
        """
        This gets called when a new population is initialized,
        update parameters for each robot in population, then run another sim on every robot, pass the new fitness to each robot 
        """
        pass
    
    def start_ga(self):
        self.ga.run()


sim_instance = SimulationInstance(n_gen=10, n_parents_mating=25, 
                                  sol_per_pop=50, keep_elitism=1, 
                                  mutation_probability=.1, parent_selection_type='sss',
                                  save_best_solutions=True)

print(sim_instance.rob_array)

sim_instance.run_ga()
print(sim_instance.ga.population)
print(sim_instance.ga.population[0], len(sim_instance.ga.population[0]))

rob = Robot()
print(rob.params)
