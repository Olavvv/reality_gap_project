import numpy as np
import pygad

coef = np.random.randint(20, size=(1,10))
print("Solution Landscape: ", coef)
desired = np.random.randint(100)
print("Desired output: ",desired)

def fitness_func(ga_instance, solution, solution_dix):
    output = np.sum(solution*coef)
    fitness = 1/(np.abs(output-desired) + 0.000000001)
    return fitness


#Gets called on 
def on_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])


sol_per_pop = 10
num_generations = 20
num_genes = len(coef)
num_parents_mating = 5



#Set the maximum and min value of the alleles of the genes.

#gen_space = [{Ampiltude}, {phi (offset)}]
gen_space = [{'low': }, {None}]

ga_instance = pygad.GA(num_generations=num_generations, 
                       sol_per_pop=sol_per_pop, 
                       fitness_func=fitness_func,
                       num_genes=num_genes, 
                       num_parents_mating=num_parents_mating, 
                       mutation_type=None, 
                       on_generation=on_gen)

ga_instance.run()

print("Parameters of the best solution: {solution}".format(solution=ga_instance.best_solution()))
print("Prediction = ", np.sum(ga_instance.best_solution()[0]*coef))

ga_instance.plot_fitness()