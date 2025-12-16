import logging
import pygad
import numpy
import benchmark_functions as bf
import opfunu.cec_based.cec2014 as cec2014

from crossover_methods import CrossoverMethods
from mutation_methods import MutationMethods

SELECTED_FUNC = "CEC"
SELECTED_CROSSOVER = "linear" 
SELECTED_MUTATION = "gaussian"

if SELECTED_FUNC == "MCCORMICK":
    n_vars = 2 
    func_inst = bf.McCormick()
    bounds_list = [[-1.5, 4.0], [-3.0, 4.0]] 
    global_min = func_inst.minimum().score
    def calculate_func(x):
        return func_inst(x)

elif SELECTED_FUNC == "CEC":
    n_vars = 10 
    func_inst = cec2014.F62014(ndim=n_vars)
    
    raw_bounds = func_inst.bounds 
    is_matrix = isinstance(raw_bounds, (list, numpy.ndarray)) and \
                len(raw_bounds) > 0 and \
                isinstance(raw_bounds[0], (list, tuple, numpy.ndarray))

    if is_matrix and len(raw_bounds) == n_vars:
        bounds_list = raw_bounds
    else:
        if isinstance(raw_bounds, (list, tuple, numpy.ndarray)) and len(raw_bounds) == 2:
            bounds_list = [raw_bounds] * n_vars
        else:
            bounds_list = [[-100, 100]] * n_vars

    global_min = func_inst.f_global
    def calculate_func(x):
        return func_inst.evaluate(x)

num_genes = n_vars 
gene_space = [{'low': b[0], 'high': b[1]} for b in bounds_list]

def fitness_func(ga_instance, solution, solution_idx):
    val = calculate_func(solution)
    fitness = 1.0 / (abs(val - global_min) + 0.00000001)
    return fitness

fitness_function = fitness_func

def custom_crossover_func(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    
    while len(offspring) < offspring_size[0]:
        parent1 = parents[idx % parents.shape[0]]
        parent2 = parents[(idx + 1) % parents.shape[0]]
        
        genes_c1, genes_c2, genes_c3 = [], [], [] 
        
        for i in range(offspring_size[1]):
            p1 = parent1[i]
            p2 = parent2[i]
            bound = bounds_list[i]
            
            if SELECTED_CROSSOVER == "arithmetic":
                g1, g2 = CrossoverMethods.arithmetic_crossover(p1, p2, bound)
                genes_c1.append(g1); genes_c2.append(g2)
                
            elif SELECTED_CROSSOVER == "linear":
                g1, g2, g3 = CrossoverMethods.linear_crossover(p1, p2, bound)
                genes_c1.append(g1); genes_c2.append(g2); genes_c3.append(g3)
                
            elif SELECTED_CROSSOVER == "blend_alpha":
                g1, g2 = CrossoverMethods.blend_alpha_crossover(p1, p2, bound, alpha=0.5)
                genes_c1.append(g1); genes_c2.append(g2)
                
            elif SELECTED_CROSSOVER == "blend_alpha_beta":
                g1, g2 = CrossoverMethods.blend_alpha_beta_crossover(p1, p2, bound, alpha=0.5, beta=0.5)
                genes_c1.append(g1); genes_c2.append(g2)
                
            elif SELECTED_CROSSOVER == "average":
                g1 = CrossoverMethods.average_crossover(p1, p2, bound)
                genes_c1.append(g1)
        
        if len(offspring) < offspring_size[0]: offspring.append(numpy.array(genes_c1))
        
        if SELECTED_CROSSOVER in ["arithmetic", "linear", "blend_alpha", "blend_alpha_beta"]:
            if len(offspring) < offspring_size[0]: offspring.append(numpy.array(genes_c2))
                
        if SELECTED_CROSSOVER == "linear":
            if len(offspring) < offspring_size[0]: offspring.append(numpy.array(genes_c3))
                
        idx += 1
    return numpy.array(offspring)

def custom_mutation_func(offspring, ga_instance):
    for idx in range(offspring.shape[0]):
        individual = offspring[idx]
        
        if SELECTED_MUTATION == "gaussian":
            mutated = MutationMethods.gaussian_mutation(
                genes=individual, p_mutation=0.2, bounds=bounds_list, sigma=0.5)
        elif SELECTED_MUTATION == "uniform":
            mutated = MutationMethods.uniform_mutation(
                genes=individual, p_mutation=0.2, bounds=bounds_list)
        else:
            mutated = individual
            
        offspring[idx] = numpy.array(mutated)
    return offspring

gene_type = float
num_generations = 100
sol_per_pop = 80
num_parents_mating = 50
mutation_num_genes = 1
parent_selection_type = "tournament"
crossover_type = custom_crossover_func
mutation_type = custom_mutation_func

level = logging.DEBUG
name = 'logfile_real.txt'
logger = logging.getLogger(name)
logger.setLevel(level)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(message)s')
console_handler.setFormatter(console_format)
if not logger.handlers:
    logger.addHandler(console_handler)

def on_generation(ga_instance):
    ga_instance.logger.info("Generation = {generation}".format(generation=ga_instance.generations_completed))
    solution, solution_fitness, solution_idx = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    
    true_val = global_min + (1.0 / solution_fitness) - 0.00000001
    
    ga_instance.logger.info("Best Fitness = {fitness}".format(fitness=solution_fitness))
    ga_instance.logger.info("Function Value = {val}".format(val=true_val))
    
    pop_vals = [global_min + (1.0/x) - 0.00000001 for x in ga_instance.last_generation_fitness]
    ga_instance.logger.info("Min (Val) = {min}".format(min=numpy.min(pop_vals)))
    ga_instance.logger.info("Max (Val) = {max}".format(max=numpy.max(pop_vals)))
    ga_instance.logger.info("Average (Val) = {average}".format(average=numpy.average(pop_vals)))
    ga_instance.logger.info("Std (Val) = {std}".format(std=numpy.std(pop_vals)))
    ga_instance.logger.info("\r\n")

ga_instance = pygad.GA(num_generations=num_generations,
           sol_per_pop=sol_per_pop,
           num_parents_mating=num_parents_mating,
           num_genes=num_genes,
           fitness_func=fitness_func,
           gene_type=gene_type,
           gene_space=gene_space,
           mutation_num_genes=mutation_num_genes,
           parent_selection_type=parent_selection_type,
           crossover_type=crossover_type,
           mutation_type=mutation_type,
           keep_elitism=1,
           K_tournament=3,
           logger=logger,
           on_generation=on_generation,
           parallel_processing=None)

ga_instance.run()

best = ga_instance.best_solution()
solution, solution_fitness, solution_idx = ga_instance.best_solution()

print("Best solution (real) : {solution}".format(solution=solution))
real_val = global_min + (1.0 / solution_fitness) - 0.00000001
print("Function value of the best solution = {val}".format(val=real_val))
print("Global minimum expected = {min}".format(min=global_min))

ga_instance.best_solutions_fitness = [global_min + (1.0 / x) - 0.00000001 for x in ga_instance.best_solutions_fitness]
ga_instance.plot_fitness()