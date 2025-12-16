import logging
import pygad
import numpy
import benchmark_functions as bf
import opfunu.cec_based.cec2014 as cec2014

SELECTED_FUNC = "CEC" 

bits_per_variable = 20 

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
        raise ValueError("Cannot determine bounds.")

    global_min = func_inst.f_global
    def calculate_func(x):
        return func_inst.evaluate(x)

num_genes = bits_per_variable * n_vars 

def decode_ind(individual):
    decoded = []
    max_int = 2**bits_per_variable - 1
    for i in range(n_vars):
        start = i * bits_per_variable
        end = (i + 1) * bits_per_variable
        sub_gene = individual[start:end]
        
        int_val = 0
        for bit in sub_gene:
            int_val = (int_val << 1) | int(bit)
            
        min_val, max_val = bounds_list[i]
        val = min_val + (int_val / max_int) * (max_val - min_val)
        decoded.append(val)
    return decoded

def fitness_func(ga_instance, solution, solution_idx):
    decoded_solution = decode_ind(solution)
    val = calculate_func(decoded_solution) 
    fitness = 1.0 / (abs(val - global_min) + 0.000001)
    return fitness

fitness_function = fitness_func
gene_type = int
num_generations = 100
sol_per_pop = 80
num_parents_mating = 50
init_range_low = 0
init_range_high = 2 
mutation_num_genes = 1
parent_selection_type = "tournament"
crossover_type = "uniform"
mutation_type = "random"


level = logging.DEBUG
name = 'logfile.txt'
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
    
    true_val = global_min + (1.0 / solution_fitness) - 0.000001
    decoded_sol = decode_ind(solution)

    ga_instance.logger.info("Best Fitness = {fitness}".format(fitness=solution_fitness))
    ga_instance.logger.info("Function Value = {val}".format(val=true_val))
    ga_instance.logger.info("Individual (Decoded, first 2) = {solution}...".format(solution=decoded_sol[:2])) 
    ga_instance.logger.info("\r\n")


ga_instance = pygad.GA(num_generations=num_generations,
           sol_per_pop=sol_per_pop,
           num_parents_mating=num_parents_mating,
           num_genes=num_genes,
           fitness_func=fitness_func,
           gene_type=gene_type,
           init_range_low=init_range_low,
           init_range_high=init_range_high,
           gene_space=[0, 1], 
           mutation_num_genes=mutation_num_genes,
           parent_selection_type=parent_selection_type,
           crossover_type=crossover_type,
           mutation_type=mutation_type,
           keep_elitism=1,
           K_tournament=3,
           random_mutation_max_val=2, 
           random_mutation_min_val=0,
           logger=logger,
           on_generation=on_generation,
           parallel_processing=None)

ga_instance.run()


best = ga_instance.best_solution()
solution, solution_fitness, solution_idx = ga_instance.best_solution()

print("Best solution (binary) : {solution}".format(solution=solution))
print("Best solution (decoded): {decoded}".format(decoded=decode_ind(solution)))

real_val = global_min + (1.0 / solution_fitness) - 0.000001
print("Function value of the best solution = {val}".format(val=real_val))
print("Global minimum expected = {min}".format(min=global_min))

ga_instance.best_solutions_fitness = [global_min + (1.0 / x) - 0.000001 for x in ga_instance.best_solutions_fitness]
ga_instance.plot_fitness()