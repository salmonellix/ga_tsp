import mlrose
import numpy as np

# Create list of city coordinates
coords_list = [(0, 1), (3, 4), (6, 5), (7, 3), (15, 0), (12, 4), (14, 10), (9, 6),(7,9), (0, 10)]

# Initialize fitness function object using coords_list
fitness_coords = mlrose.TravellingSales(coords = coords_list)

problem_fit = mlrose.TSPOpt(length = 10, fitness_fn = fitness_coords, maximize=False)

best_state, best_fitness = mlrose.genetic_alg(problem_fit, random_state = 2)

print('The best state found is: ', best_state)

print('The fitness at the best state is: ', best_fitness)