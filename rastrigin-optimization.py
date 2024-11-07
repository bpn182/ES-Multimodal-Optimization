import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sphere function: A simple convex function used for optimization problems.
# The global minimum is at the origin (0,0,...,0) with a value of 0.
def sphere_function(x):
    return sum([xi**2 for xi in x])

# Ackley function: A widely used benchmark function for optimization algorithms.
# It has a nearly flat outer region and a large hole at the center.
# The global minimum is at the origin (0,0,...,0) with a value of 0.
def ackley(x):
    n = len(x)
    sum1 = sum([xi**2 for xi in x])
    sum2 = sum([np.cos(2*np.pi*xi) for xi in x])
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    return term1 + term2 + 20 + np.e

# Rastrigin function: A non-convex function used as a performance test problem for optimization algorithms.
# It has a large number of local minima, making it difficult for optimization algorithms to find the global minimum.
# The global minimum is at the origin (0,0,...,0) with a value of 0.
def rastrigin(x):
    A = 10
    n = len(x)
    return A * n + sum([xi**2 - A * np.cos(2 * np.pi * xi) for xi in x])

# Rosenbrock function: Also known as the Rosenbrock's valley or Rosenbrock's banana function.
# It is a non-convex function used to test the performance of optimization algorithms.
# The global minimum is inside a long, narrow, parabolic shaped flat valley.
# The global minimum is at (1,1,...,1) with a value of 0.
def rosenbrock(x):
    return sum([100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1)])

# Griewank function: A complex function used for testing optimization algorithms.
# It has many widespread local minima, making it difficult for optimization algorithms to find the global minimum.
# The global minimum is at the origin (0,0,...,0) with a value of 0.
def griewank(x):
    sum_term = sum([xi**2 for xi in x]) / 4000
    prod_term = np.prod([np.cos(xi / np.sqrt(i+1)) for i, xi in enumerate(x)])
    return sum_term - prod_term + 1

# Fitness function: First using rastrigin function
def fitness(x):
    return rastrigin(x)


def es_optimize(strategy, fitness_func, dimension, population_size_mu, offspring_size_lambda, sigma, max_generations):
    """
    This function implements the Evolution Strategy (ES) optimization algorithm.

    Parameters:
    strategy (str): The strategy to use for selection. Options are "mu_lambda", "mu_plus_lambda", "one_plus_one", "one_plus_one_1_fifth", "mu_over_mu_lambda".
    fitness_func (function): The fitness function to optimize.
    dimension (int): The number of dimensions in the problem.
    population_size_mu (int): The size of the parent population.
    offspring_size_lambda (int): The size of the offspring population.
    sigma (float): The standard deviation of the Gaussian noise added during mutation.
    max_generations (int): The maximum number of generations to run the algorithm for.

    Returns:
    best_solution (np.array): The best solution found.
    best_fitness (float): The fitness of the best solution.
    best_fitnesses (list): The fitness of the best individual at each generation.
    """
    # Initialize parent population
    parents = np.random.uniform(-5.5, 5.5, size=(population_size_mu, dimension))
    best_fitnesses = []

    for gen in range(max_generations):
        # Generate offspring
        offspring = []
        for _ in range(offspring_size_lambda):
            parent_idx = np.random.randint(population_size_mu)
            offspring_candidate = parents[parent_idx] + np.random.normal(0, sigma, dimension)
            offspring.append(offspring_candidate)

        # Convert offspring to a 2D numpy array
        offspring = np.array(offspring)

        # Evaluate fitness of offspring
        offspring_fitnesses = [fitness_func(ind) for ind in offspring]

        # Select individuals based on the strategy
        if strategy == "mu_lambda":
            # Select μ fittest individuals from offspring
            sorted_indices = np.argsort(offspring_fitnesses)
            parents = offspring[sorted_indices[:population_size_mu]]
        elif strategy == "mu_plus_lambda":
            # Combine parents and offspring
            combined = np.concatenate((parents, offspring), axis=0)
            # Evaluate fitness of combined population
            combined_fitnesses = [fitness_func(ind) for ind in combined]
            # Select μ fittest individuals from combined population
            sorted_indices = np.argsort(combined_fitnesses)
            parents = combined[sorted_indices[:population_size_mu]]
        elif strategy == "one_plus_one":
            # Select the fittest individual from parent and offspring
            combined = np.concatenate((parents[0][None, :], offspring), axis=0)
            combined_fitnesses = [fitness_func(ind) for ind in combined]
            sorted_indices = np.argsort(combined_fitnesses)
            parents[0] = combined[sorted_indices[0]]
        elif strategy == "one_plus_one_1_fifth":
            # Select the fittest individual from parent and offspring
            combined = np.concatenate((parents[0][None, :], offspring), axis=0)
            combined_fitnesses = [fitness_func(ind) for ind in combined]
            sorted_indices = np.argsort(combined_fitnesses)
            parents[0] = combined[sorted_indices[0]]
            # Adjust sigma based on the success rate
            success_rate = np.mean(np.array(combined_fitnesses) < fitness_func(parents[0]))
            if success_rate > 1/5:
                sigma /= 0.82
            elif success_rate < 1/5:
                sigma *= 0.82
        elif strategy == "mu_over_mu_lambda":
            # Select the fittest individual from μ randomly selected offspring
            selected_indices = np.random.choice(offspring_size_lambda, population_size_mu, replace=False)
            selected_offspring = offspring[selected_indices]
            selected_fitnesses = [fitness_func(ind) for ind in selected_offspring]
            sorted_indices = np.argsort(selected_fitnesses)
            parents[0] = selected_offspring[sorted_indices[0]]

        # Check termination criteria
        best_fitness = fitness_func(parents[0])
        best_fitnesses.append(best_fitness)

        if best_fitness < 1e-6:
            print(f"Converged after {gen} generations.")
            break

    # Return the best solution found
    best_solution = parents[0]
    return best_solution, best_fitness, best_fitnesses


def plot_fitness(best_fitnesses, strategy):
    plt.figure()
    plt.plot(best_fitnesses)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title(f"Best Fitness over Generations ({strategy} strategy)")
    plt.tight_layout()
    plt.show()

def plot_3d_rastrigin(best_solution,strategy):
    # Define a grid for the plot
    x = np.linspace(-5.12, 5.12, 100)
    y = np.linspace(-5.12, 5.12, 100)
    x, y = np.meshgrid(x, y)

    # Calculate the z values based on the Rastrigin function
    z = np.array(
        [rastrigin(np.array([xi, yi])) for xi, yi in zip(np.ravel(x), np.ravel(y))]
    )
    z = z.reshape(x.shape)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap="viridis")

    # Overlay the best solution found by the optimization algorithm
    ax.scatter(
        best_solution[0], best_solution[1], rastrigin(best_solution), color="r", s=100
    )

    ax.set_title(f"3D Plot of the Rastrigin Function ({strategy} strategy)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout()
    plt.show()


DIMENSION = 2  # Dimensionality of the problem
POPULATION_SIZE_MU = 20  # Population size
OFFSPRING_SIZE_LAMBDA = 100  # Offspring size
SIGMA = 0.1  # Mutation step size
MAX_GENERATIONS = 500  # Maximum number of generations

STRATEGY = [
    "mu_lambda",
    "mu_plus_lambda",
    "one_plus_one",
    "one_plus_one_1_fifth",
    "mu_over_mu_lambda",
]

# Run ES optimization for all strategies
for strategy in STRATEGY:
    print(f"\nRunning {strategy} ES:")
    best_solution, best_fitness, best_fitnesses = es_optimize(
        strategy,
        fitness,
        DIMENSION,
        POPULATION_SIZE_MU,
        OFFSPRING_SIZE_LAMBDA,
        SIGMA,
        MAX_GENERATIONS,
    )
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness: {best_fitness}\n")
    plot_fitness(best_fitnesses, strategy)
    print("\n")
    plot_3d_rastrigin(best_solution, strategy)