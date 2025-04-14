import math
import random
import copy
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, coords):
        self.coords = coords
        self.count = 0

    def eval(self, tour):
        self.count += 1
        return tour_length(tour, self.coords)

def parse_tsp_file(filename):
    coords = []
    with open(filename, "r") as f:
        lines = f.readlines()
    
    if any("NODE_COORD_SECTION" in line.upper() for line in lines):
        reading_coords = False
        for line in lines:
            line = line.strip()
            if "NODE_COORD_SECTION" in line.upper():
                reading_coords = True
                continue
            if reading_coords:
                if line.upper() == "EOF" or line == "":
                    break
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        coords.append((x, y))
                    except:
                        continue
    else:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                coords.append((x, y))
            except:
                continue
    return coords

def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def tour_length(tour, coords):
    total = 0.0
    n = len(tour)
    for i in range(n):
        city_current = coords[tour[i]]
        city_next = coords[tour[(i + 1) % n]]
        total += euclidean_distance(city_current, city_next)
    return total

def tournament_selection(population, fitnesses, t_size):
    """Selects an individual via tournament selection (minimization)."""
    indices = random.sample(range(len(population)), t_size)
    best_index = min(indices, key=lambda i: fitnesses[i])
    return population[best_index]

def order_crossover(parent1, parent2):
    """Performs Order Crossover (OX) to create one offspring."""
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b+1] = parent1[a:b+1]
    fill = [gene for gene in parent2 if gene not in child]
    child = [fill.pop(0) if gene is None else gene for gene in child]
    return child

def swap_mutation(tour, mutation_rate):
    """Performs swap mutation on a tour with a given mutation probability."""
    new_tour = tour.copy()
    for i in range(len(new_tour)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(new_tour) - 1)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

def two_opt_deterministic(tour, evaluator, best_history, global_best_cost, max_evals):
    """
    Performs a deterministic 2‑opt local search.
    
    Parameters:
      - tour: the starting tour.
      - evaluator: instance of Evaluator (to count evaluations).
      - best_history: list of (eval_count, best_cost) snapshots.
      - global_best_cost: current best cost from the EA.
      - max_evals: maximum number of allowed evaluations.
      
    The search stops immediately if the evaluation limit is reached.
    """
    if evaluator.count >= max_evals:
        return tour, evaluator.eval(tour)
    
    improved = True
    best_tour = tour
    best_distance = evaluator.eval(tour)
    
    if best_distance < global_best_cost:
        best_history.append((evaluator.count, best_distance))
        global_best_cost = best_distance
    
    while improved:
        if evaluator.count >= max_evals:
            return best_tour, best_distance
        
        improved = False
        for i in range(1, len(tour) - 1):
            for j in range(i + 1, len(tour)):
                if evaluator.count >= max_evals:
                    return best_tour, best_distance
                new_tour = best_tour[:i] + best_tour[i:j+1][::-1] + best_tour[j+1:]
                new_distance = evaluator.eval(new_tour)
                if new_distance < best_distance:
                    best_tour = new_tour
                    best_distance = new_distance
                    improved = True
                    if new_distance < global_best_cost:
                        best_history.append((evaluator.count, new_distance))
                        global_best_cost = new_distance
                    break
            if improved:
                break
    return best_tour, best_distance

def run_ea(coords, pop_size=50, mutation_rate=0.05, tournament_size=5, 
           use_local_search=False, max_evals=10000):
    """
    Runs an evolutionary algorithm for the TSP until a fixed number of fitness evaluations is reached.
    
    If use_local_search is True, a deterministic 2‑opt local search is applied to each offspring,
    and improvements are recorded immediately.
    
    Returns:
       - The best tour found.
       - Its cost.
       - A history list tracking (evaluation_count, best cost) snapshots.
    """
    evaluator = Evaluator(coords)
    num_cities = len(coords)
    population = [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]
    
    # Record an initial snapshot based on the starting population
    fitnesses = []
    for tour in population:
        f = evaluator.eval(tour)
        fitnesses.append(f)
    best_index = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
    best_cost = fitnesses[best_index]
    best_solution = population[best_index]
    best_history = [(evaluator.count, best_cost)]
    
    generation = 0
    while evaluator.count < max_evals:
        new_population = []
        while len(new_population) < pop_size and evaluator.count < max_evals:
            parent1 = tournament_selection(population, fitnesses, tournament_size)
            parent2 = tournament_selection(population, fitnesses, tournament_size)
            child = order_crossover(parent1, parent2)
            child = swap_mutation(child, mutation_rate)
            
            if use_local_search and evaluator.count < max_evals:
                child, local_best_distance = two_opt_deterministic(
                    child, evaluator, best_history, best_cost, max_evals
                )
                if local_best_distance < best_cost:
                    best_cost = local_best_distance
                    best_solution = child
            
            new_population.append(child)
        population = new_population
        
        # Re-evaluate population fitnesses (or update using EA operators)
        fitnesses = []
        for tour in population:
            if evaluator.count >= max_evals:
                break
            f = evaluator.eval(tour)
            fitnesses.append(f)
        if fitnesses:
            gen_best_index = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
            if fitnesses[gen_best_index] < best_cost:
                best_cost = fitnesses[gen_best_index]
                best_solution = population[gen_best_index]
                best_history.append((evaluator.count, best_cost))
        
        generation += 1
        print(f"Generation {generation}: {evaluator.count} evals, Best tour length = {best_cost:.2f}")
    
    return best_solution, best_cost, best_history

def interpolate_history(history, grid):
    """
    Given a history (list of (eval, cost) sorted in increasing eval),
    returns a list of best cost values on the provided grid.
    For each grid point, the cost is the last recorded cost with eval <= grid point.
    """
    interp = []
    last_val = None
    j = 0
    n = len(history)
    for x in grid:
        while j < n and history[j][0] <= x:
            last_val = history[j][1]
            j += 1
        if last_val is None:
            # if no snapshot exists, use the first recorded value
            interp.append(history[0][1])
        else:
            interp.append(last_val)
    return interp

def average_histories(histories, grid):
    """
    Given a list of histories, each interpolated on the same grid,
    return a list of average cost values for each grid point.
    """
    interpolated = [interpolate_history(h, grid) for h in histories]
    avg = [sum(vals)/len(vals) for vals in zip(*interpolated)]
    return avg

def run_multiple_experiments(coords, num_runs, use_local_search, max_evals, pop_size, mutation_rate, tournament_size):
    """
    Runs the EA (or memetic EA if use_local_search is True) for a given number of independent runs,
    returns the list of histories (one per run).
    """
    histories = []
    final_costs = []
    for run in range(num_runs):
        print(f"Starting run {run+1} ...")
        _, final_cost, history = run_ea(coords, pop_size=pop_size, mutation_rate=mutation_rate,
                                        tournament_size=tournament_size, use_local_search=use_local_search,
                                        max_evals=max_evals)
        histories.append(history)
        final_costs.append(final_cost)
    return histories, final_costs

def main():
    # Parameters for experiments
    pop_size = 50
    mutation_rate = 0.01
    tournament_size = 5
    max_evals = 50000
    num_runs = 10  # number of independent runs to average over
    grid_step = 500  # evaluations between grid points
    grid = list(range(0, max_evals+1, grid_step))
    
    # Dataset 1: Plain coordinate matrix (file-tsp.txt)
    tsp_file1 = "file-tsp.txt"
    coords1 = parse_tsp_file(tsp_file1)
    print(f"Loaded dataset 1 (file-tsp.txt) with {len(coords1)} cities.")

    # Dataset 2: TSPLIB instance (fnl4461.txt)
    tsp_file2 = "fnl4461.tsp"
    coords2 = parse_tsp_file(tsp_file2)
    print(f"Loaded dataset 2 (fnl4461.txt) with {len(coords2)} cities.")
    small_instance_size = 50
    if len(coords2) > small_instance_size:
        print(f"Restricting dataset 2 to the first {small_instance_size} cities.")
        coords2 = coords2[:small_instance_size]
    
    # --- Run experiments on each dataset for both methods ---
    print("\n--- Running experiments on Dataset 1 (file-tsp.txt) ---")
    # Simple EA on Dataset 1
    histories_ea1, final_costs_ea1 = run_multiple_experiments(coords1, num_runs, use_local_search=False,
                                                               max_evals=max_evals, pop_size=pop_size,
                                                               mutation_rate=mutation_rate,
                                                               tournament_size=tournament_size)
    # Memetic EA on Dataset 1
    histories_ma1, final_costs_ma1 = run_multiple_experiments(coords1, num_runs, use_local_search=True,
                                                               max_evals=max_evals, pop_size=pop_size,
                                                               mutation_rate=mutation_rate,
                                                               tournament_size=tournament_size)
    
    print("\n--- Running experiments on Dataset 2 (small instance from fnl4461.txt) ---")
    # Simple EA on Dataset 2
    histories_ea2, final_costs_ea2 = run_multiple_experiments(coords2, num_runs, use_local_search=False,
                                                               max_evals=max_evals, pop_size=pop_size,
                                                               mutation_rate=mutation_rate,
                                                               tournament_size=tournament_size)
    # Memetic EA on Dataset 2
    histories_ma2, final_costs_ma2 = run_multiple_experiments(coords2, num_runs, use_local_search=True,
                                                               max_evals=max_evals, pop_size=pop_size,
                                                               mutation_rate=mutation_rate,
                                                               tournament_size=tournament_size)
    
    # Average histories over runs for each case
    avg_ea1 = average_histories(histories_ea1, grid)
    avg_ma1 = average_histories(histories_ma1, grid)
    avg_ea2 = average_histories(histories_ea2, grid)
    avg_ma2 = average_histories(histories_ma2, grid)
    
    # Plot for Dataset 1
    plt.figure(figsize=(10, 6))
    plt.plot(grid, avg_ea1, label="Simple EA (Dataset 1)", marker="o", markersize=3)
    plt.plot(grid, avg_ma1, label="Memetic EA (Dataset 1)", marker="s", markersize=3)
    plt.xlabel("Number of Fitness Evaluations")
    plt.ylabel("Average Best Tour Length")
    plt.title("Average Performance on Dataset 1 (file-tsp.txt)")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, max_evals)
    plt.show()
    
    # Plot for Dataset 2
    plt.figure(figsize=(10, 6))
    plt.plot(grid, avg_ea2, label="Simple EA (Dataset 2)", marker="o", markersize=3)
    plt.plot(grid, avg_ma2, label="Memetic EA (Dataset 2)", marker="s", markersize=3)
    plt.xlabel("Number of Fitness Evaluations")
    plt.ylabel("Average Best Tour Length")
    plt.title("Average Performance on Dataset 2 (Small fnl4461 Instance)")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, max_evals)
    plt.show()
    
    # Print final cost averages for reference
    print("\nFinal Average Results:")
    print(f"Dataset 1 (Simple EA): {sum(final_costs_ea1)/len(final_costs_ea1):.2f}")
    print(f"Dataset 1 (Memetic EA): {sum(final_costs_ma1)/len(final_costs_ma1):.2f}")
    print(f"Dataset 2 (Simple EA): {sum(final_costs_ea2)/len(final_costs_ea2):.2f}")
    print(f"Dataset 2 (Memetic EA): {sum(final_costs_ma2)/len(final_costs_ma2):.2f}")

if __name__ == "__main__":
    main()
