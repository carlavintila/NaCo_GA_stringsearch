import random
from collections import Counter, defaultdict
import math
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import logomaker
import numpy as np

TARGET = 'NaturalComputing'
ALPHABET = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'''
POPULATION_SIZE = 200
CROSSOVER_PROBABILITY = 1.0   
G_MAX = 100       

random.seed(42)
stats_rng = random.Random(12345)  

def random_string(length):
    return ''.join(random.choices(ALPHABET, k=length))

def compute_fitness(candidate, target):
    return sum(c1 == c2 for c1, c2 in zip(candidate, target))

def tournament_selection(population, fitnesses, k):
    contenders = random.sample(list(zip(population, fitnesses)), k)
    return max(contenders, key=lambda x: x[1])[0]

def crossover(parent1, parent2, crossover_probability):
    if random.random() < crossover_probability:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
    else:
        child1, child2 = parent1, parent2
    return child1, child2

def mutate(individual, mutation_rate):
    return ''.join(
        c if random.random() > mutation_rate else random.choice(ALPHABET)
        for c in individual
    )
    
def compute_hamming(population, sample_size=50, rng=random):
    sample = rng.sample(population, min(sample_size, len(population)))
    total_distance = 0
    comparisons = 0
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            dist = sum(c1 != c2 for c1, c2 in zip(sample[i], sample[j]))
            total_distance += dist
            comparisons += 1
    return total_distance / comparisons if comparisons > 0 else 0

def compute_entropy(population):
    L = len(population[0])
    entropies = []
    for i in range(L):
        column = [individual[i] for individual in population]
        counts = Counter(column)
        total = sum(counts.values())
        entropy = -sum((count / total) * math.log(count / total + 1e-9, 2) for count in counts.values())
        entropies.append(entropy)
    return entropies

def build_freq_matrix(population):
    L = len(population[0])
    alphabet = list(ALPHABET)
    matrix = [{char: 0 for char in alphabet} for _ in range(L)]

    for individual in population:
        for i, char in enumerate(individual):
            matrix[i][char] += 1

    df = pd.DataFrame(matrix)
    df = df.fillna(0)
    df = df.div(df.sum(axis=1), axis=0) 
    return df

def genetic_algorithm(k, mutation_rate, verbose = True, show_diversity = False, target = TARGET, population_size = POPULATION_SIZE, crossover_probability = CROSSOVER_PROBABILITY, max_generations = G_MAX):
    length = len(target)
    
    # Initial population
    population = [random_string(length) for _ in range(population_size)]
    
    diversity_over_time = []
    entropy_over_time = []
    
    for generation in range(max_generations):
        fitnesses = [compute_fitness(ind, target) for ind in population]
        
        if ((generation % 10 == 0 or generation == max_generations - 1) and show_diversity==True):
            diversity = compute_hamming(population, rng=stats_rng)
            entropy = compute_entropy(population)
            diversity_over_time.append((generation, diversity))
            entropy_over_time.append((generation, entropy))

        # Check for solution
        if max(fitnesses) == length:
            best = population[fitnesses.index(max(fitnesses))]
            if verbose:
                print(f"Found solution in generation {generation}: {best}")
            return population, generation, diversity_over_time, entropy_over_time

        # New population
        new_population = []
        for _ in range(population_size // 2):
            parent1 = tournament_selection(population, fitnesses, k)
            parent2 = tournament_selection(population, fitnesses, k)
            child1, child2 = crossover(parent1, parent2, crossover_probability)
            new_population.extend([
                mutate(child1, mutation_rate),
                mutate(child2, mutation_rate)
            ])

        population = new_population  # Replace population 

    if verbose:
        print(f"No exact solution found. Best = {population[fitnesses.index(max(fitnesses))]} in gen {generation}")
    return population, -1, diversity_over_time, entropy_over_time

def experiments123():
    k = 2
    L = len(TARGET)
    mutation_rates = [0 , 1 / L, 3 / L]
    results = defaultdict(list)
    avg_generations = []
    for mu in mutation_rates:
        generations = []
        for _ in range(10):
            _, gen, _ = genetic_algorithm(k, mu, verbose=True, max_generations=100, show_diversity=True)
            results[mu].append(gen if gen != -1 else G_MAX)
            generations.append(gen if gen != -1 else G_MAX)
        avg_generations.append(sum(generations) / len(generations))
    return results, avg_generations

def show_results123():
    results, avg = experiments123()
    print(avg)

    plot_data = []
    for mu, gens in results.items():
        for gen in gens:
            plot_data.append({'Mutation Rate': f'{mu:.2f}', 'Generations': gen})
    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 6))
    sns.stripplot(data=plot_df, x='Generations', y='Mutation Rate', hue='Mutation Rate', palette='Set2')
    plt.title(r'Distribution of $t_{\mathrm{finish}}$ for different Mutation Rates', fontsize=16)
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Mutation Rate (µ)', fontsize=14)
    plt.tight_layout()
    plt.show()

def run_experiment56(k):
    L = len(TARGET)
    mu_values = list(np.linspace(0, 2 / L, 10))
    all_generations = {}
    avg_generations = []

    for mu in mu_values:
        generations = []
        for _ in range(10):
            result = genetic_algorithm(k, mu, verbose=False, show_diversity=True)
            gen = result[1] 
            generations.append(gen if gen != -1 else G_MAX)
        all_generations[mu] = generations
        avg_generations.append(sum(generations) / len(generations))

    print(avg_generations)
    plot_data = []
    for mu, gens in all_generations.items():
        for gen in gens:
            plot_data.append({'Mutation Rate': f'{mu:.2f}', 'Generations': gen})

    plot_df = pd.DataFrame(plot_data)
    plt.figure(figsize=(10, 6))
    sns.swarmplot(data=plot_df, x='Generations', y='Mutation Rate', hue='Mutation Rate', palette='Set2')
    plt.title(r'Distribution of $t_{\mathrm{finish}}$ for different Mutation Rates', fontsize=16)
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Mutation Rate (µ)', fontsize=14)
    plt.tight_layout()
    plt.show()
    return all_generations, avg_generations

def hamming(k):
    L = len(TARGET)
    mutation_rates = [0 , 1 / L, 3 / L]
    results = defaultdict(list)
    all_diversities = defaultdict(list)
    
    for mu in mutation_rates:
        generations = []
        diversity_runs = defaultdict(list)
        
        for _ in range(10):
            _, gen, div, _ = genetic_algorithm(
                k, mu, verbose=False, max_generations=100, show_diversity=True
            )
            generations.append(gen if gen != -1 else G_MAX)
            results[mu].append(gen if gen != -1 else G_MAX)
            
            for g, d in div:
                diversity_runs[g].append(d)

        avg_diversity = [(g, sum(d_list)/len(d_list)) for g, d_list in sorted(diversity_runs.items())]
        all_diversities[mu] = avg_diversity
    return all_diversities

def plot_hamming(all_diversities):
    plt.figure(figsize=(10, 6))
    
    for mu, diversity_data in all_diversities.items():
        generations = [g for g, _ in diversity_data]
        avg_hamming = [d for _, d in diversity_data]
        plt.plot(generations, avg_hamming, label=f"Mutation rate = {mu:.4f}")
    
    plt.title("Average Hamming Distance Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Average Hamming Distance")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_entropy(k, num_runs=10):
    L = len(TARGET)
    mutation_rates = [0, 1 / L, 3 / L]

    for mu in mutation_rates:
        all_initial_entropies = []
        all_final_entropies = []

        for _ in range(num_runs):
            _, _, _, entropy_over_time = genetic_algorithm(
                k, mu, verbose=False, max_generations=100, show_diversity=True
            )

            if not entropy_over_time:
                continue

            first_gen = entropy_over_time[0][1]
            last_gen = entropy_over_time[-1][1]

            all_initial_entropies.append(first_gen)
            all_final_entropies.append(last_gen)

        if not all_initial_entropies or not all_final_entropies:
            print(f"Not enough data for µ={mu}")
            continue

        avg_initial_entropy = np.mean(all_initial_entropies, axis=0)
        avg_final_entropy = np.mean(all_final_entropies, axis=0)

        # Plot initial entropy
        plt.figure(figsize=(4, 3))
        plt.bar(range(len(avg_initial_entropy)), avg_initial_entropy, color='#6495ed')
        plt.title(f"Avg Entropy (First Gen) for µ={mu:.3f}")
        plt.xlabel("Position in String")
        plt.ylabel("Shannon Entropy")
        plt.xticks(np.arange(0, 17, 1))
        plt.ylim(0, 5)
        plt.tight_layout()
        plt.show()

        # Plot final entropy
        plt.figure(figsize=(4, 3))
        plt.bar(range(len(avg_final_entropy)), avg_final_entropy, color='#6495ed')
        plt.title(f"Avg Entropy (Final Gen) for µ={mu:.3f}")
        plt.xlabel("Position in String")
        plt.ylabel("Shannon Entropy")
        plt.xticks(np.arange(0, 17, 1))
        plt.ylim(0, 5)
        plt.tight_layout()
        plt.show()

def plot_seqlogo(num_runs=10):
    k = 2
    L = len(TARGET)
    mutation_rates = [0, 1 / L, 3 / L]

    for mu in mutation_rates:
        final_gen_pools = []

        for _ in range(num_runs):
            initial_population = [random_string(L) for _ in range(POPULATION_SIZE)]
            
            for generation in range(G_MAX):
                fitnesses = [compute_fitness(ind, TARGET) for ind in population]
                if max(fitnesses) == L:
                    break  # early stopping if solution found

                new_population = []
                for _ in range(POPULATION_SIZE // 2):
                    p1 = tournament_selection(population, fitnesses, k)
                    p2 = tournament_selection(population, fitnesses, k)
                    c1, c2 = crossover(p1, p2, CROSSOVER_PROBABILITY)
                    new_population.append(mutate(c1, mu))
                    new_population.append(mutate(c2, mu))
                population = new_population

            final_gen_pools.extend(population)

        # --- Final Generation Logo ---
        final_df = build_freq_matrix(final_gen_pools)
        logomaker.Logo(final_df,  color_scheme='dodgerblue')
        plt.title(f"Sequence Logo (Final Gen) for µ={mu:.4f}")
        plt.xlabel("Position")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()