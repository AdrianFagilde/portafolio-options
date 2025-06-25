import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
#%% md
## Preparación de la Data
#%%
# Carga y preparación de datos
file_path = "precios_semanales.xlsx"
df = pd.read_excel(file_path, sheet_name=0)
df.set_index(df.columns[0], inplace=True)
df.index = pd.to_datetime(df.index)

# Cálculo de retornos y estadísticas
returns = df.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()
#%% md
# PUNTO 1: MARKOWITZ
#%%
# Funciones básicas requeridas por Markowitz
def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(weights.T @ cov_matrix @ weights)


def portfolio_return(weights, mean_returns):
    return np.sum(mean_returns * weights)


# Generación de frontera eficiente
def efficient_frontier(mean_returns, cov_matrix, num_points=100):
    target_returns = np.linspace(min(mean_returns), max(mean_returns), num_points)
    risks = []

    for target in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: portfolio_return(w, mean_returns) - target}
        )
        bounds = [(0, 1) for _ in range(len(mean_returns))]
        init_weights = np.ones(len(mean_returns)) / len(mean_returns)

        opt = minimize(
            portfolio_volatility,
            init_weights,
            args=(cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        risks.append(opt.fun)

    return target_returns, risks
#%%
# Encontrar portafolio de mínimo riesgo global
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = [(0, 1) for _ in range(len(mean_returns))]
init_weights = np.ones(len(mean_returns)) / len(mean_returns)

min_risk_result = minimize(
    portfolio_volatility,
    init_weights,
    args=(cov_matrix),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# Resultados
optimal_weights = min_risk_result.x
print("Pesos óptimos (mínimo riesgo):")
for ticker, weight in zip(df.columns, optimal_weights):
    print(f"{ticker}: {weight:.4f}")

#%%
# Frontera eficiente
target_returns, risks = efficient_frontier(mean_returns, cov_matrix)

# Gráfico (solo frontera eficiente como pide el proyecto)
plt.figure(figsize=(10, 6))
plt.plot(risks, target_returns, 'b-', linewidth=2)
plt.scatter(portfolio_volatility(optimal_weights, cov_matrix),
            portfolio_return(optimal_weights, mean_returns),
            c='r', marker='*', s=300, label='Portafolio Mínimo Riesgo')
plt.xlabel('Riesgo (Desviación Estándar)')
plt.ylabel('Retorno Esperado')
plt.title('Frontera Eficiente de Markowitz')
plt.legend()
plt.grid(True)
plt.show()

#%% md
# PUNTO 2: ALGORITMO GENETICO
#%%
# función para δ² (varianza)
def portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights


# Generates set of random numbers whose sum is equal to 1
def chromosome(n):
    ch = np.random.rand(n)
    return ch / sum(ch)


def fitness_function(child, mean_returns, cov_matrix):
    expected_return = portfolio_return(child, mean_returns)
    variance = portfolio_variance(child, cov_matrix)
    return expected_return / variance  # Maximizar este ratio


def select_elite_population(population, frac=0.3):
    population = sorted(population, key=lambda x: fitness_function(x, mean_returns, cov_matrix), reverse=True)
    percentage_elite_idx = int(np.floor(len(population) * frac))
    return population[:percentage_elite_idx]


# Randomy choosen elements of a chromosome are swapped
def mutation(parent):
    child = parent.copy()
    n = np.random.choice(range(6), 2)

    while (n[0] == n[1]):
        n = np.random.choice(range(6), 2)

    child[n[0]], child[n[1]] = child[n[1]], child[n[0]]
    return child


def crossover(parent1, parent2, mean_returns, cov_matrix):
    ff1 = fitness_function(parent1, mean_returns, cov_matrix)
    ff2 = fitness_function(parent2, mean_returns, cov_matrix)
    diff = parent1 - parent2
    beta = np.random.rand()
    if ff1 > ff2:
        child1 = parent1 + beta * diff
        child2 = parent2 - beta * diff
    else:
        child2 = parent1 + beta * diff
        child1 = parent2 - beta * diff
    return child1, child2


# Generates new population from elite population with mutation probability as 0.4 and crossover as 0.6.
def next_generation(pop_size, elite, n, mean_returns, cov_matrix):
    new_population = []
    elite_range = range(len(elite))

    while len(new_population) < pop_size:
        if len(new_population) > 2 * pop_size / 3:  # In the final stages mutation frequency is decreased.
            mutate_or_crossover = np.random.choice([0, 1], p=[0.9, 0.1])
        else:
            mutate_or_crossover = np.random.choice([0, 1], p=[0.4, 0.6])

        if mutate_or_crossover:
            indx = np.random.choice(elite_range)
            new_population.append(mutation(elite[indx]))
        else:
            p1_idx, p2_idx = np.random.choice(elite_range, 2)
            c1, c2 = crossover(elite[p1_idx], elite[p2_idx], mean_returns, cov_matrix)

            # Validación de pesos negativos (versión mejorada)
            if any(g < 0 for g in c1) or any(g < 0 for g in c2):
                p1_idx, p2_idx = np.random.choice(elite_range, 2)
                c1, c2 = crossover(elite[p1_idx], elite[p2_idx], mean_returns, cov_matrix)

            new_population.extend([c1, c2])

    new_population = [np.clip(individual, 0, 1) / np.sum(np.clip(individual, 0, 1)) for individual in new_population]
    return new_population


# Paso 1: Población inicial
n = 9
pop_size = 100  # initial population = 100
population = np.array([chromosome(n) for _ in range(pop_size)])

# Paso 2: Selección de la población élite
elite = select_elite_population(population)

iteration = 0

while iteration <= 30:
    print('Iteration:', iteration)
    population = next_generation(100, elite, n, mean_returns, cov_matrix)
    elite = select_elite_population(population)

    iteration += 1

print('Portaflio despues de las iteraciones:\n')
[print(df.columns[i], ':', elite[0][i]) for i in list(range(n))]

print("Comprobacion de igual a 1:", sum(elite[0]))

