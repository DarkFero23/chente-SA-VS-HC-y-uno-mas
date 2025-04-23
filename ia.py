import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import random

def generar_tsp(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.randint(10, 500, size=(n, n)) * (1 - np.eye(n, dtype=int))

def fitness(sol, matriz):
    distancia = 0
    for i in range(len(sol) - 1):
        distancia += matriz[sol[i]][sol[i + 1]]
    distancia += matriz[sol[-1]][sol[0]]  # cerrar ciclo
    return distancia

def get_neighbors(sol):
    vecinos = []
    for i in range(len(sol)):
        for j in range(i + 1, len(sol)):
            vecino = sol.copy()
            vecino[i], vecino[j] = vecino[j], vecino[i]
            vecinos.append(vecino)
        return vecinos

# Algoritmo Hill Climbing
def hill_climbing(tsp):
    current_solution = list(range(len(tsp)))
    np.random.shuffle(current_solution)
    while True:
        neighbors = get_neighbors(current_solution)
        best_neighbor = min(neighbors, key=lambda x: fitness(x, tsp))
        if fitness(best_neighbor, tsp) >= fitness(current_solution, tsp):
            break
        current_solution = best_neighbor
    return current_solution, fitness(current_solution, tsp)

# Simulated Annealing
def simulated_annealing(tsp, temp=1000, cooling_rate=0.995, min_temp=0.01):
    current_solution = list(range(len(tsp)))
    np.random.shuffle(current_solution)
    best_solution = current_solution[:]
    current_fitness = fitness(current_solution, tsp)
    best_fitness = current_fitness

    while temp > min_temp:
        i, j = np.random.randint(0, len(tsp), size=2)
        neighbor = current_solution[:]
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

        new_fitness = fitness(neighbor, tsp)
        delta = new_fitness - current_fitness

        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            current_solution = neighbor
            current_fitness = new_fitness

            if current_fitness < best_fitness:
                best_solution = current_solution[:]
                best_fitness = current_fitness

        temp *= cooling_rate

    return best_solution, best_fitness

# Visualizar ruta como grafo
import matplotlib.pyplot as plt

def graficar_ruta_grafo_espacial(ruta, algoritmo, n, distancia):
    # Posiciones aleatorias fijas por ejecución (para que no se crucen tanto)
    np.random.seed(42)
    posiciones = {i: (np.random.rand(), np.random.rand()) for i in ruta}

    plt.figure(figsize=(10, 8))
    
    # Dibujar nodos
    for nodo, (x, y) in posiciones.items():
        plt.scatter(x, y, s=80, color="skyblue", zorder=2)
        plt.text(x, y + 0.01, str(nodo), fontsize=8, ha='center', zorder=3)

    # Dibujar ruta
    for i in range(len(ruta)):
        origen = ruta[i]
        destino = ruta[(i + 1) % len(ruta)]  # para cerrar el ciclo
        x_values = [posiciones[origen][0], posiciones[destino][0]]
        y_values = [posiciones[origen][1], posiciones[destino][1]]
        plt.plot(x_values, y_values, 'k-', linewidth=1, alpha=0.6, zorder=1)

    plt.title(f"Ruta en grafo espacial | {algoritmo} | n={n} | Distancia: {distancia}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Evaluación con logging detallado
def evaluar_algoritmos_con_detalle(tamaños, variaciones_temp, cooling_rates, t_min_values):
    resultados = []

    for n in tamaños:
        tsp = generar_tsp(n)

        print(f"\n---- Evaluando para n = {n} ciudades ----")

        # Hill Climbing
        start = time.time()
        ruta_hc, dist_hc = hill_climbing(tsp)
        tiempo_hc = time.time() - start
        print(f"[Hill Climbing] n={n} | Tiempo: {tiempo_hc:.4f}s | Distancia: {dist_hc}")
        resultados.append({
            "Algoritmo": "Hill Climbing",
            "n": n,
            "Ruta": ruta_hc,
            "Distancia": dist_hc,
            "Tiempo (s)": tiempo_hc,
            "Temp Inicial": "-",
            "Cooling Rate": "-",
            "T_min": "-",
            "TSP": tsp
        })
        graficar_ruta_grafo_espacial(ruta_hc, "Hill Climbing", n, dist_hc)

        # Simulated Annealing con distintas combinaciones
        for temp in variaciones_temp:
            for cooling in cooling_rates:
                for t_min in t_min_values:
                    start = time.time()
                    ruta_sa, dist_sa = simulated_annealing(tsp, temp=temp, cooling_rate=cooling, min_temp=t_min)
                    tiempo_sa = time.time() - start
                    print(f"[Simulated Annealing] n={n} | T0={temp} | CR={cooling} | Tmin={t_min} | Tiempo: {tiempo_sa:.4f}s | Distancia: {dist_sa}")
                    resultados.append({
                        "Algoritmo": "Simulated Annealing",
                        "n": n,
                        "Ruta": ruta_sa,
                        "Distancia": dist_sa,
                        "Tiempo (s)": tiempo_sa,
                        "Temp Inicial": temp,
                        "Cooling Rate": cooling,
                        "T_min": t_min,
                        "TSP": tsp
                    })
                    graficar_ruta_grafo_espacial(ruta_sa, f"SA (T={temp}, CR={cooling})", n, dist_sa)



    return pd.DataFrame(resultados)

# Parámetros
tamaños = [10, 20, 50, 100, 200, 500]  
variaciones_temp = [50,100, 500, 1000]
cooling_rates = [0.9, 0.95, 0.99, 0.995]
t_min_values = [0.01, 0.0001]

# Ejecutar evaluación
df_resultados = evaluar_algoritmos_con_detalle(tamaños, variaciones_temp, cooling_rates, t_min_values)

# Gráficas de comparación
import seaborn as sns

def graficar_distancias(df):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="n", y="Distancia", hue="Algoritmo", marker="o", style="Temp Inicial")
    plt.title("Comparación de distancia total por algoritmo")
    plt.xlabel("Número de ciudades (n)")
    plt.ylabel("Distancia total")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def graficar_tiempos(df):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="n", y="Tiempo (s)", hue="Algoritmo", marker="s", style="Temp Inicial")
    plt.title("Comparación de tiempo de ejecución por algoritmo")
    plt.xlabel("Número de ciudades (n)")
    plt.ylabel("Tiempo de ejecución (segundos)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

graficar_distancias(df_resultados)
graficar_tiempos(df_resultados)

from IPython.display import display
display(df_resultados)

# Mostrar las 5 mejores soluciones encontradas (menor distancia)
print("\n🟢 Mejores rutas encontradas (menor distancia):")
mejores_resultados = df_resultados.sort_values(by="Distancia").head()
display(mejores_resultados)