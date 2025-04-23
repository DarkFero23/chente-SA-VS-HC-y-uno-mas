import time
import numpy as np
import pandas as pd
from tsp_utils import generar_tsp, fitness, graficar_ruta_grafo_espacial, graficar_comparaciones
from matrices import matrices

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

# Ejecutar SA
tamaños = [10, 20, 50, 100, 200, 500]
variaciones_temp = [1000]

#variaciones_temp = [50, 100, 500, 1000]
cooling_rates = [0.9]
#cooling_rates = [0.9, 0.95, 0.99, 0.995]
t_min_values = [0.01, 0.0001]

resultados = []

for n in tamaños:
    tsp = matrices[n]  # 👈 esta es la línea clave
    for temp in variaciones_temp:
        for cooling in cooling_rates:
            for t_min in t_min_values:
                start = time.time()
                ruta, dist_total = simulated_annealing(tsp, temp=temp, cooling_rate=cooling, min_temp=t_min)
                duracion = time.time() - start

                print(f"[SA] n={n} | T0={temp} | CR={cooling} | Tmin={t_min} | Tiempo: {duracion:.4f}s | Distancia: {dist_total}")
                resultados.append({
                    "Algoritmo": "Simulated Annealing",
                    "n": n,
                    "Ruta": ruta,
                    "Distancia": dist_total,
                    "Tiempo (s)": duracion,
                    "Temp Inicial": temp,
                    "Cooling Rate": cooling,
                    "T_min": t_min
                })

                graficar_ruta_grafo_espacial(ruta, f"SA (T={temp}, CR={cooling})", n, dist_total)

df_sa = pd.DataFrame(resultados)
graficar_comparaciones(df_sa, "Distancia")
graficar_comparaciones(df_sa, "Tiempo (s)")

print("\n🟢 Mejores rutas encontradas (menor distancia):")
print(df_sa.sort_values(by="Distancia").head())
######################################################
print("\n🔍 Mejor resultado por cada tamaño:")
mejores_por_n = df_sa.loc[df_sa.groupby("n")["Distancia"].idxmin()]
print(mejores_por_n[["n", "Distancia", "Tiempo (s)", "Temp Inicial", "Cooling Rate", "T_min"]])