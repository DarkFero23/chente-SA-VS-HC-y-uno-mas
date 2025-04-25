# archivo: simulated_annealing_main.py
import os
import time
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
from tsp_utils import generar_tsp, fitness, graficar_ruta_grafo_espacial
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

# Hiperparámetros para experimentar
tamaños = [10, 20, 50, 100, 200, 500]
variaciones_temp = [50, 100, 500, 1000, 2000]
cooling_rates = [0.9, 0.95, 0.99, 0.995]
t_min_values = [0.01, 0.0001, 0.1]

resultados = []

with open("resultados_sa.txt", "w", encoding="utf-8") as file:
    for n in tamaños:
        tsp = matrices[n]
        for temp in variaciones_temp:
            for cooling in cooling_rates:
                for t_min in t_min_values:
                    start = time.time()
                    ruta, dist_total = simulated_annealing(tsp, temp=temp, cooling_rate=cooling, min_temp=t_min)
                    duracion = time.time() - start

                    encabezado = f"SA | T0={temp}, CR={cooling}, Tmin={t_min}"
                    resultado_str = f"[SA] n={n} | {encabezado} | Tiempo: {duracion:.4f}s | Distancia: {dist_total}"
                    print(resultado_str)
                    file.write(resultado_str + "\n")

                    resultados.append({
                        "Algoritmo": encabezado,
                        "n": n,
                        "Ruta": ruta,
                        "Distancia": dist_total,
                        "Tiempo (s)": duracion,
                        "Temp Inicial": temp,
                        "Cooling Rate": cooling,
                        "T_min": t_min
                    })

# Crear DataFrame con resultados
df_sa = pd.DataFrame(resultados)

mejores_globales = df_sa.sort_values(by="Distancia").head()
mejores_por_n = df_sa.loc[df_sa.groupby("n")["Distancia"].idxmin()]

print("\n🟢 Mejores rutas encontradas (menor distancia):")
print(mejores_globales)

print("\n📊 Mejor resultado por cada tamaño:")
print(mejores_por_n[["n", "Distancia", "Tiempo (s)", "Temp Inicial", "Cooling Rate", "T_min"]])

with open("resultados_sa.txt", "a", encoding="utf-8") as file:
    file.write("\n🟢 Mejores rutas encontradas (menor distancia):\n")
    file.write(mejores_globales.to_string(index=False) + "\n")
    file.write("\n📊 Mejor resultado por cada tamaño:\n")
    file.write(mejores_por_n[["n", "Distancia", "Tiempo (s)", "Temp Inicial", "Cooling Rate", "T_min"]].to_string(index=False) + "\n")

# Crear carpeta si no existe y guardar solo los mejores recorridos
output_folder = "mejores_rutas_sa"
os.makedirs(output_folder, exist_ok=True)

for _, fila in mejores_por_n.iterrows():
    nombre_archivo = os.path.join(output_folder, f"sa_n{fila['n']}.png")
    np.random.seed(fila['n'] + 1000)  # Cambio para posiciones distintas
    graficar_ruta_grafo_espacial(
        fila["Ruta"],
        fila["Algoritmo"],
        fila["n"],
        fila["Distancia"],
        save_path=nombre_archivo,
        seed=fila["n"] + 1000
    )