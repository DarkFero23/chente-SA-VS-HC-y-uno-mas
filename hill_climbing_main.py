import time
import numpy as np
import pandas as pd
from tsp_utils import generar_tsp, fitness, get_neighbors, graficar_ruta_grafo_espacial, graficar_comparaciones

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

# Ejecutar Hill Climbing
tamaños = [10, 20, 50, 100, 200, 500]
resultados = []

for n in tamaños:
    tsp = generar_tsp(n)
    start = time.time()
    ruta, dist_total = hill_climbing(tsp)
    duracion = time.time() - start

    print(f"[Hill Climbing] n={n} | Tiempo: {duracion:.4f}s | Distancia: {dist_total}")
    resultados.append({
        "Algoritmo": "Hill Climbing",
        "n": n,
        "Ruta": ruta,
        "Distancia": dist_total,
        "Tiempo (s)": duracion,
        "Temp Inicial": "-",
        "Cooling Rate": "-",
        "T_min": "-"
    })

    graficar_ruta_grafo_espacial(ruta, "Hill Climbing", n, dist_total)

df_hc = pd.DataFrame(resultados)
graficar_comparaciones(df_hc, "Distancia")
graficar_comparaciones(df_hc, "Tiempo (s)")

print("\n🟢 Mejores rutas encontradas (menor distancia):")
print(df_hc.sort_values(by="Distancia").head())
