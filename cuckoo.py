# archivo: cuckoo_main.py
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma
from tsp_utils import fitness, graficar_ruta_grafo_espacial, graficar_comparaciones
from matrices import matrices

# Función Levy Flight para generar nuevos candidatos
def levy_flight(Lambda):
    sigma = (gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
             (gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.randn() * sigma
    v = np.random.randn()
    step = u / abs(v) ** (1 / Lambda)
    return step

# Algoritmo Cuckoo Search para TSP
def cuckoo_search_tsp(matriz_distancia, n_nests=25, max_iter=100, pa=0.25, alpha=1.0, Lambda=1.5):
    n_cities = len(matriz_distancia)
    nests = [np.random.permutation(n_cities) for _ in range(n_nests)]
    best_nest = nests[0]
    best_fitness = fitness(best_nest, matriz_distancia)

    for _ in range(max_iter):
        new_nests = np.copy(nests)
        for i in range(n_nests):
            _ = levy_flight(Lambda) * alpha
            new_nests[i] = np.random.permutation(n_cities)

            new_fitness = fitness(new_nests[i], matriz_distancia)
            if new_fitness < fitness(nests[i], matriz_distancia):
                nests[i] = new_nests[i]

            if new_fitness < best_fitness:
                best_nest = new_nests[i]
                best_fitness = new_fitness

        abandoned = np.random.rand(n_nests) < pa
        nests = [np.random.permutation(n_cities) if abandoned[i] else nests[i] for i in range(n_nests)]

    return best_nest, best_fitness

# Ejecutar Cuckoo Search con matriz compartida
resultados = []
tamaños = [10, 20, 50, 100, 200, 500]

for n in tamaños:
    tsp = matrices[n]
    start = time.time()
    ruta, dist_total = cuckoo_search_tsp(tsp, n_nests=25, max_iter=200)
    duracion = time.time() - start

    print(f"[Cuckoo Search] n={n} | Tiempo: {duracion:.4f}s | Distancia: {dist_total}")

    resultados.append({
        "Algoritmo": "Cuckoo Search",
        "n": n,
        "Ruta": ruta,
        "Distancia": dist_total,
        "Tiempo (s)": duracion,
        "Temp Inicial": "-",
        "Cooling Rate": "-",
        "T_min": "-"
    })

    graficar_ruta_grafo_espacial(ruta, "Cuckoo Search", n, dist_total)

# Crear DataFrame con resultados
df_cuckoo = pd.DataFrame(resultados)
graficar_comparaciones(df_cuckoo, "Distancia")
graficar_comparaciones(df_cuckoo, "Tiempo (s)")

print("\n🟢 Mejores rutas encontradas (menor distancia):")
print(df_cuckoo.sort_values(by="Distancia").head())

print("\n📊 Mejor resultado por cada tamaño:")
mejores_por_n = df_cuckoo.loc[df_cuckoo.groupby("n")["Distancia"].idxmin()]
print(mejores_por_n[["n", "Distancia", "Tiempo (s)"]])
