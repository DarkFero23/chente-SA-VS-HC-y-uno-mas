# archivo: cuckoo_main.py
import os
import time
import numpy as np
import pandas as pd
from scipy.special import gamma
from tsp_utils import fitness, graficar_ruta_grafo_espacial
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

# Hiperparámetros para experimentar
n_nests_values = [10, 25, 50]
max_iter_values = [50, 200, 500]
pa_values = [0.1, 0.25, 0.6]
alpha_values = [0.5, 1.0, 2.0]
lambda_values = [1.0, 1.5, 2.0]

# Ejecutar Cuckoo Search con matriz compartida
resultados = []
tamaños = [10, 20, 50, 100, 200, 500]

with open("resultados_cuckoo.txt", "w", encoding="utf-8") as file:
    for n in tamaños:
        tsp = matrices[n]
        for n_nests in n_nests_values:
            for max_iter in max_iter_values:
                for pa in pa_values:
                    for alpha in alpha_values:
                        for Lambda in lambda_values:
                            start = time.time()
                            ruta, dist_total = cuckoo_search_tsp(tsp, n_nests=n_nests, max_iter=max_iter, pa=pa, alpha=alpha, Lambda=Lambda)
                            duracion = time.time() - start

                            encabezado = f"Cuckoo | nests={n_nests}, iter={max_iter}, pa={pa}, alpha={alpha}, lambda={Lambda}"
                            resultado_str = f"[Cuckoo Search] n={n} | {encabezado} | Tiempo: {duracion:.4f}s | Distancia: {dist_total}"
                            print(resultado_str)
                            file.write(resultado_str + "\n")

                            resultados.append({
                                "Algoritmo": encabezado,
                                "n": n,
                                "Ruta": ruta,
                                "Distancia": dist_total,
                                "Tiempo (s)": duracion,
                                "n_nests": n_nests,
                                "max_iter": max_iter,
                                "pa": pa,
                                "alpha": alpha,
                                "lambda": Lambda
                            })

# Crear DataFrame con resultados
df_cuckoo = pd.DataFrame(resultados)

mejores_globales = df_cuckoo.sort_values(by="Distancia").head()
mejores_por_n = df_cuckoo.loc[df_cuckoo.groupby("n")["Distancia"].idxmin()]

print("\n🟢 Mejores rutas encontradas (menor distancia):")
print(mejores_globales)

print("\n📊 Mejor resultado por cada tamaño:")
print(mejores_por_n[["n", "Distancia", "Tiempo (s)", "n_nests", "max_iter", "pa", "alpha", "lambda"]])

with open("resultados_cuckoo.txt", "a", encoding="utf-8") as file:
    file.write("\n🟢 Mejores rutas encontradas (menor distancia):\n")
    file.write(mejores_globales.to_string(index=False) + "\n")
    file.write("\n📊 Mejor resultado por cada tamaño:\n")
    file.write(mejores_por_n[["n", "Distancia", "Tiempo (s)", "n_nests", "max_iter", "pa", "alpha", "lambda"]].to_string(index=False) + "\n")

# Crear carpeta para guardar imágenes
carpeta = "mejores_rutas_cuckoo"
os.makedirs(carpeta, exist_ok=True)

# Graficar y guardar solo las mejores rutas por tamaño
for _, fila in mejores_por_n.iterrows():
    nombre_archivo = os.path.join(carpeta, f"cuckoo_n{fila['n']}.png")
    graficar_ruta_grafo_espacial(
        fila["Ruta"],
        fila["Algoritmo"],
        fila["n"],
        fila["Distancia"],
        save_path=nombre_archivo,
        seed=int(fila["n"] + 2000)
    )