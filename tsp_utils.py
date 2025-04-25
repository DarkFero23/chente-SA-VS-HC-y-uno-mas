# archivo: tsp_utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Funciones generales

def generar_tsp(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.randint(10, 500, size=(n, n)) * (1 - np.eye(n, dtype=int))

def fitness(sol, matriz):
    distancia = 0
    for i in range(len(sol) - 1):
        distancia += matriz[sol[i]][sol[i + 1]]
    distancia += matriz[sol[-1]][sol[0]]
    return distancia

def get_neighbors(sol):
    vecinos = []
    for i in range(len(sol)):
        for j in range(i + 1, len(sol)):
            vecino = sol.copy()
            vecino[i], vecino[j] = vecino[j], vecino[i]
            vecinos.append(vecino)
    return vecinos

def graficar_ruta_grafo_espacial(ruta, algoritmo, n, distancia, save_path=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(42)  # Valor por defecto si no se pasa seed

    posiciones = {i: (np.random.rand(), np.random.rand()) for i in ruta}

    plt.figure(figsize=(10, 8))
    for nodo, (x, y) in posiciones.items():
        plt.scatter(x, y, s=80, color="skyblue", zorder=2)
        plt.text(x, y + 0.01, str(nodo), fontsize=8, ha='center', zorder=3)
    for i in range(len(ruta)):
        origen = ruta[i]
        destino = ruta[(i + 1) % len(ruta)]
        x_values = [posiciones[origen][0], posiciones[destino][0]]
        y_values = [posiciones[origen][1], posiciones[destino][1]]
        plt.plot(x_values, y_values, 'k-', linewidth=1, alpha=0.6, zorder=1)

    plt.title(f"Ruta en grafo espacial | {algoritmo} | n={n} | Distancia: {distancia}")
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def graficar_comparaciones(df, tipo="Distancia"):
    plt.figure(figsize=(12, 6))
    if "Temp Inicial" in df.columns and df["Temp Inicial"].nunique() > 1:
        sns.lineplot(data=df, x="n", y=tipo, hue="Temp Inicial", style="Algoritmo", marker="o")
    else:
        sns.lineplot(data=df, x="n", y=tipo, hue="Algoritmo", marker="o")

    plt.title(f"Comparación de {tipo.lower()} por algoritmo")
    plt.xlabel("Número de ciudades (n)")
    plt.ylabel(tipo)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
