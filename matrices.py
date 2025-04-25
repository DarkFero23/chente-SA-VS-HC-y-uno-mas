from tsp_utils import generar_tsp

# Lista de tamaños de instancias
tamaños = [10, 20, 50, 100, 200, 500]

# Usamos la semilla fija igual al tamaño para que sea justo p
#matrices = {n: generar_tsp(n, seed=n) for n in tamaños}

# CAMBIA la semilla tmb justo
semilla_base = 5000
matrices = {n: generar_tsp(n, seed=n + semilla_base) for n in tamaños}
