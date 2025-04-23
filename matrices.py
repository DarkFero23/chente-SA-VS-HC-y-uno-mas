from tsp_utils import generar_tsp

# Lista de tamaños de instancias
tamaños = [10, 20, 50, 100, 200, 500]

# Usamos la semilla fija igual al tamaño para que sea reproducible
#matrices = {n: generar_tsp(n, seed=n) for n in tamaños}

# CAMBIA la semilla (por ejemplo usa n+999)
semilla_base = 200  # 👈 cambia este número y todo cambia
matrices = {n: generar_tsp(n, seed=n + semilla_base) for n in tamaños}
