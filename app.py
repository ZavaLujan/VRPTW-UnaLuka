import math
import random
from copy import deepcopy
from colorama import init, Fore, Style

# Para los olores de la terminal
init(autoreset=True)

# -----------------------------
# Definición de clases y funciones básicas
# -----------------------------

class Cliente:
    def __init__(self, id, nombre, x, y, demanda, ventana_inicio, ventana_fin):
        """
        Representa a un cliente con su ubicación, demanda y ventana de tiempo.
        Los tiempos se expresan en minutos (por ejemplo, 480 = 08:00).
        """
        self.id = id
        self.nombre = nombre
        self.x = x
        self.y = y
        self.demanda = demanda
        self.ventana_inicio = ventana_inicio  # Apertura (en minutos)
        self.ventana_fin = ventana_fin        # Cierre (en minutos)
        # Cálculo del ángulo polar respecto al depósito (origen)
        self.angulo = math.atan2(y, x)
    
    def __repr__(self):
        return f"{self.nombre}({self.x}, {self.y})"

class Deposito:
    def __init__(self, x, y, ventana_inicio, ventana_fin):
        """
        Representa el depósito con su ubicación y ventana de servicio.
        """
        self.x = x
        self.y = y
        self.ventana_inicio = ventana_inicio
        self.ventana_fin = ventana_fin

def distancia(punto1, punto2):
    """Calcula la distancia Euclidiana entre dos puntos (Cliente o Deposito)."""
    return math.sqrt((punto1.x - punto2.x)**2 + (punto1.y - punto2.y)**2)

# -----------------------------
# Fase 1: Agrupamiento mediante heurística de barrido (Sweep)
# -----------------------------

def agrupar_clientes(clientes, capacidad_vehiculo):
    """
    Agrupa los clientes en grupos (rutas) usando la heurística de barrido.
    Se ordenan los clientes por su ángulo polar y se agrupan hasta no exceder
    la capacidad máxima del vehículo.
    """
    clientes_ordenados = sorted(clientes, key=lambda c: c.angulo)
    grupos = []
    grupo_actual = []
    capacidad_actual = 0
    for cliente in clientes_ordenados:
        if capacidad_actual + cliente.demanda <= capacidad_vehiculo:
            grupo_actual.append(cliente)
            capacidad_actual += cliente.demanda
        else:
            grupos.append(grupo_actual)
            grupo_actual = [cliente]
            capacidad_actual = cliente.demanda
    if grupo_actual:
        grupos.append(grupo_actual)
    return grupos

# -----------------------------
# Función de evaluación de la ruta (incluye ventanas de tiempo)
# -----------------------------

def evaluar_ruta(ruta, deposito):
    """
    Evalúa una ruta (lista de clientes en un orden dado) calculando:
      - La distancia total recorrida (incluyendo el regreso al depósito).
      - Los tiempos de llegada y de espera.
      - Penalizaciones si se incumplen las ventanas de tiempo.
    
    Se asume:
      - Una velocidad de 1 unidad de distancia por minuto.
      - Servicio inmediato (sin tiempo de servicio).
    """
    distancia_total = 0.0
    tiempo_actual = deposito.ventana_inicio  # Hora de inicio en el depósito
    penalizacion = 0.0
    detalles = []  # Almacena los detalles de cada paso: (cliente, hora de llegada, tiempo de espera)
    punto_anterior = deposito
    
    for cliente in ruta:
        d = distancia(punto_anterior, cliente)
        tiempo_viaje = d  # 1 unidad de distancia = 1 minuto
        tiempo_llegada = tiempo_actual + tiempo_viaje
        tiempo_espera = 0
        
        # Si se llega antes de la ventana, se espera
        if tiempo_llegada < cliente.ventana_inicio:
            tiempo_espera = cliente.ventana_inicio - tiempo_llegada
            tiempo_llegada = cliente.ventana_inicio
        
        # Penalización por llegar tarde (fuera del intervalo permitido)
        if tiempo_llegada > cliente.ventana_fin:
            penalizacion += (tiempo_llegada - cliente.ventana_fin) * 1000
        
        distancia_total += d
        detalles.append((cliente, tiempo_llegada, tiempo_espera))
        tiempo_actual = tiempo_llegada  # Servicio inmediato
        punto_anterior = cliente

    # Regreso al depósito
    d = distancia(punto_anterior, deposito)
    distancia_total += d
    tiempo_actual += d
    detalles.append(("Deposito", tiempo_actual, 0))
    
    # La función fitness combina la distancia y penalizaciones
    fitness = distancia_total + penalizacion
    return fitness, distancia_total, penalizacion, detalles

# -----------------------------
# Funciones del Algoritmo Genético para TSP con ventanas de tiempo
# -----------------------------

def crear_poblacion_inicial(clientes, tamano_poblacion):
    """Genera la población inicial de rutas (permutaciones aleatorias de clientes)."""
    poblacion = []
    for _ in range(tamano_poblacion):
        individuo = clientes[:]  # Copia de la lista
        random.shuffle(individuo)
        poblacion.append(individuo)
    return poblacion

def seleccion_por_torneo(poblacion, fitnesses, tamano_torneo=3):
    """Selecciona un individuo usando torneo (el de menor fitness)."""
    seleccionados = random.sample(list(zip(poblacion, fitnesses)), tamano_torneo)
    seleccionados.sort(key=lambda x: x[1])
    return deepcopy(seleccionados[0][0])

def cruce_ordenado(padre1, padre2):
    """
    Operador de cruce ordenado (Order Crossover, OX):
      - Se selecciona un segmento del primer padre.
      - Se preserva el orden de los genes del segundo padre en los espacios restantes.
    """
    tamano = len(padre1)
    inicio, fin = sorted(random.sample(range(tamano), 2))
    hijo = [None] * tamano
    # Copiar segmento de padre1
    hijo[inicio:fin+1] = padre1[inicio:fin+1]
    pos = (fin + 1) % tamano
    for gen in padre2:
        if gen not in hijo:
            hijo[pos] = gen
            pos = (pos + 1) % tamano
    return hijo

def mutacion_intercambio(individuo, tasa_mutacion):
    """Operador de mutación que intercambia dos genes aleatoriamente."""
    individuo = individuo[:]  # Hacemos una copia
    for i in range(len(individuo)):
        if random.random() < tasa_mutacion:
            j = random.randint(0, len(individuo) - 1)
            individuo[i], individuo[j] = individuo[j], individuo[i]
    return individuo

def algoritmo_genetico(clientes, deposito, tamano_poblacion=30, generaciones=100, tasa_cruce=0.8, tasa_mutacion=0.1):
    """
    Implementa el algoritmo genético para optimizar la ruta (TSP) con ventanas de tiempo.
    Devuelve la mejor ruta encontrada y su fitness.
    """
    poblacion = crear_poblacion_inicial(clientes, tamano_poblacion)
    mejor_individuo = None
    mejor_fitness = float('inf')

    for gen in range(generaciones):
        fitnesses = [evaluar_ruta(ind, deposito)[0] for ind in poblacion]
        # Actualizar la mejor solución de la generación
        indice_mejor = fitnesses.index(min(fitnesses))
        fitness_actual = fitnesses[indice_mejor]
        if fitness_actual < mejor_fitness:
            mejor_fitness = fitness_actual
            mejor_individuo = deepcopy(poblacion[indice_mejor])
        
        # Mostrar progreso cada 10 generaciones
        if gen % 10 == 0:
            print(Fore.CYAN + f"[Gen {gen:03d}] Mejor fitness: {mejor_fitness:.2f}")

        # Crear nueva población
        nueva_poblacion = []
        for _ in range(tamano_poblacion):
            padre1 = seleccion_por_torneo(poblacion, fitnesses)
            padre2 = seleccion_por_torneo(poblacion, fitnesses)
            if random.random() < tasa_cruce:
                hijo = cruce_ordenado(padre1, padre2)
            else:
                hijo = padre1[:]
            hijo = mutacion_intercambio(hijo, tasa_mutacion)
            nueva_poblacion.append(hijo)
        poblacion = nueva_poblacion

    return mejor_individuo, mejor_fitness

# -----------------------------
# Función principal: Configuración de datos, ejecución y presentación de resultados
# -----------------------------

def main():
    # Definición del depósito y los clientes
    deposito = Deposito(0, 0, 480, 1080)  # Ventana: 08:00 (480 min) a 18:00 (1080 min)
    clientes = [
        Cliente(1, "A", 10, 0, 20, 540, 660),    # 09:00 - 11:00
        Cliente(2, "B", 8, 8, 30, 600, 720),      # 10:00 - 12:00
        Cliente(3, "C", 0, 10, 40, 510, 630),     # 08:30 - 10:30
        Cliente(4, "D", -8, 8, 10, 660, 780),     # 11:00 - 13:00
        Cliente(5, "E", -10, 0, 50, 720, 840),    # 12:00 - 14:00
        Cliente(6, "F", -8, -8, 30, 780, 900)     # 13:00 - 15:00
    ]
    capacidad_vehiculo = 100

    # Encabezado de presentación
    print(Style.BRIGHT + Fore.GREEN + "="*50)
    print(Style.BRIGHT + Fore.GREEN + "      Algoritmo VRPTW - Empresa UnaLuka")
    print(Style.BRIGHT + Fore.GREEN + "="*50)
    
    # Mostrar datos de entrada
    print("\n" + Style.BRIGHT + Fore.YELLOW + "Datos de entrada:")
    print(Fore.YELLOW + "Deposito: (0, 0), Ventana: [08:00, 18:00]")
    for c in clientes:
        hora_inicio, min_inicio = divmod(c.ventana_inicio, 60)
        hora_fin, min_fin = divmod(c.ventana_fin, 60)
        print(Fore.YELLOW + f"Cliente {c.nombre}: Ubicación=({c.x}, {c.y}), Demanda={c.demanda}, Ventana=[{hora_inicio:02d}:{min_inicio:02d}, {hora_fin:02d}:{min_fin:02d}]")
    
    # Fase 1: Agrupamiento de clientes
    print("\n" + Style.BRIGHT + Fore.MAGENTA + "-"*50)
    print(Style.BRIGHT + Fore.MAGENTA + "Fase 1: Agrupamiento de clientes (Heurística de Barrido)")
    print(Fore.MAGENTA + "-"*50)
    
    grupos = agrupar_clientes(clientes, capacidad_vehiculo)
    for i, grupo in enumerate(grupos, 1):
        demanda_total = sum(c.demanda for c in grupo)
        print(Fore.MAGENTA + f"Vehículo {i}: Grupo = {[c.nombre for c in grupo]} (Demanda total = {demanda_total})")
    
    # Fase 2: Optimización de rutas con Algoritmo Genético
    print("\n" + Style.BRIGHT + Fore.BLUE + "-"*50)
    print(Style.BRIGHT + Fore.BLUE + "Fase 2: Optimización de rutas (TSP con Ventanas de Tiempo) - Algoritmo Genético")
    print(Fore.BLUE + "-"*50)
    
    for i, grupo in enumerate(grupos, 1):
        print("\n" + Fore.BLUE + "="*50)
        print(Fore.BLUE + f"Optimizando ruta para Vehículo {i} - Clientes: {[c.nombre for c in grupo]}")
        mejor_ruta, mejor_fit = algoritmo_genetico(grupo, deposito,
                                                   tamano_poblacion=30,
                                                   generaciones=100,
                                                   tasa_cruce=0.8,
                                                   tasa_mutacion=0.1)
        fitness, dist_total, penal, detalles = evaluar_ruta(mejor_ruta, deposito)
        
        # Preparar la ruta final (incluye inicio y fin en el depósito)
        nombres_ruta = ["Deposito"] + [c.nombre for c in mejor_ruta] + ["Deposito"]
        print(Fore.BLUE + f"\nMejor ruta encontrada (Fitness = {fitness:.2f}):")
        print(Fore.BLUE + " -> ".join(nombres_ruta))
        
        # Mostrar detalles de la ruta
        print(Fore.WHITE + "\nDetalles de la ruta:")
        for paso in detalles:
            if paso[0] == "Deposito":
                tiempo = paso[1]
                hora, minuto = divmod(tiempo, 60)
                print(Fore.WHITE + f"Regreso al Deposito a las {int(hora):02d}:{int(minuto):02d}")
            else:
                cliente, llegada, espera = paso
                hora, minuto = divmod(llegada, 60)
                print(Fore.WHITE + f"Llegada a Cliente {cliente.nombre} a las {int(hora):02d}:{int(minuto):02d} (Espera: {espera:.2f} min)")
        print(Fore.BLUE + "="*50)
    
    print("\n" + Style.BRIGHT + Fore.GREEN + "Proceso finalizado. ¡Gracias por utilizar el Algoritmo VRPTW para UnaLuka!")

if __name__ == "__main__":
    main()
