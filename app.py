import math
import random
from copy import deepcopy
from colorama import init, Fore, Style

# Para los colores en la consola
init(autoreset=True)

# -----------------------------
# Definición de clases y funciones básicas
# -----------------------------

class Customer:
    def __init__(self, id, name, x, y, demand, window_start, window_end):
        """
        Representa a un cliente con su ubicación, demanda y ventana de tiempo.
        Los tiempos se expresan en minutos (por ejemplo, 480 = 08:00).
        """
        self.id = id
        self.name = name
        self.x = x
        self.y = y
        self.demand = demand
        self.window_start = window_start  # Apertura (en minutos)
        self.window_end = window_end      # Cierre (en minutos)
        # Cálculo del ángulo polar respecto al depósito (origen)
        self.angle = math.atan2(y, x)
    
    def __repr__(self):
        return f"{self.name}({self.x}, {self.y})"

class Depot:
    def __init__(self, x, y, window_start, window_end):
        """
        Representa el depósito con su ubicación y ventana de servicio.
        """
        self.x = x
        self.y = y
        self.window_start = window_start
        self.window_end = window_end

def distance(p1, p2):
    """Calcula la distancia Euclidiana entre dos puntos (Customer o Depot)."""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# -----------------------------
# Fase 1: Agrupamiento mediante heurística de barrido (Sweep)
# -----------------------------

def cluster_customers(customers, vehicle_capacity):
    """
    Agrupa los clientes en clusters (rutas) usando la heurística de barrido.
    Se ordenan los clientes por su ángulo polar y se agrupan hasta no exceder
    la capacidad máxima del vehículo.
    """
    sorted_customers = sorted(customers, key=lambda c: c.angle)
    clusters = []
    current_cluster = []
    current_capacity = 0
    for c in sorted_customers:
        if current_capacity + c.demand <= vehicle_capacity:
            current_cluster.append(c)
            current_capacity += c.demand
        else:
            clusters.append(current_cluster)
            current_cluster = [c]
            current_capacity = c.demand
    if current_cluster:
        clusters.append(current_cluster)
    return clusters

# -----------------------------
# Función de evaluación de la ruta (incluye ventanas de tiempo)
# -----------------------------

def evaluate_route(route, depot):
    """
    Evalúa una ruta (lista de clientes en un orden dado) calculando:
      - La distancia total recorrida (incluyendo el regreso al depósito).
      - Los tiempos de llegada y tiempos de espera.
      - Penalizaciones si se incumplen las ventanas de tiempo.
    
    Se asume que:
      - La velocidad es 1 unidad de distancia por minuto.
      - El servicio es inmediato (sin tiempo de servicio).
    """
    total_distance = 0.0
    current_time = depot.window_start  # Hora de inicio en el depósito
    penalty = 0.0
    details = []  # Almacena los detalles de cada paso: (cliente, hora de llegada, tiempo de espera)
    prev_point = depot
    
    for c in route:
        d = distance(prev_point, c)
        travel_time = d  # 1 unidad de distancia = 1 minuto
        arrival_time = current_time + travel_time
        wait_time = 0
        
        # Si se llega antes de la ventana, se espera
        if arrival_time < c.window_start:
            wait_time = c.window_start - arrival_time
            arrival_time = c.window_start
        
        # Penalización por llegada tardía (llegada fuera del intervalo)
        if arrival_time > c.window_end:
            penalty += (arrival_time - c.window_end) * 1000  # Penalización alta
        
        total_distance += d
        details.append((c, arrival_time, wait_time))
        current_time = arrival_time  # Se asume servicio inmediato
        prev_point = c

    # Retorno al depósito
    d = distance(prev_point, depot)
    total_distance += d
    current_time += d
    details.append(("Depósito", current_time, 0))
    
    # La función fitness combina la distancia total y las penalizaciones
    fitness = total_distance + penalty
    return fitness, total_distance, penalty, details

# -----------------------------
# Funciones del Algoritmo Genético (GA) para TSP con ventanas de tiempo
# -----------------------------

def create_initial_population(customers, population_size):
    """Genera la población inicial de rutas (permutaciones aleatorias de clientes)."""
    population = []
    for _ in range(population_size):
        individual = customers[:]  # Copia de la lista
        random.shuffle(individual)
        population.append(individual)
    return population

def tournament_selection(population, fitnesses, tournament_size=3):
    """Selecciona un individuo usando torneo (se elige el de menor fitness)."""
    selected = random.sample(list(zip(population, fitnesses)), tournament_size)
    selected.sort(key=lambda x: x[1])
    return deepcopy(selected[0][0])

def order_crossover(parent1, parent2):
    """
    Operador de cruce Order Crossover (OX):
      - Se selecciona un segmento del primer padre.
      - Se preserva el orden de los genes del segundo padre para los restantes.
    """
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    # Copiar segmento de parent1
    child[start:end+1] = parent1[start:end+1]
    pos = (end + 1) % size
    for gene in parent2:
        if gene not in child:
            child[pos] = gene
            pos = (pos + 1) % size
    return child

def swap_mutation(individual, mutation_rate):
    """Operador de mutación que intercambia dos genes aleatoriamente."""
    individual = individual[:]  # Copia del individuo
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]
    return individual

def genetic_algorithm(customers, depot, population_size=30, generations=100, crossover_rate=0.8, mutation_rate=0.1):
    """
    Implementa el Algoritmo Genético para optimizar la ruta (TSP) con ventanas de tiempo.
    Devuelve la mejor ruta encontrada y su fitness.
    """
    population = create_initial_population(customers, population_size)
    best_individual = None
    best_fitness = float('inf')

    for gen in range(generations):
        fitnesses = [evaluate_route(ind, depot)[0] for ind in population]
        # Actualizar la mejor solución de la generación
        current_best_index = fitnesses.index(min(fitnesses))
        current_best_fitness = fitnesses[current_best_index]
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = deepcopy(population[current_best_index])
        
        # Mostrar progreso cada 10 generaciones
        if gen % 10 == 0:
            print(Fore.CYAN + f"[Gen {gen:03d}] Mejor fitness: {best_fitness:.2f}")

        # Crear nueva población
        new_population = []
        for _ in range(population_size):
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            if random.random() < crossover_rate:
                child = order_crossover(parent1, parent2)
            else:
                child = parent1[:]
            child = swap_mutation(child, mutation_rate)
            new_population.append(child)
        population = new_population

    return best_individual, best_fitness

# -----------------------------
# Función principal: Configuración de datos, ejecución y presentación de resultados
# -----------------------------

def main():
    # Definición del depósito y de los clientes
    depot = Depot(0, 0, 480, 1080)  # Ventana: 08:00 (480 min) a 18:00 (1080 min)
    customers = [
        Customer(1, "A", 10, 0, 20, 540, 660),    # 09:00 - 11:00
        Customer(2, "B", 8, 8, 30, 600, 720),      # 10:00 - 12:00
        Customer(3, "C", 0, 10, 40, 510, 630),     # 08:30 - 10:30
        Customer(4, "D", -8, 8, 10, 660, 780),     # 11:00 - 13:00
        Customer(5, "E", -10, 0, 50, 720, 840),    # 12:00 - 14:00
        Customer(6, "F", -8, -8, 30, 780, 900)     # 13:00 - 15:00
    ]
    vehicle_capacity = 100

    # Cabecera de presentación
    print(Style.BRIGHT + Fore.GREEN + "="*50)
    print(Style.BRIGHT + Fore.GREEN + "      Algoritmo VRPTW - Empresa UnaLuka")
    print(Style.BRIGHT + Fore.GREEN + "="*50)
    
    # Datos de entrada
    print("\n" + Style.BRIGHT + Fore.YELLOW + "Datos de entrada:")
    print(Fore.YELLOW + "Depósito: (0, 0), Ventana: [08:00, 18:00]")
    for c in customers:
        start_hour, start_min = divmod(c.window_start, 60)
        end_hour, end_min = divmod(c.window_end, 60)
        print(Fore.YELLOW + f"Cliente {c.name}: Ubicación=({c.x}, {c.y}), Demanda={c.demand}, Ventana=[{start_hour:02d}:{start_min:02d}, {end_hour:02d}:{end_min:02d}]")
    
    # -----------------------------
    # Fase 1: Agrupamiento de clientes
    # -----------------------------
    print("\n" + Style.BRIGHT + Fore.MAGENTA + "-"*50)
    print(Style.BRIGHT + Fore.MAGENTA + "Fase 1: Agrupamiento de clientes (Heurística de Barrido)")
    print(Fore.MAGENTA + "-"*50)
    
    clusters = cluster_customers(customers, vehicle_capacity)
    for i, cluster in enumerate(clusters, 1):
        total_demand = sum(c.demand for c in cluster)
        print(Fore.MAGENTA + f"Vehículo {i}: Cluster = {[c.name for c in cluster]} (Demanda total = {total_demand})")
    
    # -----------------------------
    # Fase 2: Optimización de rutas con Algoritmo Genético
    # -----------------------------
    print("\n" + Style.BRIGHT + Fore.BLUE + "-"*50)
    print(Style.BRIGHT + Fore.BLUE + "Fase 2: Optimización de rutas (TSP con Ventanas de Tiempo) - Algoritmo Genético")
    print(Fore.BLUE + "-"*50)
    
    for i, cluster in enumerate(clusters, 1):
        print("\n" + Fore.BLUE + "="*50)
        print(Fore.BLUE + f"Optimizando ruta para Vehículo {i} - Clientes: {[c.name for c in cluster]}")
        best_route, best_fit = genetic_algorithm(cluster, depot,
                                                 population_size=30,
                                                 generations=100,
                                                 crossover_rate=0.8,
                                                 mutation_rate=0.1)
        fitness, total_distance, penalty, details = evaluate_route(best_route, depot)
        
        # Preparar la ruta final incluyendo el depósito al inicio y al final
        route_names = ["Depósito"] + [c.name for c in best_route] + ["Depósito"]
        print(Fore.BLUE + f"\nMejor ruta encontrada (Fitness = {fitness:.2f}):")
        print(Fore.BLUE + " -> ".join(route_names))
        
        # Mostrar detalles de la ruta
        print(Fore.WHITE + "\nDetalles de la ruta:")
        for step in details:
            if step[0] == "Depósito":
                time = step[1]
                hour, minute = divmod(time, 60)
                print(Fore.WHITE + f"Regreso al Depósito a las {int(hour):02d}:{int(minute):02d}")
            else:
                c, arrival, wait = step
                hour, minute = divmod(arrival, 60)
                print(Fore.WHITE + f"Llegada a Cliente {c.name} a las {int(hour):02d}:{int(minute):02d} (Espera: {wait:.2f} min)")
        print(Fore.BLUE + "="*50)
    
    print("\n" + Style.BRIGHT + Fore.GREEN + "Proceso finalizado. ¡Gracias por utilizar el Algoritmo VRPTW para UnaLuka!")

if __name__ == "__main__":
    main()
