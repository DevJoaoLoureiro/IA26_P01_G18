"""Projeto IA26_P01_G18 — algoritmos de rotas (instância data_tiny).

Modelo: cada artigo é uma tarefa pickup/delivery; um veículo parte do centro,
respeita janelas, precedência e carga, e regressa ao centro.
A data da encomenda é considerada a primeira data permitida para recolha.
Inclui as abordagens para um veículo e para frota de dois veículos.
Os algoritmos de pesquisa usam limites e não garantem otimalidade global.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import heapq
import itertools
import math
import time
import pandas as pd

@dataclass(frozen=True)
class Vehicle:
    vehicle_id: int
    capacity_kg: float
    center_location: int

@dataclass(frozen=True)
class Request:
    item_id: int
    order_id: int
    pickup: int
    delivery: int
    weight: float
    ready: pd.Timestamp
    pickup_end: pd.Timestamp
    delivery_start: pd.Timestamp
    delivery_end: pd.Timestamp

@dataclass
class Problem:
    vehicle: Vehicle
    requests: list[Request]
    distance: dict[tuple[int,int],float]
    duration: dict[tuple[int,int],float]
    start_time: pd.Timestamp
    rejected: list[tuple[int,str]] = field(default_factory=list)

@dataclass
class State:
    location: int
    moment: pd.Timestamp
    load: float = 0.0
    picked: frozenset[int] = frozenset()
    delivered: frozenset[int] = frozenset()
    route: tuple[tuple[str,int,int,str],...] = ()
    km: float = 0.0

@dataclass
class Result:
    algorithm: str
    feasible: bool
    distance_km: float
    expanded: int
    runtime_s: float
    route: list[tuple[str,int,int,str]]
    delivered: int
    total: int
    message: str = ""


def load_problem(folder="data", vehicle_id=1, start_time=None):
    """Lê os 7 CSV necessários, mesmo que os outros estejam presentes."""
    folder = Path(folder)
    names = ["orders", "order_items", "products", "producers", "vehicles",
             "locations", "distance_matrix"]
    data = {name: pd.read_csv(folder / f"{name}.csv") for name in names}
    locs = data["locations"]
    def locate(kind, ref):
        rows = locs[(locs.location_type == kind) & (locs.ref_id == ref)]
        if len(rows) != 1:
            raise ValueError(f"Localização não única: {kind}, {ref}")
        return int(rows.iloc[0].location_id)
    vrow = data["vehicles"].set_index("vehicle_id").loc[vehicle_id]
    vehicle = Vehicle(int(vehicle_id), float(vrow.capacity_kg),
                      locate("collection_center", int(vrow.center_id)))
    orders = data["orders"].set_index("order_id")
    products = data["products"].set_index("product_id")
    requests = []
    rejected = []
    for r in data["order_items"].itertuples(index=False):
        order = orders.loc[r.order_id]
        product = products.loc[r.product_id]
        placed = pd.Timestamp(order.order_datetime)
        available = pd.Timestamp(r.availability_start)
        ready = max(placed, available)
        pickup_end = pd.Timestamp(r.availability_end)
        delivery_start = pd.Timestamp(r.delivery_start)
        delivery_end = pd.Timestamp(r.delivery_end)
        weight = float(r.quantity) * float(product.unit_weight_kg)
        item_id = int(r.order_item_id)
        if pickup_end < ready:
            rejected.append((item_id, "recolha termina antes de a encomenda poder ser recolhida"))
            continue
        if delivery_end < ready or delivery_end < delivery_start:
            rejected.append((item_id, "janela de entrega impossível"))
            continue
        if weight > vehicle.capacity_kg:
            rejected.append((item_id, "peso excede capacidade do veículo"))
            continue
        requests.append(Request(item_id, int(r.order_id),
            locate("producer", int(product.producer_id)),
            locate("delivery_location", int(order.delivery_location_id)),
            weight, ready, pickup_end, delivery_start, delivery_end))
    dist = {(int(r.origin_location_id),int(r.destination_location_id)):float(r.distance_km)
            for r in data["distance_matrix"].itertuples(index=False)}
    dur = {(int(r.origin_location_id),int(r.destination_location_id)):float(r.duration_min)
           for r in data["distance_matrix"].itertuples(index=False)}
    if start_time is None:
        start_time = min((r.ready for r in requests), default=pd.Timestamp.now())
    return Problem(vehicle, requests, dist, dur, pd.Timestamp(start_time), rejected)


def travel(problem, origin, destination):
    if origin == destination:
        return 0.0, 0.0
    return problem.distance[(origin,destination)], problem.duration[(origin,destination)]


def initial(problem):
    return State(problem.vehicle.center_location, problem.start_time)


def actions(problem, state):
    """Transições viáveis; espera até ao início da janela, quando necessário."""
    for req in problem.requests:
        if req.item_id not in state.picked:
            if state.load + req.weight > problem.vehicle.capacity_kg + 1e-9:
                continue
            km, minutes = travel(problem, state.location, req.pickup)
            arrival = state.moment + pd.Timedelta(minutes=minutes)
            arrival = max(arrival, req.ready)
            if arrival > req.pickup_end:
                continue
            yield State(req.pickup, arrival, state.load + req.weight,
                state.picked | {req.item_id}, state.delivered,
                state.route + (("recolha", req.item_id, req.pickup, str(arrival)),), state.km + km)
        elif req.item_id not in state.delivered:
            km, minutes = travel(problem, state.location, req.delivery)
            arrival = max(state.moment + pd.Timedelta(minutes=minutes), req.delivery_start)
            if arrival > req.delivery_end:
                continue
            yield State(req.delivery, arrival, max(0.0,state.load - req.weight),
                state.picked, state.delivered | {req.item_id},
                state.route + (("entrega", req.item_id, req.delivery, str(arrival)),), state.km + km)


def finish(problem, state):
    km, minutes = travel(problem, state.location, problem.vehicle.center_location)
    arrival = state.moment + pd.Timedelta(minutes=minutes)
    return State(problem.vehicle.center_location, arrival, state.load,
        state.picked, state.delivered,
        state.route + (("regresso", 0, problem.vehicle.center_location, str(arrival)),), state.km + km)


def heuristic(problem, state, variant=1):
    """h1: menor distância até uma ação pendente; h2: maior distância até uma ação pendente.
    Heurísticas informativas, não garantidamente admissíveis para A*."""
    targets = []
    for req in problem.requests:
        if req.item_id not in state.picked:
            targets.append(req.pickup)
        elif req.item_id not in state.delivered:
            targets.append(req.delivery)
    if not targets:
        return travel(problem,state.location,problem.vehicle.center_location)[0]
    values = [travel(problem,state.location,x)[0] for x in targets]
    return min(values) if variant == 1 else max(values)


def search(problem, method="astar", heuristic_variant=1, max_expansions=50000):
    """Procura no espaço de tarefas de um único veículo. A* sem garantia de otimalidade
    com estas heurísticas. Se não houver solução, devolve resultado inviável."""
    start = time.perf_counter()
    counter = itertools.count()
    root = initial(problem)
    heap = [(heuristic(problem,root,heuristic_variant),next(counter),root)]
    expanded = 0
    best = {}
    n = len(problem.requests)
    while heap and expanded < max_expansions:
        _,_,state = heapq.heappop(heap)
        if len(state.delivered) == n:
            done = finish(problem,state)
            return Result(method, True, done.km, expanded, time.perf_counter()-start,
                          list(done.route), n, n)
        key = (state.location,state.picked,state.delivered,
               round(state.load,5))
        # Dominância: menor distância e chegada não posterior.
        previous = best.get(key, [])
        if any(k <= state.km + 1e-9 and t <= state.moment for k,t in previous):
            continue
        best[key] = [(k,t) for k,t in previous
                     if not (state.km <= k + 1e-9 and state.moment <= t)] + [(state.km,state.moment)]
        expanded += 1
        for nxt in actions(problem,state):
            h = heuristic(problem,nxt,heuristic_variant)
            priority = h if method == "greedy" else nxt.km + h
            heapq.heappush(heap,(priority,next(counter),nxt))
    return Result(method,False,math.inf,expanded,time.perf_counter()-start,
                  [],0,n,"Sem solução completa ou limite de expansões atingido")


def greedy(problem, heuristic_variant=1, max_expansions=50000):
    return search(problem,"greedy",heuristic_variant,max_expansions)


def astar(problem, heuristic_variant=1, max_expansions=50000):
    return search(problem,"astar",heuristic_variant,max_expansions)


def evaluate_sequence(problem, sequence):
    """Avalia uma sequência fixa de ('recolha'|'entrega', item_id)."""
    state = initial(problem)
    lookup = {r.item_id:r for r in problem.requests}
    for kind, item in sequence:
        options = list(actions(problem,state))
        found = next((s for s in options if s.route[-1][0] == kind and
                      s.route[-1][1] == item),None)
        if found is None:
            return None
        state = found
    if len(state.delivered) != len(problem.requests):
        return None
    return finish(problem,state)


def hill_climbing(problem, initial_result, max_iterations=200):
    """Melhoria local por trocas de duas operações, mantendo apenas rotas viáveis."""
    start = time.perf_counter()
    n = len(problem.requests)
    if not initial_result.feasible:
        return Result("hill_climbing",False,math.inf,0,time.perf_counter()-start,[],0,n,
                      "Requer solução inicial completa")
    sequence = [(kind,item) for kind,item,_,_ in initial_result.route if kind != "regresso"]
    current = evaluate_sequence(problem,sequence)
    iterations = 0
    while iterations < max_iterations:
        best_state = current
        best_sequence = sequence
        for i in range(len(sequence)):
            for j in range(i+1,len(sequence)):
                candidate = sequence.copy()
                candidate[i],candidate[j] = candidate[j],candidate[i]
                outcome = evaluate_sequence(problem,candidate)
                if outcome is not None and outcome.km < best_state.km - 1e-9:
                    best_state,best_sequence = outcome,candidate
        iterations += 1
        if best_sequence == sequence:
            break
        current,sequence = best_state,best_sequence
    return Result("hill_climbing",True,current.km,iterations,time.perf_counter()-start,
                  list(current.route),n,n)


def csp_backtracking(problem, max_nodes=50000):
    """CSP: variáveis = sequência de operações; domínios = ações ainda possíveis;
    restrições = precedência, capacidade e janelas. Retorna primeira solução."""
    start = time.perf_counter()
    nodes = 0
    backtracks = 0
    n = len(problem.requests)
    def dfs(state):
        nonlocal nodes,backtracks
        if len(state.delivered) == n:
            return finish(problem,state)
        if nodes >= max_nodes:
            return None
        nodes += 1
        choices = sorted(actions(problem,state),key=lambda s:(s.moment,s.km))
        for nxt in choices:
            answer = dfs(nxt)
            if answer is not None:
                return answer
        backtracks += 1
        return None
    answer = dfs(initial(problem))
    if answer is None:
        return Result("csp",False,math.inf,nodes,time.perf_counter()-start,
                      [],0,n,f"Sem solução; backtracks={backtracks}")
    return Result("csp",True,answer.km,nodes,time.perf_counter()-start,
                  list(answer.route),n,n,f"backtracks={backtracks}")


def summary(results):
    return pd.DataFrame([{"algoritmo":r.algorithm,"viavel":r.feasible,
        "distancia_km":r.distance_km if r.feasible else None,
        "nos_ou_iteracoes":r.expanded,"tempo_s":r.runtime_s,
        "artigos_entregues":r.delivered,"total_artigos":r.total,
        "observacoes":r.message} for r in results])

# ================== EXTENSÃO: FROTA DE DOIS VEÍCULOS ==================
# Mantém as funções anteriores para compatibilidade com os testes do notebook.

@dataclass
class FleetProblem:
    vehicles: list[Vehicle]
    requests: list[Request]
    distance: dict
    duration: dict
    start_time: pd.Timestamp
    rejected: list[tuple[int,str]] = field(default_factory=list)

@dataclass
class FleetResult:
    algorithm: str
    feasible: bool
    distance_km: float
    expanded: int
    runtime_s: float
    routes: dict[int,list[tuple[str,int,int,str]]]
    delivered: int
    total: int
    message: str = ""


def load_fleet_problem(folder="data", start_time=None):
    """Carrega os dois veículos e todos os artigos temporalmente elegíveis.
    Não rejeita artigos em função da capacidade de apenas uma carrinha.
    """
    folder = Path(folder)
    dfv = pd.read_csv(folder / "vehicles.csv")
    base = load_problem(folder, vehicle_id=int(dfv.iloc[dfv.capacity_kg.argmax()].vehicle_id),
                        start_time=start_time)
    locs = pd.read_csv(folder / "locations.csv")
    vehicles = []
    for v in dfv.itertuples(index=False):
        rows = locs[(locs.location_type == "collection_center") & (locs.ref_id == v.center_id)]
        if len(rows) != 1:
            raise ValueError(f"Centro não encontrado para veículo {v.vehicle_id}")
        vehicles.append(Vehicle(int(v.vehicle_id), float(v.capacity_kg), int(rows.iloc[0].location_id)))
    return FleetProblem(vehicles, base.requests, base.distance, base.duration,
                        base.start_time, base.rejected)


def _fleet_candidates(problem, states, item_ids):
    """Gera recolhas/entregas viáveis para cada carrinha; cada artigo só pertence a uma."""
    lookup = {r.item_id:r for r in problem.requests}
    for vi, state in enumerate(states):
        vehicle = problem.vehicles[vi]
        for item in item_ids:
            r = lookup[item]
            # item já recolhido por outro veículo: não pode mudar de carrinha
            owner = next((i for i,s in enumerate(states) if item in s.picked), None)
            if owner is None:
                if state.load + r.weight > vehicle.capacity_kg + 1e-9:
                    continue
                loc, kind = r.pickup, "recolha"
                window_start, window_end = r.ready, r.pickup_end
                delta_load = r.weight
            elif owner == vi and item not in state.delivered:
                loc, kind = r.delivery, "entrega"
                window_start, window_end = r.delivery_start, r.delivery_end
                delta_load = -r.weight
            else:
                continue
            km = 0.0 if state.location == loc else problem.distance[(state.location,loc)]
            minutes = 0.0 if state.location == loc else problem.duration[(state.location,loc)]
            arrival = max(state.moment + pd.Timedelta(minutes=minutes), window_start)
            if arrival > window_end:
                continue
            nxt = State(loc, arrival, max(0.,state.load + delta_load),
                        state.picked | ({item} if kind == 'recolha' else set()),
                        state.delivered | ({item} if kind == 'entrega' else set()),
                        state.route + ((kind,item,loc,str(arrival)),),state.km+km)
            yield vi,nxt,km,kind,item


def _fleet_complete(problem, states, delivered_count, expanded, start, algorithm, message=""):
    routes = {}
    distance = 0.0
    for vehicle,state in zip(problem.vehicles,states):
        km = 0.0 if state.location == vehicle.center_location else problem.distance[(state.location,vehicle.center_location)]
        minutes = 0.0 if state.location == vehicle.center_location else problem.duration[(state.location,vehicle.center_location)]
        arrival = state.moment + pd.Timedelta(minutes=minutes)
        routes[vehicle.vehicle_id] = list(state.route) + [("regresso",0,vehicle.center_location,str(arrival))]
        distance += state.km + km
    return FleetResult(algorithm,delivered_count==len(problem.requests),distance,expanded,
                       time.perf_counter()-start,routes,delivered_count,len(problem.requests),message)


def fleet_greedy(problem, heuristic_variant=1, max_steps=10000):
    """Constrói uma solução conjunta para os veículos, sem duplicar artigos.
    h1 privilegia deslocação curta; h2 privilegia próxima janela a fechar.
    O resultado pode ser parcial; feasible=True só se todas as entregas ocorrerem.
    """
    start = time.perf_counter()
    states = tuple(State(v.center_location,problem.start_time) for v in problem.vehicles)
    item_ids = {r.item_id for r in problem.requests}
    expanded = 0
    for _ in range(max_steps):
        delivered = set().union(*(s.delivered for s in states))
        if delivered == item_ids:
            return _fleet_complete(problem,states,len(delivered),expanded,start,"fleet_greedy")
        candidates = list(_fleet_candidates(problem,states,item_ids-delivered))
        expanded += 1
        if not candidates:
            return _fleet_complete(problem,states,len(delivered),expanded,start,"fleet_greedy",
                                   "Sem próximas ações válidas; solução parcial")
        if heuristic_variant == 1:
            selected = min(candidates,key=lambda c:(c[2],c[1].moment, c[0]))
        elif heuristic_variant == 2:
            lookup = {r.item_id:r for r in problem.requests}
            selected = min(candidates,key=lambda c:(
                (lookup[c[4]].pickup_end if c[3]=='recolha' else lookup[c[4]].delivery_end)-c[1].moment,
                c[2],c[0]))
        else:
            raise ValueError("heuristic_variant deve ser 1 ou 2")
        vi,nxt,_,_,_ = selected
        states = tuple(nxt if j==vi else s for j,s in enumerate(states))
    delivered = set().union(*(s.delivered for s in states))
    return _fleet_complete(problem,states,len(delivered),expanded,start,"fleet_greedy","Limite de passos")


def fleet_astar(problem, heuristic_variant=1, max_expansions=10000):
    """Procura conjunta limitada; pode não encontrar solução no limite.
    Heurísticas sem garantia de admissibilidade -> não garante ótimo.
    """
    start = time.perf_counter()
    root = tuple(State(v.center_location,problem.start_time) for v in problem.vehicles)
    ids = frozenset(r.item_id for r in problem.requests)
    counter = itertools.count()
    queue = [(0.,next(counter),root)]
    seen = set()
    expanded = 0
    best_partial = root
    while queue and expanded < max_expansions:
        _,_,states = heapq.heappop(queue)
        delivered = frozenset().union(*(s.delivered for s in states))
        if len(delivered)>len(frozenset().union(*(s.delivered for s in best_partial))):
            best_partial=states
        if delivered==ids:
            return _fleet_complete(problem,states,len(delivered),expanded,start,"fleet_astar")
        key=tuple((s.location,s.picked,s.delivered,round(s.load,5),s.moment) for s in states)
        if key in seen:
            continue
        seen.add(key)
        expanded+=1
        for vi,nxt,km,kind,item in _fleet_candidates(problem,states,ids-delivered):
            new_states=tuple(nxt if j==vi else s for j,s in enumerate(states))
            cost=sum(s.km for s in new_states)
            remaining=len(ids)-len(frozenset().union(*(s.delivered for s in new_states)))
            # h1: 0 é admissível mas pouco informativa; h2: distância mínima 
            # a uma operação possível (não garantidamente admissível).
           
            h = 0.0

            if heuristic_variant == 2:
                if remaining > 0:
                    pendentes = ids - frozenset().union(
                        *(s.delivered for s in new_states)
                    )

                    opts = list(
                        _fleet_candidates(
                            problem,
                            new_states,
                            pendentes
                        )
                    )

                    h = min(
                        (x[2] for x in opts),
                        default=0.0
                    )

            elif heuristic_variant != 1:
                raise ValueError(
                    "heuristic_variant deve ser 1 ou 2"
                )

            heapq.heappush(queue,(cost+h,next(counter),new_states))
    delivered=frozenset().union(*(s.delivered for s in best_partial))
    result=_fleet_complete(problem,best_partial,len(delivered),expanded,start,"fleet_astar",
                 "Procura terminou sem solução completa (limite ou inviabilidade)")
    result.feasible=False
    return result


def fleet_summary(results):
    return pd.DataFrame([{
        "algoritmo":r.algorithm,"viavel":r.feasible,"distancia_km":round(r.distance_km,2),
        "nos_ou_passos":r.expanded,"tempo_s":r.runtime_s,
        "artigos_entregues":r.delivered,"total_artigos":r.total,
        "observacoes":r.message
    } for r in results])


# ================== HILL CLIMBING E CSP PARA A FROTA ==================

def evaluate_fleet_sequences(problem, sequences):
    """Simula rotas fixas dos veículos e devolve FleetResult se forem válidas."""
    start = time.perf_counter()
    by_id = {r.item_id: r for r in problem.requests}
    states = [State(v.center_location, problem.start_time)
              for v in problem.vehicles]
    used = set()
    for vi, sequence in enumerate(sequences):
        vehicle = problem.vehicles[vi]
        state = states[vi]
        for kind, item in sequence:
            if item not in by_id or kind not in ("recolha", "entrega"):
                return None
            req = by_id[item]
            owner = next((j for j, s in enumerate(states)
                          if item in s.picked), None)
            if kind == "recolha":
                if owner is not None or item in used:
                    return None
                if state.load + req.weight > vehicle.capacity_kg + 1e-9:
                    return None
                loc, earliest, latest = req.pickup, req.ready, req.pickup_end
                load = state.load + req.weight
            else:
                if owner != vi or item in state.delivered:
                    return None
                loc, earliest, latest = (req.delivery, req.delivery_start,
                                         req.delivery_end)
                load = state.load - req.weight
            km = 0 if loc == state.location else problem.distance[(state.location,loc)]
            minutes = 0 if loc == state.location else problem.duration[(state.location,loc)]
            moment = max(state.moment + pd.Timedelta(minutes=minutes), earliest)
            if moment > latest:
                return None
            state = State(loc, moment, load,
                          state.picked | ({item} if kind == "recolha" else set()),
                          state.delivered | ({item} if kind == "entrega" else set()),
                          state.route + ((kind,item,loc,str(moment)),), state.km + km)
            states[vi] = state
            if kind == "recolha":
                used.add(item)
    done = set().union(*(s.delivered for s in states))
    if done != set(by_id):
        return None
    return _fleet_complete(problem, tuple(states), len(done), 0, start,
                           "fleet_evaluation")


def fleet_hill_climbing(problem, initial_result, max_iterations=100):
    """Trocas de operações e realocações entre veículos; preserva viabilidade.
    history regista o custo após cada iteração para os gráficos do relatório.
    """
    start = time.perf_counter()
    count = len(problem.requests)
    if not initial_result.feasible:
        return FleetResult("fleet_hill_climbing", False, math.inf, 0,
                           time.perf_counter()-start, {}, 0, count,
                           "É necessária uma solução inicial completa")
    sequences = [
        [(kind,item) for kind,item,_,_ in initial_result.routes[v.vehicle_id]
         if kind != "regresso"]
        for v in problem.vehicles
    ]
    current = evaluate_fleet_sequences(problem, sequences)
    if current is None:
        return FleetResult("fleet_hill_climbing", False, math.inf, 0,
                           time.perf_counter()-start, {}, 0, count,
                           "A rota inicial não passou na validação")
    history = [current.distance_km]
    iterations = 0
    for _ in range(max_iterations):
        best = current
        best_sequences = sequences
        # Permuta pares de operações na mesma rota.
        for vi, seq in enumerate(sequences):
            for i in range(len(seq)):
                for j in range(i+1,len(seq)):
                    candidate = [list(s) for s in sequences]
                    candidate[vi][i], candidate[vi][j] = candidate[vi][j], candidate[vi][i]
                    outcome = evaluate_fleet_sequences(problem, candidate)
                    if outcome and outcome.distance_km < best.distance_km - 1e-9:
                        best, best_sequences = outcome, candidate
        # Move o par recolha/entrega de um veículo para outro (no final).
        for vi, seq in enumerate(sequences):
            items = {item for kind,item in seq if kind == "recolha"}
            for item in items:
                for vj in range(len(sequences)):
                    if vj == vi:
                        continue
                    candidate = [[op for op in s if not (j == vi and op[1] == item)]
                                 for j,s in enumerate(sequences)]
                    candidate[vj].extend([("recolha",item), ("entrega",item)])
                    outcome = evaluate_fleet_sequences(problem, candidate)
                    if outcome and outcome.distance_km < best.distance_km - 1e-9:
                        best, best_sequences = outcome, candidate
        iterations += 1
        if best_sequences is sequences:
            break
        sequences, current = best_sequences, best
        history.append(current.distance_km)
    current.algorithm = "fleet_hill_climbing"
    current.expanded = iterations
    current.runtime_s = time.perf_counter()-start
    current.history = history
    return current


def fleet_csp(problem, max_nodes=50000):
    """Procura por backtracking para encontrar atribuições/ordens válidas.
    Variáveis: ações a atribuir a veículos e sua ordem.
    Domínios: ações permitidas por veículo.
    Restrições: precedência, capacidade e janelas temporais.
    """
    start = time.perf_counter()
    root = tuple(State(v.center_location, problem.start_time)
                 for v in problem.vehicles)
    ids = frozenset(r.item_id for r in problem.requests)
    nodes, backtracks = 0, 0
    def dfs(states):
        nonlocal nodes, backtracks
        delivered = frozenset().union(*(s.delivered for s in states))
        if delivered == ids:
            return states
        if nodes >= max_nodes:
            return None
        nodes += 1
        choices = list(_fleet_candidates(problem, states, ids-delivered))
        choices.sort(key=lambda x: (x[1].moment, x[2], x[0]))
        for vi,nxt,_,_,_ in choices:
            new_states = tuple(nxt if j == vi else s
                               for j,s in enumerate(states))
            answer = dfs(new_states)
            if answer is not None:
                return answer
            if nodes >= max_nodes:
                break
        backtracks += 1
        return None
    answer = dfs(root)
    if answer is None:
        return FleetResult("fleet_csp",False,math.inf,nodes,
                           time.perf_counter()-start,{},0,len(ids),
                           f"Sem solução completa ou limite atingido; backtracks={backtracks}")
    return _fleet_complete(problem,answer,len(ids),nodes,start,"fleet_csp",
                           f"backtracks={backtracks}")


if __name__ == "__main__":
    p = load_fleet_problem("data")
    p.requests = p.requests[:3]
    g = fleet_greedy(p)
    a = fleet_astar(p, max_expansions=2000)
    h = fleet_hill_climbing(p, a if a.feasible else g)
    c = fleet_csp(p, max_nodes=2000)
    print(fleet_summary([g,a,h,c]).to_string(index=False))
