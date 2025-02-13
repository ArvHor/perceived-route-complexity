from asyncio import PriorityQueue
import itertools
import osmnx as ox
import pyproj
import logging
import heapq
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',filename='app.log', filemode='w')
info_handler = logging.FileHandler('info.log')
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
def simplest_path_weight_algorithm_optimized(G, start_node):
    """
    Calculates edge weights using an optimized algorithm with a priority queue.

    Args:
        G: The graph (NetworkX graph).
        start_node: The starting node.

    Returns:
        G: The graph with updated edge weights.
    """

    print(f'{G.size()} edges in the graph')

    # 1. Initialize Edge Pairs
    E_pairs = set()
    S = set()
    E = set(G.edges(keys=False))
    start_node = int(start_node)
    for u, v in G.edges():
        for _, w in G.out_edges(v):
            E_pairs.add(((u, v), (v, w)))

    # 2. Initialize Edge Weights and Priority Queue
    for u, v in G.edges():
        if u == start_node:
            print(f'start node: {u} and end node: {v}')
            G.edges[(u, v, 0)]['decision_complexity'] = 0
        else:
            G.edges[(u, v, 0)]['decision_complexity'] = float('inf')
        

    n_left = len(G.edges())

    while E.difference(S):
        
        min_edge = min(E.difference(S), key=lambda e: G.edges[e[0],e[1],0]['decision_complexity'])
        # Skip edges that have already been processed
        if min_edge in S:
            continue

        S.add(min_edge)

        n_left -= 1
        if n_left % 1000 == 0:
            print(f'{n_left} edges left')

        out_edges = [e for e in G.out_edges(min_edge[1]) if e not in S]

        for out_edge in out_edges:
            u, v = min_edge
            _, w = out_edge
            if ((u, v), (v, w)) in E_pairs:
                decision_complexity, turn_type = calculate_decisionpoint_complexity(G, (u, v), (v, w))
                #logging.info(f'Calculating complexity for edgepair u, v, with complexity {G.edges[(u, v, 0)]['decision_complexity']} adding {decision_complexity}')
                old_complexity = G.edges[(v, w, 0)]['decision_complexity']
                new_complexity = G.edges[(u, v, 0)]['decision_complexity'] + decision_complexity
                if new_complexity < old_complexity:
                    #logging.info(f'Updating edge {(v, w)} with complexity {new_complexity} from edge {(u, v)} with turn type {turn_type}')
                    G.edges[(v, w, 0)]['turn_complexity'] = turn_type
                    G.edges[(v, w, 0)]['decision_complexity'] = new_complexity
    return G



def simplest_path_weight_algorithm(G,start_node):
    """Calculate the weight of the edges in the graph using the simplest path algorithm.
    
    In the simplest path algorithm, the weight of an edge is calculated as the sum of the weights of the edges leading to it.
    ----
    Parameters:
    G: The graph.
    start_node: The origin node of possible paths.
    
    ----
    Returns:
    G: The graph with updated edge weights.
    cs: The evaluation mapping of edges.
    
    
    """
    print(f'{G.size()} edges in the graph, start node: {start_node}')
    
    S = set()
    E = set(G.edges(keys=False)) 
    E_pairs = set()
    cs = {e: float('inf') if e[0] != start_node else 0 for e in E}
    
    for u, v in G.edges():
        G.edges[(u,v,0)]['decision_complexity'] = float('inf')
        if u == start_node:
            print(f'start node: {u} and end node: {v}')
            cs[(u, v)] = 0
            G.edges[(u,v,0)]['decision_complexity'] = 0
        for _, w in G.out_edges(v):
            E_pairs.add(((u, v), (v, w)))
            
    n_left = len(E)
    while E.difference(S):

        min_edge = min(E.difference(S), key=lambda e: cs[e])

        S.add(min_edge)
        n_left -= 1
        if n_left % 1000 == 0:
            print(f'{n_left} edges left')
        u, v = min_edge
        
        out_edges = []
        for edge in G.out_edges(v):
            if edge not in S:
                out_edges.append(edge)

        for out_edge in out_edges:
            _, w = out_edge
            if ((u, v), (v, w)) in E_pairs:
                decision_complexity,turn_type = calculate_decisionpoint_complexity(G, (u, v), (v, w))
                #logging.info(f'Calculating complexity for edge pair {((u, v), (v, w))} with turn type {turn_type}, decision complexity: {decision_complexity}')
                new_complexity = G.edges[(u, v, 0)]['decision_complexity'] + decision_complexity
                old_complexity = G.edges[(v, w, 0)]['decision_complexity']
                min_complexity = min(old_complexity, new_complexity)
                #logging.info(f'min complexity for edge pair {((u, v), (v, w))}: {min_complexity}')
                G.edges[(v, w, 0)]['decision_complexity'] = min_complexity
                cs[(v, w)] = min_complexity

                if new_complexity < old_complexity:
                    #logging.info(f'Updating edge {(v, w)} with complexity {min_complexity} from edge {(u, v)} with turn type {turn_type}')
                    G.edges[(u, v, 0)]['turn_complexity'] = turn_type
            else:
                pass
                #logging.info(f"Edge pair {((u, v), (v, w))} not found in E_pairs")



    total_edges = len(E)
    n_inf_edges = sum(1 for u, v, data in G.edges(data=True) if G.edges[u,v,0]['decision_complexity'] == float('inf'))
    percentage_inf_edges = (n_inf_edges / total_edges) * 100

    max_weight = 0
    for u,v,data in G.edges(data=True):
        if data["decision_complexity"] > max_weight:
            max_weight = data["decision_complexity"]

    #print(f'Percentage of edges with infinite weight_decision_complexity: {percentage_inf_edges:.2f}%')
    logging.info(f'Percentage of edges with infinite decision_complexity: {percentage_inf_edges:.2f}%')
    return G,max_weight


def add_deviation_from_prototypical_weights(G):
    max_weight = 0
    for u in G.nodes():
        if G.out_degree(u) >= 2:
            successor_bearings = get_bearings_to_successors(G,u)
            deviation_weight = calculate_deviation_from_prototypical(successor_bearings)
            if deviation_weight > max_weight:
                max_weight = deviation_weight
            for u,v,k in G.out_edges(u,keys=True):
                G.edges[u,v,k]["deviation_from_prototypical"] = deviation_weight
        else:
            for u,v,k in G.out_edges(u,keys=True):
                G.edges[u,v,k]["deviation_from_prototypical"] = 0

    return G, max_weight

def add_instruction_equivalent_weights(G):
    max_weight = 0
    for u in G.nodes():
        if G.out_degree(u) >= 2:
            successor_bearings = get_bearings_to_successors(G,u)
            instruction_weight = calculate_instruction_equivalent(successor_bearings)
            if instruction_weight > max_weight:
                max_weight = instruction_weight
            for u,v,k in G.out_edges(u,keys=True):
                G.edges[u,v,k]["instruction_equivalent"] = instruction_weight
        else:
            for u,v,k in G.out_edges(u,keys=True):
                G.edges[u,v,k]["instruction_equivalent"] = 0
    return G,max_weight

def add_node_degree_weights(G):
    max_degree = 0
    for u in G.nodes():
        n_degree = G.out_degree(u)
        if n_degree > max_degree:
            max_degree = n_degree

        for u,v,k in G.out_edges(u,keys=True):
            G.edges[u,v,k]["node_degree"] = n_degree
                
    return G,max_degree

def get_turn_type(G, origin, intermediate, destination):
    bearing_origin_to_intermediate = get_azimuth(G, origin, intermediate)
    bearing_intermediate_to_destination = get_azimuth(G, intermediate, destination)

    bearing_difference = (bearing_intermediate_to_destination - bearing_origin_to_intermediate)

    bearing_difference = bearing_difference % 360

    if bearing_difference > 180:
        bearing_difference -= 360
    elif bearing_difference < -180:
        bearing_difference += 360

    if -45 < bearing_difference < 45:
        if -45 <= bearing_difference < -10:
            return 'slight_left_turn'
        elif 10 < bearing_difference <= 45:
            return 'slight_right_turn'
        elif -10 <= bearing_difference <= 10:
            return 'straight'
    elif 45 <= bearing_difference:
        return 'right_turn'
    elif bearing_difference <= -45:
        return 'left_turn'


def is_t_turn(G,u,v):
    
    
    successors = set(G.successors(v))

    if len(successors) > 3 or len(successors) == 1:
        return False

    t_turn_successors = set()
    for w in successors:
            t_turn_successors.add(w)

    if len(successors) == 2 and u in t_turn_successors:
        return False

    if len(successors) == 3 and u in t_turn_successors:
        t_turn_successors.remove(u)
        turns = []
        w1 = t_turn_successors.pop()
        w2 = t_turn_successors.pop()
        turn_type1 = get_turn_type(G,u,v,w1)
        turn_type2 = get_turn_type(G,u,v,w2)
        turns.append(turn_type1)
        turns.append(turn_type2)

        #print(f'Turn type 1: {turn_type1} to {w1}, Turn type 2: {turn_type2} to {w2}')
        if 'left_turn' in turns and 'right_turn' in turns:
            return True
        else:
            return False
        
    elif len(successors) == 2 and u not in t_turn_successors:
        w1 = t_turn_successors.pop()
        w2 = t_turn_successors.pop()
        turn_type1 = get_turn_type(G,w1,v,w2)
        turn_type2 = get_turn_type(G,w1,v,w2)
        if turn_type1 == 'straight' and turn_type2 == 'straight':
            return True
        else:
            return False


def calculate_decisionpoint_complexity(G,e1,e2):
    origin_node = e1[0]
    intermediate_node = e1[1]
    destination_node = e2[1]
    intermediate_node_degree = G.out_degree(intermediate_node)
    turn_type = get_turn_type(G,origin_node,intermediate_node,destination_node)
    t_turn = is_t_turn(G,origin_node,intermediate_node)
    #print(f'Is t turn: {t_turn}')
    turn_complexity = "ERROR"
    #print(f'Turn type: {turn_type}')
    slots = None
    if turn_type == 'straight' or turn_type == 'slight_left_turn' or turn_type == 'slight_right_turn':
        slots = 1
        if intermediate_node_degree == 1:
            turn_complexity = "1_way_straight"
            #bearing_origin_to_intermediate = get_azimuth(G, origin_node, intermediate_node)
            #bearing_intermediate_to_destination = get_azimuth(G, intermediate_node, destination_node)
            #logging.info(f"Error: Straight u-turn at node {intermediate_node}, azimuth from origin to intermediate: {bearing_origin_to_intermediate}, azimuth from intermediate to destination: {bearing_intermediate_to_destination}")
        elif intermediate_node_degree == 2:
            turn_complexity = "2_way_straight"
        elif intermediate_node_degree >= 3:
            turn_complexity = "N_way_straight"

    elif turn_type == 'left_turn' or turn_type == 'right_turn':
        if intermediate_node_degree == 1:
            turn_complexity = "1_way-turn"
            slots = 4
        elif intermediate_node_degree == 2:
            turn_complexity = "2_way_turn"
            slots = 4
        elif intermediate_node_degree == 3 and t_turn:
            #logging.info(f"T-Turn at node {intermediate_node} yay")
            turn_complexity = "t_turn"
            slots = 6
        elif intermediate_node_degree == 3 and not t_turn:
            turn_complexity = "3_way_turn"
            slots = 5 + intermediate_node_degree
        elif intermediate_node_degree > 3:
            turn_complexity = "N_way_turn"
            slots = 5 + intermediate_node_degree
    if slots is None:
        print(f"Error calculating turn complexity for edge pair {e1} and {e2}, turn type: {turn_type}, intermediate node degree: {intermediate_node_degree}")
    return slots, turn_complexity


def get_bearings_to_successors(G,node):
    successors = list(G.successors(node))
    bearing_list = []
    for successor in successors:
        fwd_azimuth = get_azimuth(G, node, successor) 
        bearing_list.append(fwd_azimuth)
        
    return bearing_list
    
def calculate_instruction_equivalent(bearing_list):
    """How many turns at a decision point can be described with the same linguistic label?"""
    bearing_difference_list = []
    if len(bearing_list) > 1:
        for bearing_a, bearing_b in itertools.combinations(bearing_list, 2):
            bearing_difference = bearing_b - bearing_a
            bearing_difference = bearing_difference % 360

            if bearing_difference > 180:
                bearing_difference -= 360 
            elif bearing_difference < -180:
                bearing_difference += 360

            bearing_difference_list.append(bearing_difference)
        
        if bearing_difference_list:
            zero_to_ninety = len([bearing for bearing in bearing_difference_list if 0 < bearing < 90])
            ninety_to_oneeighty = len([bearing for bearing in bearing_difference_list if 90 < bearing < 180])
            minus_ninety_to_zero = len([bearing for bearing in bearing_difference_list if -90 < bearing < 0])
            minus_oneeighty_to_minus_ninety = len([bearing for bearing in bearing_difference_list if -180 < bearing < -90])
            max_count = max(zero_to_ninety, ninety_to_oneeighty, minus_ninety_to_zero, minus_oneeighty_to_minus_ninety, 1)
            return max_count
        else:
            return 1
    else:
        return 1
    
def calculate_deviation_from_prototypical(bearing_list):
    """Calculate the deviation of the bearing from the closest cardinal direction.
    
    ----
    Parameters:
    bearing: Forward azimuth bearings from a node to its successors.
    
    ----
    Returns:
    Deviation value: The deviation of the bearing from the closest cardinal direction.
    
    """
    deviation_list = []
    if len(bearing_list) > 1:
        for bearing_a, bearing_b in itertools.combinations(bearing_list, 2):
            bearing_difference = bearing_a - bearing_b
            bearing_difference = bearing_difference % 360
            if bearing_difference > 180:
                bearing_difference -= 360
            elif bearing_difference < -180:
                bearing_difference += 360
            deviation = calculate_deviation(bearing_difference)
            deviation_list.append(deviation)
        #print(deviation_list)
        avg_deviations = sum(deviation_list) / len(deviation_list)
    else:
        avg_deviations = 0
    return avg_deviations



def calculate_deviation(bearing):
    if bearing < 0:
        if  bearing >= -90:
            a = -90 - bearing
            deviation = abs(min(a, bearing))
            return deviation
        elif bearing >= -180:
            a = -180 - bearing
            b = -90 - bearing
            deviation = abs(min(a, b))
            return deviation
    elif bearing > 0:
        if bearing <= 90:
            a = 90 - bearing
            deviation = abs(min(a, bearing))
            return deviation
        elif bearing <= 180:
            a = 180 - bearing
            b = 90 - bearing
            deviation = abs(min(a, b))
            return deviation
    return 0

def get_azimuth(G, node_a, node_b, return_all=False):
    """Calculate the azimuth and distance between two points.
    
    Pyproj uses an equidistant azimuthal projection with the north pole as the center of a flat circle.
    This means that the azimuth
    ----
    Parameters:
    point_a: The coordinates of the first point.
    point_b: The coordinates of the second point.
    
    ----
    Returns:
    fwd_azimuth: The forward azimuth from point_a to point_b.
    back_azimuth: The back azimuth from point_b to point_a.
    distance: The distance between the two points.
    
    """
    point_a = [G.nodes[node_a]['y'], G.nodes[node_a]['x']]
    point_b = [G.nodes[node_b]['y'], G.nodes[node_b]['x']]
    if ox.projection.is_projected(G.graph['crs']):
        lat1, lat2, long1, long2 = point_a[0], point_b[0], point_a[1], point_b[1]
        transformer = pyproj.Transformer.from_crs(G.graph['crs'], "EPSG:4326", always_xy=True)
        lon1, lat1 = transformer.transform(long1, lat1)
        lon2, lat2 = transformer.transform(long2, lat2)
        geodesic = pyproj.Geod(ellps='WGS84')
        fwd_azimuth, back_azimuth, distance = geodesic.inv(lon1, lat1, lon2, lat2)
        if return_all:
            return fwd_azimuth, back_azimuth, distance
        else:
            return fwd_azimuth
    else:
        lat1, lat2, long1, long2 = point_a[0], point_b[0], point_a[1], point_b[1]
        geodesic = pyproj.Geod(ellps='WGS84')
        fwd_azimuth, back_azimuth, distance = geodesic.inv(long1, lat1, long2, lat2)
        if return_all:
            return fwd_azimuth, back_azimuth, distance
        else:
            return fwd_azimuth