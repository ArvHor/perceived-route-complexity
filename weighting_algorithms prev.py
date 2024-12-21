import itertools
import math
import osmnx as ox
import networkx as nx
import pyproj
import pandas as pd
import heapq


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
    print(f'{G.size()} edges in the graph')
    
    S = set()
    E = set(G.edges(keys=False)) 
    E_pairs = set()
    cs = {e: float('inf') if e[0] != start_node else 0 for e in E}
    
    for u, v in G.edges():
        G.edges[(u,v,0)]['weight_decision_complexity'] = float('inf')
        _,_,G.edges[(u,v,0)]['weight_distance'] = get_azimuth(G,u,v,return_all=True)
        G.edges[(u,v,0)]['weight_instruction_equivalent'] = calculate_instruction_equivalent(G,u)
        G.edges[(u,v,0)]['weight_deviation_from_prototypical'] = calculate_deviation_from_prototypical(G,u)
        
        if u == start_node:
            print(f'start node: {u} and end node: {v}')
            cs[(u, v)] = 0
            G.edges[(u,v,0)]['weight_decision_complexity'] = 0
            G.edges[(u,v,0)]['weight_instruction_equivalent'] = 0
            G.edges[(u,v,0)]['weight_deviation_from_prototypical'] = 0
            G.edges[(u,v,0)]['weight_distance'] = 0
        for _, w in G.out_edges(v):

            E_pairs.add(((u, v), (v, w)))


    #edge_heap = [(G.edges[edge[0], edge[1], 0]['weight_decision_complexity'], edge) for edge in E]
    #heapq.heapify(edge_heap)
    #print(f'Heap size: {len(edge_heap)}')
    print(f'edge set size: {len(E)}')

    while E.difference(S):
        min_edge = min(E.difference(S), key=lambda e: cs[e])

        S.add(min_edge)
        u, v = min_edge
        
        out_edges = []
        for edge in G.out_edges(v):
            if edge not in S:
                out_edges.append(edge)

        for out_edge in out_edges:
            _, w = out_edge
            if ((u, v), (v, w)) in E_pairs:
                decision_complexity = calculate_decisionpoint_complexity(G, (u, v), (v, w))
                new_complexity = G.edges[(u, v, 0)]['weight_decision_complexity'] + decision_complexity
                old_complexity = G.edges[(v, w, 0)]['weight_decision_complexity']
                min_complexity = min(old_complexity, new_complexity)
                print(f'min complexity for edge pair {((u, v), (v, w))}: {min_complexity}')
                G.edges[(v, w, 0)]['weight_decision_complexity'] = min_complexity
                cs[(v, w)] = min_complexity
                #heapq.heappush(edge_heap, (min_complexity, (v, w)))
            else:
                print(f"Edge pair {((u, v), (v, w))} not found in E_pairs")



    total_edges = len(E)
    inf_edges = sum(1 for u, v, data in G.edges(data=True) if G.edges[u,v,0]['weight_decision_complexity'] == float('inf'))
    percentage_inf_edges = (inf_edges / total_edges) * 100

    print(f'Percentage of edges with infinite weight_decision_complexity: {percentage_inf_edges:.2f}%')
    return G


def simplest_path_retrieval(G,cs,start_node,end_node):
    path = [end_node]
    current_vertex = start_node
    
    # Continue until we reach the start vertex
    while current_vertex != start_node:
        # Find incoming edge with minimum complexity score
        min_complexity = float('inf')
        best_previous_vertex = None
        
        # Check all incoming edges to current vertex
        for predecessor in G.predecessors(current_vertex):
            edge = (predecessor, current_vertex)
            if edge in cs:  # Check if edge exists in complexity scores
                if cs[edge] < min_complexity:
                    min_complexity = cs[edge]
                    best_previous_vertex = predecessor
        
        # If no valid previous vertex found, path doesn't exist
        if best_previous_vertex is None:
            raise ValueError(f"No valid path found from {start_node} to {end_node}")
        
        # Prepend the best previous vertex to the path
        path.insert(0, best_previous_vertex)
        current_vertex = best_previous_vertex
    
    return path
    




def add_deviation_prototypical_weight(G):
    for u, v, data in G.edges(data=True):
        if len(G.out_edges(u)) > 2:
            data['weight_deviation_from_prototypical'] = calculate_deviation_from_prototypical(G,u)
    return G
def add_instruction_equivalent_weight(G):
    for u, v, data in G.edges(data=True):
        if len(G.out_edges(u)) > 2:
            data['weight_instruction_equivalent'] = calculate_instruction_equivalent(G,u)
    return G

def get_turn_type(G, origin, intermediate, destination):
    bearing_origin_to_intermediate = get_azimuth(G, origin, intermediate)
    bearing_intermediate_to_destination = get_azimuth(G, intermediate, destination)
    bearing_difference = (bearing_origin_to_intermediate - bearing_intermediate_to_destination)

    bearing_difference = bearing_difference % 360

    if bearing_difference > 180:
        bearing_difference -= 360  # Normalize to -180 to 180

    if -45 < bearing_difference < 45:
        return 'straight'
    elif 45 <= bearing_difference:
        return 'left_turn'
    elif bearing_difference <= -45:
        return 'right_turn'


def is_t_turn(G,u,v):
    node_degree = len(set(G.out_edges(v)))
    if node_degree == 3:
        intersection_turns = []
        turn_bearings = []
        bearing_uv = get_azimuth(G,u,v)
        for _,w in G.out_edges(v):
            if w == u:
                continue
            bearing_vw = get_azimuth(G,v, w)
            turn_bearings.append(bearing_vw)
            intersection_turns.append(get_turn_type(G,u,v,w))
        
        if 'left_turn' in intersection_turns and 'right_turn' in intersection_turns:
            return True
        else:
            return False
    else:
        return False
    

def calculate_decisionpoint_complexity(G,e1,e2):
    origin_node = e1[0]
    intermediate_node = e1[1]
    destination_node = e2[1]
    intermediate_node_degree = len(G.out_edges(intermediate_node))
    turn_type = get_turn_type(G,origin_node,intermediate_node,destination_node)
    t_turn = is_t_turn(G,intermediate_node,destination_node)
    #print(f'Is t turn: {t_turn}')
    
    #print(f'Turn type: {turn_type}')
    slots = 20
    if turn_type == 'straight':
        slots = 1
    elif turn_type == 'left_turn' or turn_type == 'right_turn':
        if intermediate_node_degree == 2:
            slots = 4
        elif intermediate_node_degree == 3 and t_turn:
            #print('T-turn')
            slots = 6
        elif intermediate_node_degree > 3:
            slots = 5 + intermediate_node_degree
    #print(f'Slots: {slots}')
    return slots

def calculate_edgepair_weight(G,e1,e2):
    origin_node = e1[0]
    intermediate_node = e1[1]
    destination_node = e2[1]
    bearing_weight = 1/360
    bearing_origin_intermediate,_,_ = get_azimuth(G, origin_node, intermediate_node)
    bearing_intermediate_destination,_,_ = get_azimuth(G, intermediate_node, destination_node)
    bearing_difference = abs(bearing_origin_intermediate - bearing_intermediate_destination)
    
    return bearing_difference*bearing_weight

def get_bearings_to_successors(G,node):
    successors = list(G.successors(node))
    bearing_list = []
    for successor in successors:
        fwd_azimuth = get_azimuth(G, node, successor) 
        bearing_list.append(fwd_azimuth)
        
    return bearing_list
    
def calculate_instruction_equivalent(G,node):
    """How many turns at a decision point can be described with the same linguistic label?"""
    bearing_list = get_bearings_to_successors(G,node)
    bearing_difference_list = []
    if len(bearing_list) > 1:
        for bearing_a, bearing_b in itertools.combinations(bearing_list, 2):
            bearing = abs(bearing_a - bearing_b)
            bearing_difference_list.append(bearing)
        
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
    
def calculate_deviation_from_prototypical(G,node):
    """Calculate the deviation of the bearing from the closest cardinal direction.
    
    ----
    Parameters:
    bearing: Forward azimuth bearings from a node to its successors.
    
    ----
    Returns:
    Deviation value: The deviation of the bearing from the closest cardinal direction.
    
    """
    bearing_list = get_bearings_to_successors(G,node)
    deviation_list = []
    if len(bearing_list) > 1:
        for bearing_a, bearing_b in itertools.combinations(bearing_list, 2):
            bearing = abs(bearing_a - bearing_b)
            deviation = calculate_deviation(bearing)
            deviation_list.append(deviation)
        #print(deviation_list)
        avg_deviations = sum(deviation_list) / len(deviation_list)
    else:
        avg_deviations = 0
    return avg_deviations
    
def calculate_deviation(bearing):
    if bearing <= 90:
        a = 90 - bearing
        return min(a, bearing)
    elif bearing <= 180:
        a = 180 - bearing
        b = bearing - 90
        return min(a, b)
    elif bearing <= 270:
        a = 270 - bearing
        b = bearing - 180
        return min(a, b)
    elif bearing <= 360:
        a = 360 - bearing
        b = bearing - 270
        return min(a, b)
    else:
        return 0 

def get_azimuth(G, node_a, node_b, return_all=False):
    """Calculate the azimuth and distance between two points.
    
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

    
    

