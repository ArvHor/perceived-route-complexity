
import random
import networkx as nx
import osmnx as ox
import pyproj
import network_properties as sn
import networkx.algorithms.isomorphism as iso


def get_possible_paths(G, origin_node, target_distance, margin, model, origin_name):
    possible_paths = []
    #G = ox.project_graph(to_crs=None)
    path = {}
    distant_nodes = get_distant_nodes(G, origin_node, target_distance, margin)
    print(f'origin node: {origin_node}, distant nodes: {distant_nodes}')
    for node in distant_nodes:
        if node != origin_node and nx.has_path(G, node, origin_node):
            if model == 'shortest':        
                unchecked_path = get_dijkstra_path(G, origin=origin_node, destination=node, cost_string='length')
                if is_targetlength(G, unchecked_path["nodes"], target_distance, margin):
                    path = unchecked_path
                    path['start_node'] = origin_node
                    path['end_node'] = node
                else:
                    continue    
            elif model == 'least_decision_complex':
                unchecked_path = get_dijkstra_path(G, origin=node, destination=origin_node, cost_string='normalized_decision_complexity')
                if is_targetlength(G, unchecked_path['nodes'], target_distance, margin):
                    path = unchecked_path
                    path['start_node'] = origin_node
                    path['end_node'] = node
                else:
                    continue  
            path['city_name'] = G.graph['city_name']
            path['city_name'] = G.graph['city_name']
            path['distance'] = get_edges_sum(G, path['edges'], 'length')
            path['avg_complexity'] = get_edges_avg(G, path['edges'], 'normalized_decision_complexity')
            path['sum_decision_complexity'] = get_edges_sum(G, path['edges'], 'normalized_decision_complexity')
            #path['sum_complexity'] = get_edges_sum(G, path['edges'], 'combined_complexity_weight')
            path['n_decisionpoints'] = len(path['nodes'])
            path['order_category'] = G.graph['order_category']
            #path['graph_filepath'] = G.graph['graph_filepath']
            path['origin_name'] = origin_name

            if model == 'shortest':
                path['model'] = 'shortest'
            elif model == 'least_complex':
                path = get_dijkstra_path(G,  origin=origin_node, destination=node, cost_string='complexity_weight')
            elif model == 'least_decision_complex':
                path['model'] = 'least_decision_complex'

            path['unique_id'] = hash(tuple(path['nodes']))

            possible_paths.append(path)
    
    if len(possible_paths) == 0:
        print('No possible paths found')
        return None
    else:
        return possible_paths


def get_distant_nodes(G, origin_node, target_distance, margin):
    max_dist = target_distance + (target_distance * margin) + 500
    distant_nodes = []
    for node in G.nodes:
        _,_, dist = get_azimuth(G, origin_node, node)
        if dist <= max_dist:
            distant_nodes.append(node)
    return distant_nodes

def get_azimuth(G, node_a, node_b):
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
        return fwd_azimuth, back_azimuth, distance
    else:
        lat1, lat2, long1, long2 = point_a[0], point_b[0], point_a[1], point_b[1]
        geodesic = pyproj.Geod(ellps='WGS84')
        fwd_azimuth, back_azimuth, distance = geodesic.inv(long1, lat1, long2, lat2)
        return fwd_azimuth, back_azimuth, distance

def is_targetlength(graph, path, target_length, margin, weight='length'):
    edge_values = []
    for i in range(len(path) - 1):
        source = path[i]
        target = path[i + 1]
        edge_data = graph.get_edge_data(source, target,0)
        edge_values.append(edge_data.get(weight))
    path_length = sum(edge_values)
    if path_length <= target_length + (target_length * margin) and path_length >= target_length - (target_length * margin):
        return True
    else:
        return False

def get_edges_avg(G, edges, weight):
    edges_avg = 0
    for edge in edges:
        edges_avg += float(G.edges[edge].get(weight, 0))
    return edges_avg / len(edges)

def get_edges_sum(G, edges, weight):
    edges_sum = 0
    for edge in edges:
        edge_length = float(G.edges[edge].get(weight, 0))
        #print(f'edge: {edge}, weight: {edge_length}')
        edges_sum += edge_length
    #print(f'edges_sum: {edges_sum}')
    return edges_sum

def find_equivalent_paths(G, target_distance, target_complexity, length_error_margin,complexity_error_margin, n_paths):
    paths = []
    
    for i in range(n_paths):
        random_path = get_random_path(G, target_distance, length_error_margin, 'least_complex')
        random_path_complexity = get_edges_avg(G, random_path['edges'], 'env_complexity_weight')
        while (random_path_complexity <= target_complexity - (target_complexity * complexity_error_margin) and random_path_complexity >= target_complexity + (target_complexity * complexity_error_margin)):
            random_path = get_random_path(G, target_distance, length_error_margin, 'least_complex')
            random_path_complexity = get_edges_avg(G, random_path['edges'], 'env_complexity_weight')
        paths.append(random_path['nodes'])
    
    return paths

def get_path_weight(G, path, weight):
    path_weight = 0
    for i in range(len(path) - 1):
        for key in G[path[i]][path[i + 1]]:
            path_weight += G.edges[path[i], path[i + 1], key].get(weight, 0)
    return path_weight

def get_random_path(G, target_distance, margin, search_model):
    dist_min = target_distance - (target_distance * margin)
    dist_max = target_distance + (target_distance * margin)
    source = random.choice(list(G.nodes()))
    destination = random.choice(list(G.nodes()))
    _,_,dist = sn.get_azimuth(G,[G.nodes[source]['y'], G.nodes[source]['x']],[G.nodes[destination]['y'], G.nodes[destination]['x']])
    while (dist < dist_min or dist > dist_max) or not nx.has_path(G, source, destination):
        print(dist)
        source = random.choice(list(G.nodes()))
        destination = random.choice(list(G.nodes()))
        _,_,dist = sn.get_azimuth(G,[G.nodes[source]['y'], G.nodes[source]['x']],[G.nodes[destination]['y'], G.nodes[destination]['x']])
    if search_model == 'least_complex':
        path = path_finder(G, 2,source, destination,traffic=False)
    elif search_model == 'least_distance':
        path = path_finder(G, 1,source, destination,traffic=False)
    print('great circle distance: ', dist)
    return path


def has_acute_angle(G, source, destination):
    angle = ox.bearing.calculate_bearing(G.nodes[source]['y'], G.nodes[source]['x'], G.nodes[destination]['y'], G.nodes[destination]['x'])
    if abs(angle - 0) > 20 and abs(angle - 90) > 20 and abs(angle - 180) > 20 and abs(angle - 270) > 20:
        return True
    else:
        return False


def get_dijkstra_path_dict(MultiDigraph, origin, destination, cost_string):
    

    path = nx.dijkstra_path(MultiDigraph, origin, destination, weight=cost_string)
    path_weights_sum = 0
    
    edges = []
    for i in range(len(path) - 1):
        # Get the key of the edge with the minimum weight
        min_weight = float('inf')
        min_key = None
        for key in MultiDigraph[path[i]][path[i + 1]]:
            weight = MultiDigraph.edges[path[i], path[i + 1], key].get(cost_string, 0)
            if weight < min_weight:
                min_weight = weight
                min_key = key
        edges.append((path[i], path[i + 1], min_key)) 
        weight = MultiDigraph.edges[path[i], path[i + 1], min_key].get(cost_string, 0)
        path_weights_sum += min_weight
        
    path = {
        'model_weight': cost_string,
        'nodes': path,
        'edges': edges,
        'weight': path_weights_sum
    }
    #print(path['edges'])
    return path