import math
import os
import osmnx as ox
import urllib.request  
from urllib.error import HTTPError
import xml.sax
import networkx as nx
from statistics import mean
import itertools
import pyproj
import requests
import random
import time
import osmnx as ox
from shapely.geometry import Polygon
#from routes import get_bing_route

def calculate_bbox_entropy(G, bbox_tuple, city_name,condition):

    G = ox.graph_from_bbox(bbox=bbox_tuple, network_type='all', simplify=False, retain_all=True, truncate_by_edge=False)
    G = ox.convert.to_undirected(G)
    G = ox.add_edge_bearings(G)
    # Truncate the graph to the bounding box
    #G = ox.truncate.truncate_graph_bbox(G,bbox=bbox, retain_all=True, truncate_by_edge=False)
    
    entropy = ox.bearing.orientation_entropy(G, num_bins=36, min_length=0)
    order = calculate_orientation_order(entropy, num_bins=36)
    
    result = {'city_name':city_name,'entropy':entropy, 'order':order, 'condition':condition}
    return result


def calculate_local_entropy(G, node):
    
    # Calculate the coordinates of the bounding box
    print((G.nodes[node]['y'], G.nodes[node]['x']))
    city_name = G.graph['city_name']
    bbox = ox.utils_geo.bbox_from_point((G.nodes[node]['y'], G.nodes[node]['x']), dist=2000)

    G = ox.graph_from_bbox(bbox=bbox, network_type='all', simplify=False, retain_all=True, truncate_by_edge=False)
    G = ox.convert.to_undirected(G)
    G = ox.add_edge_bearings(G)
    # Truncate the graph to the bounding box
    #G = ox.truncate.truncate_graph_bbox(G,bbox=bbox, retain_all=True, truncate_by_edge=False)
    
    entropy = ox.bearing.orientation_entropy(G, num_bins=36, min_length=0)
    order = calculate_orientation_order(entropy, num_bins=36)
    
    result = {'city_name':city_name,'entropy':entropy, 'order':order}
    return result

def route_intersection_orientation_entropy(G, route, num_bins=36, min_length=0):
    """
    Calculate the orientation entropy of the route.
    
    Parameters:
    G (networkx.Graph): The graph.
    route (list): The list of nodes in the route.
    num_bins (int): The number of bins to divide the orientation into.
    min_length (int): The minimum length of a route segment.
    
    Returns:
    entropy (float): The orientation entropy of the route.
    """
    # Calculate the orientation entropy of the route
    neighbors = []
    for i in range(len(route)):
        neighbors.append(G.neighbors(route[i]))
    # Calculate the orientation entropy of the route
    subgraph = G.subgraph(list[set(neighbors+route)])
    
    entropy = ox.bearing.orientation_entropy(subgraph, num_bins=num_bins, min_length=min_length)
    
    max_nats = math.log(num_bins)
    min_nats = math.log(2)
    orientation_order = 1 - ((entropy - min_nats) / (max_nats - min_nats))**2
    
    return entropy, orientation_order

def calculate_orientation_order(entropy, num_bins=36):
    # Natural logarithm of the number of bins, and theoretical minimum number of bins
    max_nats = math.log(num_bins)
    min_nats = math.log(4)
    orientation_order = 1 - ((entropy - min_nats) / (max_nats - min_nats))**2

    return orientation_order

def calculate_avg_route_complexity(G, n_route_cutoff):
    """
    Calculate the average edge complexity of n simple paths in the graph G.
    
    Parameters:
    G (networkx.Graph): The graph.
    n_route_cutoff (int): The number of simple paths to consider.
    
    Returns:
    avg_complexity (float): The average edge complexity of the n simple paths.
    """
    # Get all simple paths in the graph
    simple_paths = nx.all_simple_paths(G, source=None, target=None, cutoff=None)
    
    # Select n simple paths
    selected_paths = itertools.islice(simple_paths, n_route_cutoff)
    
    # Calculate the edge complexity for each selected path
    complexities = []
    for path in selected_paths:
        path_complexity = calculate_path_complexity(G, path)
        complexities.append(path_complexity)
    
    # Calculate the average edge complexity
    avg_complexity = mean(complexities)
    
    return avg_complexity
    
    
def create_walking_decisionpoints_multigraph(city_name,bbox_tuple):
    """Expects a bounding box and returns a MultiDiGraph with nodes classified as decision points and weights assigned to edges."""  
    
    print("Creating directed multigraph using osmnx...")
    print(f"Bounding box: {bbox_tuple}")
    G = ox.graph_from_bbox(bbox=bbox_tuple, network_type="all", truncate_by_edge=True, simplify=False, retain_all=False)
    #G = ox.project_graph(G, to_crs="EPSG:4326")
    #G = ox.consolidate_intersections(G, tolerance=2,rebuild_graph=True,dead_ends=False,reconnect_edges=True)
    G.graph['name'] = city_name

    for node_id in G.nodes():
        if len(list(G.successors(node_id))) > 1:
            G.nodes[node_id]['is_decision_point'] = True
            
    # The distance and bearing of each edge is calculated using OSMNX built-in functions, these are not used in the final model
    
    # Calculate the parameters for each decision point and the edge for each decision point
    G = set_decisionpoint_parameters(G)    
    
    # For every edge that has no distance, assign the distance between the nodes, or calculate the distance between the nodes
    for (node1, node2) in G.edges():
        for edge_key in G[node1][node2]:
            if 'distance' not in G.edges[node1, node2, edge_key] and 'distance' in G.edges[node2, node1,edge_key]:
                G.edges[node1, node2,edge_key]['distance'] = G.edges[node2, node1,edge_key]['distance']
                way_id = G.edges[node2, node1,edge_key]['osmid']
                way = G.edges[way_id]
                
    
    
    return G

def set_decisionpoint_parameters(G, landmark=True):
    """
    For every node in the graph:
            - set lat, long and id
            - Calculate the number of successors
            - If the node has two or more successors:
                - For every combination of two successors:
                    - Calculate the difference in bearing between the two successors
                    - Calculate the distance to successors
                    - Calculate and assign distance and bearing to all outgoing edges
            - If the node has one successor: 
                - Calculate the distance to successors
                - Calculate and assign distance and bearing to all outgoing edges
            - Get the instruction equivalence complexity of the node
            - Get the instruction complexity of the node
            - Get the landmark complexity of the node    
    """
    n_nodes_left = len(G.nodes)
    print("is multigraph", G.is_multigraph())
    for dp_node_id in G.nodes():
        
        n_nodes_left = n_nodes_left - 1 
        print( f"Calculating parameters for node {dp_node_id}, nodes left to calculate: {n_nodes_left}")
        
        DP = G.nodes[dp_node_id]
        

        successor_list = list(G.successors(dp_node_id))
        n_successors = len(successor_list)
        
        
        G.nodes[dp_node_id]['lat'] = DP['y']
        G.nodes[dp_node_id]['lon'] = DP['x']
        G.nodes[dp_node_id]['id'] = dp_node_id
        G.nodes[dp_node_id]['n_successors'] = n_successors

        deviation_list = []
        bearing_differences_list = []
        neighbor_distance_list = []
        
        # If the decision point has two or more successors, calculate the bearing and distance between all combinations of the two successors
        if n_successors >= 2:

            G.nodes[dp_node_id]['isDecisionPoint'] = True
            for successor_a, successor_b in itertools.combinations(successor_list, 2):
                node_a, node_b = G.nodes[successor_a], G.nodes[successor_b]
                pointA, pointDP, pointB = [node_a['y'], node_a['x']], [DP['y'], DP['x']], [node_b['y'], node_b['x']]
                
                
                fwd_azimuth_DPA, _, distance_DPA = get_azimuth(G,pointDP, pointA)
                neighbor_distance_list.append(distance_DPA)
                fwd_azimuth_DPB, _, distance_DPB = get_azimuth(G,pointDP, pointB)
                neighbor_distance_list.append(distance_DPB)
                
                bearing = abs(fwd_azimuth_DPA - fwd_azimuth_DPB)
                bearing_differences_list.append(bearing)
                deviation_list.append(calculate_deviation(bearing))

                if G.has_edge(dp_node_id, successor_a):
                    for edge_key in G[dp_node_id][successor_a]:
                        G.edges[dp_node_id,successor_a,edge_key]['bearing'] = fwd_azimuth_DPA
                        G.edges[dp_node_id,successor_a,edge_key]['distance'] = distance_DPA
                        
                if G.has_edge(dp_node_id, successor_b):
                    for edge_key in G[dp_node_id][successor_b]:
                        G.edges[dp_node_id,successor_b,edge_key]['bearing'] = fwd_azimuth_DPB
                        G.edges[dp_node_id,successor_b,edge_key]['distance'] = distance_DPB

                    
        # If the decision point has only one neighbor, calculate the azimuth and distance between the decision point and the neighbor
        elif n_successors == 1:
            G.nodes[dp_node_id]['isDecisionPoint'] = False
            successor_a = successor_list[0]
            node_a = G.nodes[successor_a]
            pointA, pointDP = [node_a['y'], node_a['x']], [DP['y'], DP['x']]
            fwd_azimuth_DPA, _, distance_DPA = get_azimuth(G, pointDP, pointA)
            neighbor_distance_list.append(distance_DPA)

            if G.has_edge(dp_node_id, successor_a):
                for edge_key in G[dp_node_id][successor_a]:
                    G.edges[dp_node_id, successor_a, edge_key]['bearing'] = fwd_azimuth_DPA
                    G.edges[dp_node_id, successor_a, edge_key]['distance'] = distance_DPA


                
        # If the decision point has no neighbors, set the average deviation to 0           
        G.nodes[dp_node_id]['avgDeviation'] = mean(deviation_list) if deviation_list else 0
        

        bearing_list = []
        for neighbor in successor_list:
            if G.has_edge(dp_node_id, neighbor):
                for edge_key in G[dp_node_id][neighbor]:
                    bearing = G.edges[dp_node_id, neighbor,edge_key]['bearing']
            bearing_list.append(bearing)
        
        # Calculate the instruction complexity, instruction equivalence, and landmark availability
        G.nodes[dp_node_id]['instComplexity'] = calculate_instruction_complexity(bearing_differences_list, n_successors)
        G.nodes[dp_node_id]['instEquivalent'] = calculate_instruction_equivalent(bearing_list)
        if landmark:
            G.nodes[dp_node_id]['landmark'] = calculate_landmark(G,DP, neighbor_distance_list)         
        #print(f"instcomplexity:{G.nodes[dp_node_id]['instComplexity']} instequivalent: {G.nodes[dp_node_id]['instEquivalent']} landmark: {G.nodes[dp_node_id]['landmark']}")
    
    return G

def get_nearest_node_with_successors(G, point):
    nearest_node = ox.distance.nearest_nodes(G, point[1], point[0])
    while G.out_degree(nearest_node) == 0:
            G.remove_node(nearest_node)
            nearest_node = ox.distance.nearest_nodes(G, point[1], point[0])
            
    return nearest_node


def get_azimuth(G, point_a, point_b):
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


def calculate_deviation(bearing):
    """Calculate the deviation of the bearing from the closest cardinal direction.
    
    ----
    Parameters:
    bearing: The bearing of the edge.
    
    ----
    Returns:
    Deviation value: The deviation of the bearing from the closest cardinal direction.
    
    """
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


def calculate_instruction_equivalent(bearing_list):
    """How many turns at a decision point can be described with the same linguistic label?"""
    if bearing_list:
        zero_to_ninety = len([bearing for bearing in bearing_list if 0 < bearing < 90])
        ninety_to_oneeighty = len([bearing for bearing in bearing_list if 90 < bearing < 180])
        minus_ninety_to_zero = len([bearing for bearing in bearing_list if -90 < bearing < 0])
        minus_oneeighty_to_minus_ninety = len([bearing for bearing in bearing_list if -180 < bearing < -90])
        max_count = max(zero_to_ninety, ninety_to_oneeighty, minus_ninety_to_zero, minus_oneeighty_to_minus_ninety, 1)
        return max_count
    else:
        return 1


def calculate_instruction_complexity(bearing_differences_list, num_of_branches):
    """Calculate the instruction complexity of a decision point. 
    Using the bearing differences between the edges to determine how many slots the decision point has for instructions.
    
    ----
    Parameters:
    bearing_differences_list: List of the differences in bearing between the edges.
    num_of_branches: Number of branches at the decision point.
    
    ----
    Returns:
    Complexity value: The instruction complexity of the decision point.
    
    """
    
    if bearing_differences_list:
        complexity_list = []
        for bearing_diff in bearing_differences_list:
            if -10 <= bearing_diff <= 10:
                complexity_list.append(1)
            else:
                if num_of_branches == 3:
                    complexity_list.append(6)
                else:
                    complexity_list.append(5 + num_of_branches)
        return mean(complexity_list)
    else:
        return 0


def calculate_landmark(G,DP, neighbor_distance_list):
    """Calculate the number of salient landmarks are around a decision point and how far away they are.
    ----
    Parameters:
    DP: Decision point node.
    neighbor_distance_list: List of distances between the decision point and its neighbors.
    
    ----
    Returns:
    Amenity value: The sum of the weights of the landmarks around the decision point.
    """
        
    global response
    AMENITY_TYPE_LIST = [
        'others', 'school', 'restaurant', 'police', 'park',
        'hotel', 'hospital', 'embassy', 'cinema', 'cafe', 'bank'
    ]

    W1 = 1
    W2 = 0

    if len(neighbor_distance_list) != 0:
        around_value = min(50, (max(neighbor_distance_list) * 1000) / 2)
    else:
        around_value = 50

    amenity_salience_list = []
    amenity_distance_list = []
    amenity_value = 0

    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
        [out:json];
        (node["amenity"](around:{}, {}, {}); 
         way["amenity"](around:{}, {}, {});
         rel["amenity"](around:{}, {}, {});
        );
        out center;
        """.format(around_value, DP['y'], DP['x'], around_value, DP['y'], DP['x'], around_value, DP['y'], DP['x'])
    headers = {'referer': 'ArvidHorned_referer'}
    retry = True
    while retry:
        response = requests.get(overpass_url, params={'data': overpass_query}, headers=headers)
        if response.ok and 'json' in response.headers.get('Content-Type'):
            retry = False
            #print('landmark api: success')
            #print(DP['id'])
        else:
            #print('landmark api: fail')
            #print(DP['id'])
            time.sleep(2)

    data = response.json()
    for element in data['elements']:
        if element['type'] == 'node':
            lon = element.get('lon', element['lon'])
            lat = element.get('lat', element['lat'])
            coords_1 = (lat, lon)
        elif element['type'] == 'way':
            lon = element.get('lon', element['center']['lon'])
            lat = element.get('lat', element['center']['lat'])
        elif element['type'] == 'relation':
            continue
            
        coords_1 = (lat, lon)
        
        coords_2 = (DP['y'], DP['x'])
        fwd_azimuth, back_azimuth, distance = get_azimuth(G,coords_1, coords_2)
        amenity_distance_list.append(distance)

        tags = element['tags']
        amenity_type = tags.get('amenity')
        if amenity_type in AMENITY_TYPE_LIST:
            amenity_salience_list.append(AMENITY_TYPE_LIST.index(amenity_type) + 1)
        else:
            amenity_salience_list.append(1)

    amenity_distance_list_max = max(amenity_distance_list, default=1)
    for i in range(len(amenity_distance_list)):
        if amenity_distance_list_max != 0:
            weight = (W1 * (1 - (amenity_distance_list[i] / amenity_distance_list_max))) + (W2 * amenity_salience_list[i])
        else:
            weight = 0
        amenity_value += weight

    return amenity_value



def add_polygon_relation_edges(G,bbox_tuple):
    multipolygons_gdf = ox.features_from_bbox(bbox=bbox_tuple, tags={'type':"multipolygon"})

    #print(multipolygons_gdf)
    if 'highway' in multipolygons_gdf.columns:
        pedestrian_multipolygons = multipolygons_gdf[multipolygons_gdf['highway'] == 'pedestrian'].dropna(axis=1, how="any")
        counter = 0
        for multipolygon in pedestrian_multipolygons.geometry:
            print(multipolygon.exterior.coords)
            
            poly_graph_nodes = []
            for point in multipolygon.exterior.coords:
                nearest_node = ox.nearest_nodes(G,X=point[0], Y=point[1],return_dist=False)
                if nearest_node not in poly_graph_nodes:
                    poly_graph_nodes.append(nearest_node)

            for u, v in itertools.combinations(poly_graph_nodes, 2):  # Fix the indexing issue
                if not G.has_edge(u, v):
                    G.add_edge(u, v, 0)
                    x1 = G.nodes[u]['x']
                    y1 = G.nodes[u]['y']
                    x2 = G.nodes[v]['x']
                    y2 = G.nodes[v]['y']
                    G.edges[u,v,0]['length'] = ox.distance.euclidean(x1, y1, x2, y2)
                    counter += 1
                        
        print(f"Added {counter} edges to the graph")
    else:
        print("no pedestrian highẃay polygons to add to the graph")
    return G