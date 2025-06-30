import math
from typing import Any
import shapely
import logging
import networkx as nx
import numpy as np
import osmnx as ox

from . import geo_util
from . import path_search
logging.basicConfig(
    filename='app.log',          # Log file name
    filemode='a',                # 'a' for append, 'w' for overwrite
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO           # Set the minimum logging level
)
def get_nodes_avg(G,route_nodes,weightstring):
    nodes_sum = 0
    for node in route_nodes:
        nodes_sum += G.nodes[node][weightstring]
    return nodes_sum/len(route_nodes)

def get_nodes_sum(G,route_nodes,weightstring):
    nodes_sum = 0
    for node in route_nodes:
        nodes_sum += G.nodes[node][weightstring]
    return nodes_sum

def get_edges_avg(G,route_edges,weightstring):
    edges_sum = 0
    for edge in route_edges:
        edges_sum += G.edges[edge[0],edge[1],0][weightstring]
    return edges_sum/len(route_edges)

def get_edges_sum(G,route_edges,weightstring):
    edges_sum = 0
    for edge in route_edges:
        edges_sum += G.edges[edge[0],edge[1],0][weightstring]
    return edges_sum

def extract_route_bearings(G, route_nodes, weight=None,get_undirected=True):
    weights = []
    bearings = []
    # Iterate over consecutive pairs of route nodes to get the edges in the route
    for i in range(len(route_nodes) - 1):
        u = route_nodes[i]
        v = route_nodes[i + 1]
        if u != v and G.has_edge(u, v):
            w = G.edges[u, v, 0][weight] if weight is not None else 1.0
            length = G.edges[u, v, 0]["length"]
            bearings.append(length)
            weights.append(w)

    bearings_array = np.array(bearings)
    weights_array = np.array(weights)
    keep_idx = ~np.isnan(bearings_array)
    bearings_array = bearings_array[keep_idx]
    weights_array = weights_array[keep_idx]

    if get_undirected:
        bearings_array = np.concatenate([bearings_array, (bearings_array - 180) % 360])
        weights_array = np.concatenate([weights_array, weights_array])
    return bearings_array, weights_array


def get_route_bearing_dist(G,route_nodes,weight=None,num_bins=36, get_undirected=True) -> tuple[np.ndarray[tuple[int, ...], np.dtype[Any]], np.ndarray[tuple[int, ...], np.dtype[Any]]]:
    num_split_bins = num_bins * 2
    split_bin_edges = np.arange(num_split_bins + 1) * 360 / num_split_bins

    bearings, weights = extract_route_bearings(G, route_nodes, weight=weight,get_undirected=get_undirected)

    split_bin_counts, split_bin_edges = np.histogram(
        bearings,
        bins=split_bin_edges,
        weights=weights,
    )

    # Move last bin to front, so eg 0.01 degrees and 359.99 degrees will be
    # binned together. Then combine counts from pairs of split bins.
    split_bin_counts = np.roll(split_bin_counts, 1)
    bin_counts = split_bin_counts[::2] + split_bin_counts[1::2]
    bin_centers = split_bin_edges[range(0, num_split_bins - 1, 2)]
    return bin_counts, bin_centers

def get_route_complexity(G,route_edges):
    turn_types = []
    total_complexity = 0
    complexities = []
    previous_edge_complexity = 0  # Initialize for the first edge

    for i in range(len(route_edges) - 1):
        u, v = route_edges[i]
        v, w = route_edges[i + 1]  # Correct indexing for consecutive edges

        # 1. Calculate the complexity of the turn between the edges
        decision_complexity, turn = path_search.calculate_decisionpoint_complexity(G, (u, v), (v, w))

        # 2. Calculate the complexity of this edge segment
        current_edge_complexity = previous_edge_complexity + decision_complexity

        # 3. Accumulate complexities
        complexities.append(current_edge_complexity)
        total_complexity += decision_complexity

        previous_edge_complexity = current_edge_complexity

        turn_types.append(turn)

    result = {
        'sum': total_complexity,
        'complexity_list': complexities,
        'turn_types': turn_types
    }

    return result


def get_origin_destination_betweenness_centrality(graph, route_nodes, origin, destination, weightstring='length'):
    # Find all shortest paths from origin to destination
    all_shortest_paths = list(nx.all_shortest_paths(graph, origin, destination, weight=weightstring))

    # If no path exists, return 0
    if not all_shortest_paths:
        return 0.0

    # Calculate the total number of shortest paths
    num_shortest_paths = len(all_shortest_paths)

    # Calculate betweenness for each node in route_nodes
    od_betweenness_sum = 0.0

    for node in route_nodes:
        # Skip if the node is the origin or destination
        if node == origin or node == destination:
            continue

        # Count how many shortest paths contain this node
        node_path_count = sum(1 for path in all_shortest_paths if node in path[1:-1])

        # Calculate betweenness for this node
        if node_path_count > 0:
            node_betweenness = node_path_count / num_shortest_paths
            od_betweenness_sum += node_betweenness

    return od_betweenness_sum


def get_n_route_segments(route_linestring,thold=50):
    route_linestring_coords = route_linestring.coords
    n_before = len(route_linestring_coords)
    route_linestring_coords = geo_util.douglas_peucker(route_linestring_coords,thold=thold)
    n_after = len(route_linestring_coords)

    return n_after,n_before,route_linestring


def get_route_bearing_sum(G, route_nodes, absolute=False):
    sum_difference = 0
    for i in range(0, len(route_nodes) - 2):
        origin = route_nodes[i]
        intermediate = route_nodes[i + 1]
        destination = route_nodes[i + 2]
        
        bearing_difference = geo_util.get_bearing_difference(G,origin, intermediate, destination)
        if math.isnan(bearing_difference):
            raise ValueError("Bearing difference is NaN. Check the coordinates or the graph.")
        if absolute:
            sum_difference += abs(bearing_difference)
        else:
            sum_difference += bearing_difference
           
    return sum_difference

