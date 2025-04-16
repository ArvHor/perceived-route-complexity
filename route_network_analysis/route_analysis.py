import shapely
import logging
import networkx as nx

# Local modules
from .performance_tracker import PerformanceTracker, track_performance
from . import geo_util
from . import path_search
logging.basicConfig(
    filename='app.log',          # Log file name
    filemode='a',                # 'a' for append, 'w' for overwrite
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO           # Set the minimum logging level
)

def get_city_name(graph):
    return graph.graph['city_name']

def get_start_node(graph):
    return graph.graph['start_node']

def get_avg_degree(graph):
    degrees = [d for _, d in graph.out_degree()]
    return sum(degrees) / len(degrees) if degrees else 0

def get_node_count(graph):
    return len(graph.nodes)

def get_edge_count(graph):
    return len(graph.edges)

def get_density(graph):
    return nx.density(graph)

def get_len(array_type):
    return len(array_type)

route_metrics = {
    'city_name': get_city_name,
    'start_node': get_start_node,
    'nodes': get_node_count,
    'edges': get_edge_count,
    'avg_degree': get_avg_degree,
    'n_count': get_len,
}
route_tracker = PerformanceTracker(output_file='route_analysis_performance.json')
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

@track_performance(route_tracker, metrics_funcs=route_metrics)
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


def get_route_bearing_sum(G, route_linestring):
    sum_difference = 0
    for i in range(0,len(route_linestring.coords)-2):
        origin = route_linestring.coords[i]
        intermediate = route_linestring.coords[i+1]
        destination = route_linestring.coords[i+2]
        bearing_difference = geo_util.get_bearing_difference(G,origin,destination,intermediate)
        sum_difference += bearing_difference

    return sum_difference

