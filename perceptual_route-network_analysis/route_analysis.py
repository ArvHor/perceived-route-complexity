import networkx as nx
import hashlib
import osmnx as ox
from shapely.lib import get_num_points, get_num_coordinates
import logging
import geo_util as geo_util
import path_search
logging.basicConfig(
    filename='../app.log',          # Log file name
    filemode='a',                # 'a' for append, 'w' for overwrite
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO           # Set the minimum logging level
)




def get_route_avg(G,route_nodes,weighstring):
    nodes_sum = 0
    for node in route_nodes:
        nodes_sum += G.nodes[node][weighstring]
    return nodes_sum/len(route_nodes)

def get_route_sum(G,route_nodes,weighstring):
    nodes_sum = 0
    for node in route_nodes:
        nodes_sum += G.nodes[node][weighstring]
    return nodes_sum

def get_edges_avg(G,route_edges,weighstring):
    edges_sum = 0
    for edge in route_edges:
        edges_sum += G.edges[edge[0],edge[1]][weighstring]
    return edges_sum/len(route_edges)

def get_edges_sum(G,route_edges,weighstring):
    edges_sum = 0
    for edge in route_edges:
        edges_sum += G.edges[edge[0],edge[1]][weighstring]
    return edges_sum

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


def get_origin_destination_betweenness_centrality(graph, route_nodes, origin, destination, weightstring):
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

    n_before = get_num_coordinates(route_linestring)
    route_linestring= geo_util.douglas_peucker(route_linestring,thold=thold)
    n_after = get_num_coordinates(route_linestring)

    return n_after


def get_route_bearing_sum(G, route_linestring):
    sum_difference = 0
    for i in range(len(route_linestring)):
        origin = route_linestring.coords[i]
        intermediate = route_linestring.coords[i+1]
        destination = route_linestring.coords[i+2]
        bearing_difference = geo_util.get_bearing_difference(G,origin,destination,intermediate)
        sum_difference += bearing_difference

    return sum_difference

