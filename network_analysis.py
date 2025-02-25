import networkx as nx
import numpy as np

def add_betweenness_centrality(graph, origin, destination, weightstring):
    for node in graph.nodes():
        graph.nodes[node]['node_betweenness'] = 0.0
    try:
        all_shortest_paths = list(nx.all_shortest_paths(graph, origin, destination,weight=weightstring))
    except nx.NetworkXNoPath:
        # If no path exists, return the graph with zero betweenness for all nodes
        return graph
    
    # Calculate the total number of shortest paths
    num_shortest_paths = len(all_shortest_paths)
    
    # For each node, count how many shortest paths it appears in
    path_counts = {}
    for path in all_shortest_paths:
        # Skip the first and last nodes (origin and destination)
        for node in path[1:-1]:
            if node in path_counts:
                path_counts[node] += 1
            else:
                path_counts[node] = 1
    
    # Calculate point-to-point betweenness for each node
    for node, count in path_counts.items():
        graph.nodes[node]['node_betweenness'] = count / num_shortest_paths
    
    return graph
