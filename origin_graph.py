from typing import List, Optional, Set, Dict
from dataclasses import dataclass
import logging
import osmnx as ox
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
class Route:
    def __init__(self, start: str, end: str, distance: float):
        self.start = start
        self.end = end
        self.distance = distance
        
    def __str__(self) -> str:
        return f"Route({self.start} -> {self.end}, distance: {self.distance})"
    

class Graph:
    def __init__(self, start_point:tuple, city_name: str, origin_name: Optional[str] = None,
                  edge_attr_diff: Optional[str] = None,network_type: Optional[str] = 'walk',
                  load_graphml: Optional[str] = None, remove_parallel_edges: Optional[bool] = False,
                  ):
        self.city_name = city_name
        self.routes: Set[Route] = set()
        self.start_point = start_point
        self.origin_name = origin_name
        self.edge_attr_diff = edge_attr_diff
        self.network_type = network_type
        
        if load_graphml:
            self.graph = ox.load_graphml(load_graphml)
            print('Loaded graph from file')
        else:
            self.graph = self.create_graph()
            print('Created graph')

        self.start_node = self.find_start_node()

    def create_graph(self):
        """
        Create a graph from the center point and distance.

        Output:
        - Returns a graph object created from the center point and distance.
        """
        if self.network_type not in ['walk', 'bike', 'drive', 'drive_service', 'all', 'all_private']:
            G = ox.graph_from_point(self.center_point, dist=self.distance_from_point, custom_filter=self.network_type, simplify=False, truncate_by_edge=True)
            print('Using custom filter to create graph')        
        else:
            G = ox.graph_from_point(self.center_point, dist=self.distance_from_point, network_type=self.network_type, simplify=False, truncate_by_edge=True)

        if self.edge_attributes == 'osmid':
            G = ox.simplification.simplify_graph(G,edge_attrs_differ=['osmid'])
            print('Simplifying graph retaining unique osmid')
        else:
            G = ox.simplify_graph(G)
        G = ox.bearing.add_edge_bearings(G)
        G = ox.distance.add_edge_lengths(G)
        G.graph['city_name'] = self.city_name
        G.graph['origin_name'] = self.origin_name
        G.graph['order_category'] = self.order_category
        return G

    def find_start_node(self):
        """
        Find the nearest with node outgoing edges that is closest to the center point 

        Output:
        - Returns the start node of the graph.
        """
        nearest_node = ox.distance.nearest_nodes(self.graph, self.center_point[1], self.center_point[0])
        temp_graph = self.graph.copy()
        while self.graph.out_degree(nearest_node) == 0:
            temp_graph.remove_node(nearest_node)
            nearest_node = ox.distance.nearest_nodes(temp_graph, self.center_point[1], self.center_point[0])
        return nearest_node

    def remove_parallel_edges(self):
        """
        Remove parallel edges from the graph, keeping only the shortest one.

        Output:
        - Modifies the graph by removing parallel edges.
        """
        edges_to_remove = []
        for u, v, k in self.graph.edges(keys=True):
            parallel_edges = [(u, v, key) for key in self.graph[u][v]]
            if len(parallel_edges) > 1:
                shortest_edge = min(parallel_edges, key=lambda edge: self.graph.edges[edge]['length'])
                for edge in parallel_edges:
                    if edge != shortest_edge:
                        edges_to_remove.append(edge)
        self.remove_parallel_edges = edges_to_remove
        self.graph.remove_edges_from(edges_to_remove)                                                   
    
    def remove_infinite_edges(self):
        G = self.graph
        edges_to_remove = []
        for (u, v, data) in G.edges(data=True):
            if float(G.edges[(u, v, 0)]['weight_decision_complexity']) == float('inf'):
                edges_to_remove.append((u, v, 0))
        G.remove_edges_from(edges_to_remove)
        print(f'Removed {len(edges_to_remove)} edges with infinite decision complexity')
        self.graph = G
        self.removed_inf_edges = edges_to_remove