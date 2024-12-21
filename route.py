import networkx as nx
import hashlib
from utils import visualize_route  # Assume you have a function for visualization

class Route:
    def __init__(self, nodes, edges, route_info, graph):
        self.nodes = nodes
        self.edges = edges
        self.route_info = route_info  # e.g., {'length': 10, 'algorithm': 'dijkstra'}
        self.subgraph = self.create_subgraph(graph)
        self.identifier = self.generate_identifier()

    def create_subgraph(self, graph):
        """Creates a subgraph from the original graph containing only the route nodes and edges."""
        subgraph = graph.subgraph(self.nodes).copy()
        return subgraph

    def generate_identifier(self):
        """Generates a unique identifier for the route (e.g., using SHA-256)."""
        # Create a string representation of the route (e.g., sorted node list)
        route_string = str(sorted(self.nodes))
        # Hash the string using SHA-256
        return hashlib.sha256(route_string.encode()).hexdigest()

    def visualize(self, **kwargs):
        """Visualizes the route on a map."""
        visualize_route(self.subgraph, **kwargs)  # Pass any visualization options

    def to_dict(self):
        """Converts the route to a dictionary."""
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'route_info': self.route_info,
            'identifier': self.identifier,
            'subgraph': list(self.subgraph.edges(data=True)) # Convert subgraph to a serializable format (e.g. list of edges)
        }