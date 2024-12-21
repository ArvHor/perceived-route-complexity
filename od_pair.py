import networkx as nx
import pandas as pd
from route import Route

class OriginDestinationPair:
    def __init__(self, origin, destination, distance):
        self.origin = origin
        self.destination = destination
        self.distance = distance
        self.routes = set()
        self.subgraph = None # Initialize as None, will be created when routes are added

    def find_routes(self, algorithm, graph, **kwargs):
        """Finds routes using different algorithms and adds them to the set of routes."""
        if algorithm == 'dijkstra':
            try:
                path = nx.dijkstra_path(graph, self.origin, self.destination, **kwargs)
                length = nx.dijkstra_path_length(graph, self.origin, self.destination, **kwargs)
                # Extract edges from the path
                edges = [(path[i], path[i+1], graph[path[i]][path[i+1]]) for i in range(len(path)-1)]
                
                route_info = {'length': length, 'algorithm': algorithm}
                new_route = Route(path, edges, route_info, graph)
                self.add_route(new_route)
            except nx.NetworkXNoPath:
                print(f"No path found between {self.origin} and {self.destination} using Dijkstra's algorithm.")
        # Add more algorithms (e.g., A*, BFS, DFS) as needed
        elif algorithm == 'astar':
            try:
                path = nx.astar_path(graph, self.origin, self.destination, **kwargs)
                length = nx.astar_path_length(graph, self.origin, self.destination, **kwargs)
                # Extract edges from the path
                edges = [(path[i], path[i+1], graph[path[i]][path[i+1]]) for i in range(len(path)-1)]
                
                route_info = {'length': length, 'algorithm': algorithm}
                new_route = Route(path, edges, route_info, graph)
                self.add_route(new_route)
            except nx.NetworkXNoPath:
                print(f"No path found between {self.origin} and {self.destination} using A* algorithm.")
        else:
            raise ValueError("Unsupported algorithm specified.")
        
        #Update subgraph after adding new routes
        self.update_subgraph()

    def add_route(self, route):
        """Adds a route to the set of routes if it's unique."""
        if route.identifier not in [r.identifier for r in self.routes]:
            self.routes.add(route)

    def update_subgraph(self):
        """Updates the subgraph to include all nodes and edges from all routes."""
        if not self.routes:
            self.subgraph = None  # No routes, subgraph should be empty or None
            return

        # Collect all nodes and edges from all routes
        all_nodes = set()
        all_edges = set()
        for route in self.routes:
            all_nodes.update(route.nodes)
            all_edges.update(route.edges)

        # Create a new subgraph containing all collected nodes and edges
        # Assuming you want to preserve the original graph structure:
        # 1. Create an empty graph
        self.subgraph = nx.Graph()
        # 2. Add all nodes
        self.subgraph.add_nodes_from(all_nodes)
        # 3. Add all edges with attributes from the original graph
        for u, v in all_edges:
            # Retrieve attributes from the original graph if available
            edge_data = self.original_graph.get_edge_data(u, v, default={})
            self.subgraph.add_edge(u, v, **edge_data)

    def to_dataframe(self):
        """Converts the OD pair data and its routes to a Pandas DataFrame."""
        data = []
        for route in self.routes:
            data.append({
                'od_pair_id': f"{self.origin}-{self.destination}",  # You might want a more robust ID
                'origin': self.origin,
                'destination': self.destination,
                'distance': self.distance,
                'route_id': route.identifier,
                'route_length': route.route_info.get('length'),
                'algorithm': route.route_info.get('algorithm'),
                'nodes': route.nodes,
                'edges': route.edges,
                'subgraph_edges': list(route.subgraph.edges) # Convert subgraph to a serializable format
            })
        return pd.DataFrame(data)