import logging
import networkx as nx
import hashlib
import osmnx as ox
import weighting_algorithms as wa
info_handler = logging.FileHandler('info.log')
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))


class route:
    def __init__(self, graph, origin, destination, weighstring):

        self.graph = graph
        self.origin_node = origin
        self.destination_node = destination
        self.weightstring = weighstring
        if weighstring == "decision_complexity":
            self.nodes = self.find_simplest_path(self.graph, self.origin_node, self.destination_node)
            self.edges = list(nx.utils.pairwise(self.nodes))
            self.complexity, self.complexity_list, self.turn_types = self.get_route_complexity_optimized(self.graph, self.edges)
            self.route_geometry = ox.routing.route_to_gdf(self.graph, self.nodes, weight=weighstring)["geometry"].unary_union
        else:
            self.nodes = ox.routing.shortest_path(self.graph, self.origin_node, self.destination_node, weight=weighstring)
            self.edges = list(nx.utils.pairwise(self.nodes))
            self.complexity, self.complexity_list, self.turn_types = self.calculate_route_complexity_post_hoc(self.graph, self.edges)
            self.route_geometry = ox.routing.route_to_gdf(self.graph, self.nodes, weight=weighstring)["geometry"].unary_union
        self.length = self.get_edges_sum('length')
        self.n_nodes = len(self.nodes)
        self.sum_deviation_from_prototypical = self.get_edges_sum('deviation_from_prototypical')
        self.sum_node_degree = self.get_edges_sum('node_degree')
        self.sum_instruction_equivalent = self.get_edges_sum('instruction_equivalent')
        #print(f"route complexity: {self.complexity}")
        self.identifier = self.generate_identifier()

    def create_subgraph(self, graph):
        """Creates a subgraph from the original graph containing only the route nodes and edges."""
        subgraph = graph.subgraph(self.nodes).copy()
        return subgraph

    def generate_identifier(self):
        """Generates a unique identifier for the route (e.g., using SHA-256)."""
        # Create a string representation of the route (e.g., sorted node list)
        route_string = str(self.nodes)
        # Hash the string using SHA-256
        return hashlib.sha256(route_string.encode()).hexdigest()


    def get_edges_avg(self,weightstring):
        edges_avg = 0
        for edge in self.edges:
            edges_avg += float(self.graph.edges[edge].get(weightstring, 0))
        return edges_avg / len(self.graph.edges)

    def get_edges_sum(self,weightstring):
        edges_sum = 0
        for edge in self.edges:
            u,v = edge
            edge_length = float(self.graph.edges[u,v,0].get(weightstring, 0))
            edges_sum += edge_length
        return edges_sum

    def add_route_complexity(self):
        max_complexity = float(self.graph.graph['max_decision_complexity'])
        edges = self.edges
        complexity = 0
        turn_types = []
        for i in range(len(edges) - 2):
            u = edges[i][1]
            v = edges[i + 1][0]
            w = edges[i + 1][1]
            
            decision_complexity,turn = wa.calculate_decisionpoint_complexity(self.graph, (u,v), (v,w))
            complexity = complexity + (decision_complexity / max_complexity)
            turn_types.append(turn)


        return complexity,turn_types

    def to_dict(self):
        """Converts the route to a dictionary."""
        return {
            'id': self.identifier,
            'nodes': self.nodes,
            'edges': self.edges,
            'weightstring': self.weightstring,
            'length': self.length
            
        }
    @staticmethod
    def get_route_complexity(G, edges):
            turn_types = []
            previous_edge_complexity = 0
            total_complexity = 0
            complexities = []
            for i in range(len(edges) - 1):
                u = edges[i][0]
                v = edges[i][1]
                w = edges[i + 1][1]

                # 1. Complexity score of the turn
                # calculate the complexity of the turn between the edges
                decision_complexity,turn = wa.calculate_decisionpoint_complexity(G, (u,v), (v,w))
                
                # 2.Calculate the complexity of this edge 
                # Add the complexity of the turn to the accumulated complexities
                current_edge_complexity = previous_edge_complexity + decision_complexity

                # 3. Set the accumulated complexities to the current edge complexity
                previous_edge_complexity = current_edge_complexity

                complexities.append(current_edge_complexity)



                turn_types.append(turn)
            total_complexity = sum(complexities)


            return total_complexity,turn_types
    @staticmethod
    def get_dijkstra_path(MultiDigraph, origin, destination, weightstring):
        # First I find the path using Networkx's dijkstra algorithm
        path = nx.dijkstra_path(MultiDigraph, origin, destination, weight=weightstring)
        path_turns = []
        
        # Because the dijkstra algorithm only returns the nodes and the graph may contain multiple edges between the same nodes
        # I find the edge with the minimum weight for each pair of nodes in the path
        
        edges = []
        weight_list = []
        for i in range(len(path) - 1):
            # Get the key of the edge with the minimum weight
            min_weight = float('inf')
            min_key = None

            for key in MultiDigraph[path[i]][path[i + 1]]:
                weight = MultiDigraph.edges[path[i], path[i + 1], key].get(weightstring, 0)
                if weight < min_weight:
                    min_weight = weight
                    min_key = key
                    
            edges.append((path[i], path[i + 1], min_key)) 
            weight = MultiDigraph.edges[path[i], path[i + 1], min_key].get(weightstring, 0)
            turn = MultiDigraph.edges[path[i], path[i + 1], min_key].get("turn_complexity", "unknown")
            weight_list.append(weight)
            path_turns.append(turn)
            
            
        path = {
            'nodes': path,
            'edges': edges,
            "path_weights_sum": sum(weight_list),
            "path_turns": path_turns
        }
        #print(path['edges'])
        return path
    @staticmethod
    def find_simplest_path(G,origin,destination):
        
            t = destination
            s = origin
            p = []
            e = []
            w_list = []
            p.append(t)
            while t != s:
                in_edges = list(G.in_edges(t,data=True))
                if len(in_edges) == 0:
                    print(f"Error: {t} has no in edges city: {G.graph['city_name']} origin: {s} destination: {destination}")
                    print(f"nr of edges: {len(G.edges)}")
                    logging.error(f"Error: {t} has no in edges city: {G.graph['city_name']} origin: {s} destination: {destination}")
                    logging.error(f"nr of edges: {len(G.edges)}")
                    break
                min_decision_complexity = float('inf')
                min_edge = None
                for u, v, data in in_edges:
                    if v != t:
                        print(f"Error: {v} != {t}")
                    decision_complexity = data.get('decision_complexity', float('inf'))
                    if decision_complexity < min_decision_complexity:
                        min_decision_complexity = decision_complexity
                        w_list.append(decision_complexity)
                        min_edge = (u, v)

                t = min_edge[0]
                p = [t] + p
                e = [min_edge] + e

            return p
    @staticmethod
    def get_route_complexity_optimized(G, route_edges):
        """
        Calculates the total decision complexity of a route based on pre-calculated edge weights.

        Args:
            G: The graph with edge weights calculated by simplest_path_weight_algorithm_optimized.
            route_edges: A list of edges representing the route.

        Returns:
            total_complexity: The total decision complexity of the route.
            complexities: A list of cumulative complexities at each edge in the route.
            turn_types: A list of turn types at each turn in the route.
        """
        turn_types = []
        total_complexity = 0
        complexities = []

        if not route_edges:
            return 0, [], []

        # Retrieve the decision_complexity of the last edge. This represents the
        # total accumulated complexity up to that point, as calculated by the
        # optimized weighting algorithm.
        last_edge = route_edges[-1]
        total_complexity = G.edges[last_edge[0], last_edge[1], 0]['decision_complexity']


        # Iterate to get the turn types and intermediate complexities if needed
        for i in range(len(route_edges) - 1):
            u, v = route_edges[i]
            v, w = route_edges[i+1] # Corrected line: Use v, w for the second edge
            _, turn = wa.calculate_decisionpoint_complexity(G, (u, v), (v, w))
            turn_types.append(turn)

            # If you need the complexity at each edge along the route:
            # This is useful for debugging or analysis, but not needed for the final sum.
            complexities.append(G.edges[u, v, 0]['decision_complexity'])
        
        # Add the complexity of the last edge to the complexities list (optional)
        complexities.append(total_complexity)
        return total_complexity, complexities, turn_types
    @staticmethod
    def calculate_route_complexity_post_hoc(G, route_edges):
        """
        Calculates the total decision complexity of a route post-hoc,
        without relying on pre-calculated edge weights.

        Args:
            G: The graph (NetworkX graph).
            route_edges: A list of edges representing the route.

        Returns:
            total_complexity: The total decision complexity of the route.
            complexities: A list of cumulative complexities at each edge in the route.
            turn_types: A list of turn types at each turn in the route.
        """
        turn_types = []
        total_complexity = 0
        complexities = []
        previous_edge_complexity = 0  # Initialize for the first edge

        for i in range(len(route_edges) - 1):
            u, v = route_edges[i]
            v, w = route_edges[i + 1]  # Correct indexing for consecutive edges

            # 1. Calculate the complexity of the turn between the edges
            decision_complexity, turn = wa.calculate_decisionpoint_complexity(G, (u, v), (v, w))

            # 2. Calculate the complexity of this edge segment
            current_edge_complexity = previous_edge_complexity + decision_complexity

            # 3. Accumulate complexities
            complexities.append(current_edge_complexity)
            total_complexity += decision_complexity  # Update the running total

            # 4. Update previous_edge_complexity for the next iteration
            previous_edge_complexity = current_edge_complexity

            turn_types.append(turn)

        return total_complexity, complexities, turn_types
    @staticmethod
    def static_get_edges_sum(G,edges,weightstring):
        edges_sum = 0
        for edge in edges:
            u,v = edge
            edge_length = float(G.edges[u,v,0].get(weightstring, 0))
            edges_sum += edge_length
        return edges_sum