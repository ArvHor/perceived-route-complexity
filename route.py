import logging
import networkx as nx
import hashlib
import osmnx as ox
import weighting_algorithms as wa
import map_plotting as mp
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
            self.nodes = self.retrieve_simplest_path(self.graph, self.origin_node, self.destination_node)
            self.edges = list(nx.utils.pairwise(self.nodes))
            complexity_dict = self.get_route_complexity(self.graph, self.edges)
            self.complexity = complexity_dict['sum']
            self.complexity_list = complexity_dict['complexity_list']
            self.turn_types = complexity_dict['turn_types']
            self.route_geometry = ox.routing.route_to_gdf(self.graph, self.nodes, weight=weighstring)["geometry"].unary_union
        else:
            self.nodes = ox.routing.shortest_path(self.graph, self.origin_node, self.destination_node, weight=weighstring)
            self.edges = list(nx.utils.pairwise(self.nodes))
            complexity_dict = self.get_route_complexity(self.graph, self.edges)
            self.complexity = complexity_dict['sum']
            self.complexity_list = complexity_dict['complexity_list']
            self.turn_types = complexity_dict['turn_types']
            self.route_geometry = ox.routing.route_to_gdf(self.graph, self.nodes, weight=weighstring)["geometry"].unary_union

            
        self.length = self.get_edges_sum('length')
        self.n_nodes = len(self.nodes)
        self.sum_deviation_from_prototypical = self.get_edges_sum('deviation_from_prototypical')
        self.sum_node_degree = self.get_edges_sum('node_degree')
        self.sum_instruction_equivalent = self.get_edges_sum('instruction_equivalent')
        self.identifier = self.generate_identifier()
        self.map_bbox = None

    def get_route_subgraph(self, graph):
        """Creates a subgraph from the original graph containing only the route nodes and edges."""
        subgraph = graph.subgraph(self.nodes).copy()
        return subgraph

    def generate_identifier(self):
        """Generates a unique identifier for the route (e.g., using SHA-256)."""
        route_string = str(self.nodes)
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
    
    def create_route_map(self,filepath,map_tiles="CartoDB.VoyagerNoLabels",flip=False):
        """Adds a map of the route to the route object."""
        #def plot_route_gdf(G, route_gdf,start_node,end_node,info_text="null",imgpath="route_on_map.png"
        # ,file_path="route_on_map.png",map_tiles="CartoDB.VoyagerNoLabels",return_bbox=False, flip=False):

        route_gdf = ox.routing.route_to_gdf(self.graph,self.nodes,weight=self.weightstring)
        bbox = mp.plot_route_gdf(self.graph,route_gdf,self.origin_node,self.destination_node,info_text=self.identifier,file_path=filepath,flip=flip,map_tiles=map_tiles,return_bbox=True)
        self.map_bbox = bbox
    
    def get_map_bbox(self):
        if self.map_bbox is not None:
            return self.map_bbox
        else:
            bbox = mp.get_routegdf_bbox(self.graph,self.nodes,self.origin_node,self.destination_node)
            self.map_bbox = bbox
            return bbox


    @staticmethod
    def retrieve_simplest_path(G,origin,destination):
            t = destination
            route = []
            w_list = []
            route.append(t)
            while t != origin:
                in_edges = list(G.in_edges(t,data=True))
                min_decision_complexity = float('inf')
                min_edge = None
                for u, v, data in in_edges:
                    decision_complexity = data.get('decision_complexity', float('inf'))
                    if decision_complexity < min_decision_complexity:
                        min_decision_complexity = decision_complexity
                        w_list.append(decision_complexity)
                        min_edge = (u, v)

                t = min_edge[0]
                route = [t] + route

            return route
    
    @staticmethod
    def get_route_complexity(G, route_edges):
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
            total_complexity += decision_complexity

            previous_edge_complexity = current_edge_complexity

            turn_types.append(turn)


        result = {
            'sum': total_complexity,
            'complexity_list': complexities,
            'turn_types': turn_types
        }

        return result