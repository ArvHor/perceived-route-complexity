import networkx as nx
import hashlib
import osmnx as ox
import weighting_algorithms as wa
import map_plotting as mp
import network_analysis
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


        self.turn_count = sum('turn' in s.lower() for s in self.turn_types)
        self.length = self.get_edges_sum('length')
        self.turn_frequency = self.turn_count/self.length
        self.n_nodes = len(self.nodes)
        self.sum_deviation_from_prototypical = self.get_edges_sum('deviation_from_prototypical')
        self.sum_node_degree = self.get_edges_sum('node_degree')
        self.sum_instruction_equivalent = self.get_edges_sum('instruction_equivalent')
        
        self.sum_od_betweenness = self.get_nodes_sum("od_betweenness")
        self.sum_betweenness = self.get_nodes_sum("betweenness_centrality")
        
        self.avg_od_betweenness = self.sum_od_betweenness / self.n_nodes
        self.avg_betweenness = self.sum_betweenness / self.n_nodes
        
        self.identifier = self.generate_identifier()
        self.map_bbox = self.get_map_bbox()
        self.map_road_length, self.map_intersection_count, self.map_street_count = self.get_map_clutter()


    @classmethod
    def from_nodes(cls, graph, nodes, weightstring):
        # Create instance without calling __init__
        instance = cls.__new__(cls)
        
        # Set the attributes that would normally be set in __init__
        instance.graph = graph
        instance.nodes = nodes
        instance.weightstring = weightstring
        
        # Set origin and destination from the nodes list
        instance.origin_node = nodes[0]
        instance.destination_node = nodes[-1]

        # Calculate edges and other attributes
        instance.edges = list(nx.utils.pairwise(instance.nodes))
        complexity_dict = instance.get_route_complexity(instance.graph, instance.edges)
        instance.complexity = complexity_dict['sum']
        instance.complexity_list = complexity_dict['complexity_list']
        instance.turn_types = complexity_dict['turn_types']
        instance.route_geometry = ox.routing.route_to_gdf(instance.graph, instance.nodes, weight=weightstring)["geometry"].unary_union
        
        # Set all the other attributes that would be set in __init__
        instance.turn_count = sum('turn' in s.lower() for s in instance.turn_types)
        instance.length = instance.get_edges_sum('length')
        instance.turn_frequency = instance.turn_count/instance.length
        instance.n_nodes = len(instance.nodes)
        instance.sum_deviation_from_prototypical = instance.get_edges_sum('deviation_from_prototypical')
        instance.sum_node_degree = instance.get_edges_sum('node_degree')
        instance.sum_instruction_equivalent = instance.get_edges_sum('instruction_equivalent')

        instance.sum_betweenness = instance.get_nodes_sum("betweenness_centrality")
        instance.sum_od_betweenness = instance.get_nodes_sum("od_betweenness")
        instance.avg_od_betweenness = instance.sum_od_betweenness / instance.n_nodes
        instance.avg_betweenness = instance.sum_betweenness / instance.n_nodes

        instance.identifier = instance.generate_identifier()
        instance.map_bbox = instance.get_map_bbox()
        instance.map_road_length, instance.map_intersection_count, instance.map_street_count = instance.get_map_clutter()
        
        return instance
        

        
    def get_route_subgraph(self, graph):
        """Creates a subgraph from the original graph containing only the route nodes and edges."""
        subgraph = graph.subgraph(self.nodes).copy()
        return subgraph
    
    def get_map_clutter(self):
        undirected_G = ox.convert.to_undirected(self.graph)
        undirected_G = ox.truncate.truncate_graph_bbox(undirected_G,self.map_bbox)
        length = sum(data[2]["length"] for data in undirected_G.edges(data=True))
        intersection_count =len(undirected_G.nodes())
        street_count = len(undirected_G.edges())
                           
        return length,intersection_count,street_count

    def generate_identifier(self):
        """Generates a unique identifier for the route (e.g., using SHA-256)."""
        route_string = str(self.nodes)
        return hashlib.sha256(route_string.encode()).hexdigest()

    def get_nodes_avg(self, weightsting):
        total_betweenness = sum(self.graph.nodes[node][weightsting] for node in self.nodes)
        avg_betweenness = total_betweenness / len(self.nodes)

        return avg_betweenness

    def get_nodes_sum(self, weightstring):
        total_sum = sum(self.graph.nodes[node].get(weightstring, 0) for node in self.nodes)
        return total_sum

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
        route_gdf = ox.routing.route_to_gdf(self.graph,self.nodes,weight=self.weightstring)
        bbox = mp.plot_route_gdf(self.graph,route_gdf,self.origin_node,self.destination_node,info_text=self.identifier,file_path=filepath,flip=flip,map_tiles=map_tiles,return_bbox=True)
        self.map_bbox = bbox
    
    def get_map_bbox(self):
        bbox = mp.get_routegdf_bbox(self.graph,self.nodes,self.origin_node,self.destination_node)
        self.map_bbox = bbox
        return bbox

    def to_dict(self):
        """
        Returns a dictionary containing all attributes of the route.
        """
        attributes = {}
        
        # Get all attributes that don't start with underscore
        for attr_name in dir(self):
            # Skip methods and private attributes
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                attributes[attr_name] = getattr(self, attr_name)
        
        # Handle special cases for geometry objects which might not be serializable
        if 'route_geometry' in attributes:
            # Convert to WKT (Well-Known Text) format if it's a geometry object
            try:
                from shapely.geometry import mapping
                attributes['route_geometry'] = mapping(attributes['route_geometry'])
            except (ImportError, AttributeError):
                # If shapely isn't available or it's not a geometry object
                attributes['route_geometry'] = str(attributes['route_geometry'])
        
        # Handle graph object which is likely not serializable
        if 'graph' in attributes:
            attributes['graph'] = "NetworkX graph object (not serialized)"
        
        return attributes

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
    