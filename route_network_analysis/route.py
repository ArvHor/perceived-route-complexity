import networkx as nx
import hashlib
import osmnx as ox
from shapely.geometry import LineString
import shapely.ops

# Local modules
from . import map_plotting as mp
from . import route_analysis
from . import path_search
from . import map_analysis

class route:
    def __init__(self, graph, origin, destination, weighstring):

        # Set the attributes that constrain the route
        self.graph = graph
        self.origin_node = origin
        self.destination_node = destination
        self.weightstring = weighstring

        # Retrieve the nodes of a route given a particular weightstring
        if weighstring == "decision_complexity":
            self.nodes = path_search.retrieve_simplest_path(self.graph, self.origin_node, self.destination_node)
        else:
            self.nodes = ox.routing.shortest_path(self.graph, self.origin_node, self.destination_node, weight=weighstring)

        # Get the edges, geometry and length of the route
        self.edges = list(nx.utils.pairwise(self.nodes))



        geometry= ox.routing.route_to_gdf(self.graph, self.nodes, weight=weighstring)["geometry"].unary_union
        if geometry.geom_type == 'LineString':
            self.route_linestring = geometry
        elif geometry.geom_type == 'MultiLineString':
            # Convert to LineString (since you know len == 1)
            self.route_linestring = LineString(list(geometry.geoms[0].coords))
        else:
            # Handle any other unexpected geometry types
            raise ValueError(f"Expected LineString or MultiLineString, got {geometry.geom_type}")




        self.length = route_analysis.get_edges_sum(G=self.graph,route_edges=self.edges,weightstring='length')

        # Get the attributes derived from Duckham and Kulik's simplest path search algorithm
        complexity_dict = route_analysis.get_route_complexity(self.graph, self.edges)
        self.complexity = complexity_dict['sum']
        self.complexity_list = complexity_dict['complexity_list']
        self.turn_types = complexity_dict['turn_types']

        # Get the number of intersections, turns and frequency of turns
        self.turn_count = sum('turn' in s.lower() for s in self.turn_types)
        self.turn_frequency = self.turn_count/self.length
        self.n_nodes = len(self.nodes)

        # Get the number of segments and total turn degree
        self.n_segments,self.n_segments_before,self.route_linestring = route_analysis.get_n_route_segments(self.route_linestring,thold=50)
        self.total_turn_degree = route_analysis.get_route_bearing_sum(self.graph, self.route_linestring)
        self.avg_turn_degree = self.total_turn_degree / self.n_segments

        # Get attributes of the nodes in the route
        self.sum_deviation_from_prototypical = route_analysis.get_edges_sum(G=self.graph,route_edges=self.edges,weightstring='deviation_from_prototypical')
        self.avg_deviation_from_prototypical = self.sum_deviation_from_prototypical / self.n_nodes
        self.sum_node_degree = route_analysis.get_edges_sum(G=self.graph,route_edges=self.edges,weightstring='node_degree')
        self.avg_node_degree = self.sum_node_degree / self.n_nodes
        self.sum_instruction_equivalent = route_analysis.get_edges_sum(G=self.graph,route_edges=self.edges,weightstring='instruction_equivalent')
        self.sum_betweenness = route_analysis.get_nodes_sum(G=self.graph,route_nodes=self.nodes,weightstring="betweenness_centrality")
        self.avg_betweenness = self.sum_betweenness / self.n_nodes

        # Get the betweenness centrality based on all shortest paths between the origin and destination
        self.sum_od_betweenness = route_analysis.get_origin_destination_betweenness_centrality(self.graph, self.nodes, self.origin_node, self.destination_node)
        self.avg_od_betweenness = self.sum_od_betweenness / self.n_nodes

        # Get the bbox containing the route and get cartographic clutter metrics
        self.map_bbox = map_analysis.get_routegdf_bbox(self.graph,self.nodes,buffer_percentage=0.1)
        print(self.map_bbox)
        self.map_road_length, self.map_intersection_count, self.map_street_count = map_analysis.get_map_clutter(G=self.graph,map_bbox=self.map_bbox)

        # Generate a unique identifier
        self.identifier = self.generate_identifier()



    @classmethod
    def from_nodes(cls, graph, nodes, weightstring):
        """
        Construct a route object from a list of nodes in a graph given a weighstring.
        """
        instance = cls.__new__(cls)
        
        # Set the attributes that constrain the route
        instance.graph = graph
        instance.nodes = nodes
        instance.weightstring = weightstring
        instance.origin_node = nodes[0]
        instance.destination_node = nodes[-1]

        # Get the edges, geometry and length of the route
        geometry = ox.routing.route_to_gdf(instance.graph, instance.nodes, weight=weightstring)["geometry"].unary_union

        if geometry.geom_type == 'LineString':
            instance.route_linestring = geometry
        elif geometry.geom_type == 'MultiLineString':
            # Convert to LineString (since you know len == 1)
            instance.route_linestring = LineString(list(geometry.geoms[0].coords))
        else:
            # Handle any other unexpected geometry types
            raise ValueError(f"Expected LineString or MultiLineString, got {geometry.geom_type}")

        instance.length = route_analysis.get_edges_sum(G=instance.graph,route_edges=instance.edges,weightstring='length')
        instance.edges = list(nx.utils.pairwise(instance.nodes))

        # Get the attributes derived from Duckham and Kulik's simplest path search algorithm
        complexity_dict = route_analysis.get_route_complexity(instance.graph, instance.edges)
        instance.complexity = complexity_dict['sum']
        instance.complexity_list = complexity_dict['complexity_list']
        instance.turn_types = complexity_dict['turn_types']

        # Get the number of intersections, turns and frequency of turns
        instance.turn_count = sum('turn' in s.lower() for s in instance.turn_types)
        instance.turn_frequency = instance.turn_count/instance.length
        instance.n_nodes = len(instance.nodes)

        # Get the number of segments and total turn degree
        instance.n_segments,instance.n_segments_before,instance.route_linestring = route_analysis.get_n_route_segments(instance.graph, instance.route_linestring)
        instance.total_turn_degree = route_analysis.get_route_bearing_sum(instance.graph, instance.route_linestring)
        instance.avg_turn_degree = instance.total_turn_degree / instance.n_segments

        # Get attributes of the nodes in the route
        instance.sum_deviation_from_prototypical = route_analysis.get_edges_sum(G=instance.graph,route_edges=instance.edges,weightstring='deviation_from_prototypical')
        instance.sum_node_degree = route_analysis.get_edges_sum(G=instance.graph,route_edges=instance.edges,weightstring='node_degree')
        instance.sum_instruction_equivalent = route_analysis.get_edges_sum(G=instance.graph,route_edges=instance.edges,weightstring='instruction_equivalent')
        instance.sum_betweenness = route_analysis.get_nodes_sum(G=instance.graph,route_nodes=instance.nodes,weightstring="betweenness_centrality")
        instance.avg_betweenness = instance.sum_betweenness / instance.n_nodes

        # Get the betweenness centrality based on all shortest paths between the origin and destination
        instance.sum_od_betweenness = route_analysis.get_origin_destination_betweenness_centrality(instance.graph, instance.nodes, instance.origin_node, instance.destination_node)
        instance.avg_od_betweenness = instance.sum_od_betweenness / instance.n_nodes



        # Get the bbox containing the route and get cartographic clutter metrics
        instance.map_bbox = map_analysis.get_routegdf_bbox()
        instance.map_road_length, instance.map_intersection_count, instance.map_street_count = map_analysis.get_map_clutter(G=instance.graph,map_bbox=instance.map_bbox)

        # Generate a unique identifier for the route
        instance.identifier = instance.generate_identifier()

        return instance

    def generate_identifier(self):
        """Generates a unique identifier for the route (e.g., using SHA-256)."""
        route_string = str(self.nodes)
        return hashlib.sha256(route_string.encode()).hexdigest()


    def create_route_map(self,filepath,map_tiles="CartoDB.VoyagerNoLabels",flip=False):
        """Adds a map of the route to the route object."""
        route_gdf = ox.routing.route_to_gdf(self.graph,self.nodes,weight=self.weightstring)
        bbox = mp.plot_route_gdf(self.graph,route_gdf,self.origin_node,self.destination_node,info_text=self.identifier,file_path=filepath,flip=flip,map_tiles=map_tiles,return_bbox=True)
        self.map_bbox = bbox
    



