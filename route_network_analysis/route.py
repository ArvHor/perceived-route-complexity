import os
import networkx as nx
import hashlib
import osmnx as ox
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
from geopandas import GeoSeries

# Local modules
from . import map_plotting as mp
from . import route_analysis
from . import path_search
from . import map_analysis
from . import geo_util


class route:
    def __init__(self, graph, origin, destination, weightstring):

        # Set the attributes that constrain the route
        graph = graph
        origin_node = origin
        destination_node = destination
        self.weightstring = weightstring
        # Retrieve the nodes of a route given a particular weightstring
        if weightstring == "decision_complexity":
            self.nodes = path_search.retrieve_simplest_path(
                graph, origin_node, destination_node
            )
        else:
            self.nodes = ox.routing.shortest_path(
                graph, origin_node, destination_node, weight=weightstring
            )
        self._initialize_basics(graph, origin_node, destination_node)

        # Initialize map-related attributes (set by from_nodes method when needed)
        self.map_bbox = None
        self.map_road_length = None
        self.map_intersection_count = None
        self.map_street_count = None

        # Generate a unique identifier
        self.identifier = self.generate_identifier(city_name=graph.graph["city_name"],weightstring=weightstring,origin_node=origin_node, destination_node=destination_node)

    @classmethod
    def from_nodes(cls, graph, nodes, weightstring):
        """
        Construct a route object from a list of nodes in a graph given a weightstring.
        """
        instance = cls.__new__(cls)

        # Set the attributes that constrain the route
        graph = graph
        instance.nodes = nodes
        weightstring = weightstring
        origin_node = nodes[0]
        destination_node = nodes[-1]
        city_name = graph.graph["city_name"]
        instance.identifier = instance.generate_identifier()
        instance._initialize_basics(graph, origin_node, destination_node)

        instance.analyze_map_clutter(graph, generate_plot=True, city_name=city_name)

        return instance

    def _initialize_basics(self, graph, origin_node, destination_node):

        # Get the edges, geometry and length of the route
        self.edges = list(nx.utils.pairwise(self.nodes))

        route_geometry = ox.routing.route_to_gdf(
            graph, self.nodes, weight=self.weightstring
        )["geometry"].unary_union
        merged_geometry_series = GeoSeries([route_geometry]).line_merge()

        merged_geometry = merged_geometry_series.iloc[0]

        if merged_geometry.geom_type == "LineString":
            self.route_linestring = merged_geometry
        elif merged_geometry.geom_type == "MultiLineString":
            self.route_linestring = LineString(list(merged_geometry.geoms[0].coords))
        else:
            raise ValueError(
                f"Expected LineString or MultiLineString, got {merged_geometry.geom_type}"
            )

        self.length = route_analysis.get_edges_sum(
            G=graph, route_edges=self.edges, weightstring="length"
        )

        self.directed_bearings, self.directed_weights = (
            route_analysis.extract_route_bearings(
                graph, self.nodes, get_undirected=True, weight="length"
            )
        )
        self.undirected_bearings, self.undirected_weights = (
            route_analysis.extract_route_bearings(
                graph, self.nodes, get_undirected=True, weight="length"
            )
        )
        
        self.undirected_bearing_dist_weighted, _ = route_analysis.get_route_bearing_dist(
            graph, self.nodes, get_undirected=True, num_bins=36, weight="length"
        )
        self.undirected_bearing_dist, _ = route_analysis.get_route_bearing_dist(
            graph, self.nodes, get_undirected=True, num_bins=36,
        )
        self.directed_bearing_dist_weighted, _ = route_analysis.get_route_bearing_dist(
            graph, self.nodes, get_undirected=True, num_bins=36, weight="length"
        )
        self.directed_bearing_dist, _ = route_analysis.get_route_bearing_dist(
            graph, self.nodes, get_undirected=True, num_bins=36,
        )

        # Get the attributes derived from Duckham and Kulik's simplest path search algorithm
        complexity_dict = route_analysis.get_route_complexity(graph, self.edges)
        self.complexity = complexity_dict["sum"]
        self.complexity_list = complexity_dict["complexity_list"]
        self.turn_types = complexity_dict["turn_types"]

        # Get the number of intersections, turns and frequency of turns
        self.turn_count = sum("turn" in s.lower() for s in self.turn_types)
        self.turn_frequency = self.turn_count / self.length
        self.n_nodes = len(self.nodes)

        # Get the number of segments and total turn degree
        self.n_segments = len(self.route_linestring.coords) - 1
        self.total_turn_degree = route_analysis.get_route_bearing_sum(graph, self.nodes)
        self.total_turn_degree_abs = route_analysis.get_route_bearing_sum(
            graph, self.nodes, absolute=True
        )
        self.avg_turn_degree = self.total_turn_degree_abs / self.n_segments

        # Get attributes of the nodes in the route
        self.sum_deviation_from_prototypical = route_analysis.get_edges_sum(
            G=graph, route_edges=self.edges, weightstring="deviation_from_prototypical"
        )
        self.avg_deviation_from_prototypical = (
            self.sum_deviation_from_prototypical / self.n_nodes
        )
        self.sum_node_degree = route_analysis.get_edges_sum(
            G=graph, route_edges=self.edges, weightstring="node_degree"
        )
        self.avg_node_degree = self.sum_node_degree / self.n_nodes
        self.sum_instruction_equivalent = route_analysis.get_edges_sum(
            G=graph, route_edges=self.edges, weightstring="instruction_equivalent"
        )
        self.sum_betweenness = route_analysis.get_nodes_sum(
            G=graph, route_nodes=self.nodes, weightstring="betweenness_centrality"
        )
        self.avg_betweenness = self.sum_betweenness / self.n_nodes

        # Get the betweenness centrality based on all shortest paths between the origin and destination
        self.sum_od_betweenness = (
            route_analysis.get_origin_destination_betweenness_centrality(
                graph, self.nodes, origin_node, destination_node
            )
        )
        self.avg_od_betweenness = self.sum_od_betweenness / self.n_nodes

    def analyze_map_clutter(self, graph, generate_plot=False, city_name=None):
        """Analyze map clutter metrics for this route."""
        if generate_plot and city_name:
            # Generate route plot
            route_gdf = ox.routing.route_to_gdf(graph, self.nodes, weight="length")
            cwd = os.getcwd()
            filepath = os.path.join(cwd, f"{city_name}_route_map.html")
            origin_node = self.nodes[0]
            destination_node = self.nodes[-1]

            self.map_bbox = mp.plot_route_gdf(
                graph,
                route_gdf,
                origin_node,
                destination_node,
                file_path=filepath,
                info_text=city_name,
                return_bbox=True,
            )
        else:
            # Get bbox without plotting
            self.map_bbox = map_analysis.get_routegdf_bbox(
                graph, self.nodes, buffer_percentage=0.1
            )

        # Calculate map clutter metrics
        self.map_road_length, self.map_intersection_count, self.map_street_count = (
            map_analysis.get_map_clutter(G=graph, map_bbox=self.map_bbox)
        )


    def get_route_dict(self):
        """Get a comprehensive dictionary with all route metrics and attributes."""
        route_dict = {

            # Basic route information
            "identifier": self.identifier,
            "length": self.length,
            "n_nodes": self.n_nodes,
            "n_segments": self.n_segments,
            
            # Bearing data
            "directed_bearings": self.directed_bearings.tolist(),
            "directed_weights": self.directed_weights.tolist(),
            "undirected_bearings": self.undirected_bearings.tolist(),
            "undirected_weights": self.undirected_weights.tolist(),
            "undirected_bearing_dist": self.undirected_bearing_dist.tolist(),
            "undirected_bearing_dist_weighted": self.undirected_bearing_dist_weighted.tolist(),
            "directed_bearing_dist": self.directed_bearing_dist.tolist(),
            "directed_bearing_dist_weighted": self.directed_bearing_dist_weighted.tolist(),

            # Complexity metrics
            "complexity": self.complexity,
            "complexity_list": self.complexity_list,
            "turn_types": self.turn_types,
            "turn_count": self.turn_count,
            "turn_frequency": self.turn_frequency,
            
            # Turn degree metrics
            "total_turn_degree": self.total_turn_degree,
            "total_turn_degree_abs": self.total_turn_degree_abs,
            "avg_turn_degree": self.avg_turn_degree,
            
            # Node attributes
            "sum_deviation_from_prototypical": self.sum_deviation_from_prototypical,
            "avg_deviation_from_prototypical": self.avg_deviation_from_prototypical,
            "sum_node_degree": self.sum_node_degree,
            "avg_node_degree": self.avg_node_degree,
            "sum_instruction_equivalent": self.sum_instruction_equivalent,
            "sum_betweenness": self.sum_betweenness,
            "avg_betweenness": self.avg_betweenness,
            "sum_od_betweenness": self.sum_od_betweenness,
            "avg_od_betweenness": self.avg_od_betweenness,
            }
            # Map-related attributes (may be None if not calculated)

        if self.map_bbox:
            route_dict.update({
            "map_bbox": self.map_bbox,
            "map_road_length": self.map_road_length,
            "map_intersection_count": self.map_intersection_count,
            "map_street_count": self.map_street_count,
            })
        
        return route_dict

    def generate_identifier(self,city_name,weightstring,origin_node, destination_node):
        route_string = f"{city_name}-{weightstring}-{origin_node}-{destination_node}"
        return route_string
