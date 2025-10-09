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
        self.identifier = self.generate_identifier(
            city_name=graph.graph["city_name"],
            weightstring=weightstring,
            origin_node=origin_node,
            destination_node=destination_node,
        )

    @classmethod
    def from_nodes(cls, graph, nodes, weightstring):
        """
        Construct a route object from a list of nodes in a graph given a weightstring.
        """
        instance = cls.__new__(cls)

        # Set the attributes that constrain the route
        graph = graph
        instance.nodes = nodes
        instance.weightstring = weightstring
        origin_node = nodes[0]
        destination_node = nodes[-1]
        city_name = graph.graph["city_name"]
        instance.identifier = instance.generate_identifier(
            city_name, instance.weightstring, origin_node, destination_node
        )
        instance._initialize_basics(graph, origin_node, destination_node)

        instance.analyze_map_clutter(graph, generate_plot=False, city_name=city_name)

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

        self.undirected_bearing_dist_weighted, _ = (
            route_analysis.get_route_bearing_dist(
                graph, self.nodes, get_undirected=True, num_bins=36, weight="length"
            )
        )
        self.undirected_bearing_dist, _ = route_analysis.get_route_bearing_dist(
            graph,
            self.nodes,
            get_undirected=True,
            num_bins=36,
        )
        self.directed_bearing_dist_weighted, _ = route_analysis.get_route_bearing_dist(
            graph, self.nodes, get_undirected=True, num_bins=36, weight="length"
        )
        self.directed_bearing_dist, _ = route_analysis.get_route_bearing_dist(
            graph,
            self.nodes,
            get_undirected=True,
            num_bins=36,
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

    def analyze_map_clutter(self, graph, generate_plot=False, city_name=None):
        """Analyze map clutter metrics for this route."""
        route_gdf = ox.routing.route_to_gdf(graph, self.nodes, weight="length")
        origin_node = self.nodes[0]
        destination_node = self.nodes[-1]
        
        if generate_plot and city_name:
            cwd = os.getcwd()
            filepath = os.path.join(cwd, f"{city_name}_route_map.html")
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
            geom = route_gdf["geometry"].unary_union
            route_gdf["geometry"] = geo_util.merge_and_simplify_geometry(geom, 0.0001)
            start_location = (graph.nodes[origin_node]["y"], graph.nodes[origin_node]["x"])
            end_location = (graph.nodes[destination_node]["y"], graph.nodes[destination_node]["x"])
            midpoint = (
                (start_location[0] + end_location[0]) / 2,
                (start_location[1] + end_location[1]) / 2,
            )
            self.map_bbox = map_analysis.calculate_bounding_box(center_lat=midpoint[0],center_lng=midpoint[1])
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
        }

        if self.map_bbox:
            route_dict.update(
                {
                    "map_bbox": self.map_bbox,
                    "map_road_length": self.map_road_length,
                    "map_intersection_count": self.map_intersection_count,
                    "map_street_count": self.map_street_count,
                }
            )

        return route_dict

    def generate_identifier(
        self, city_name, weightstring, origin_node, destination_node
    ):
        route_string = f"{city_name}-{weightstring}-{origin_node}-{destination_node}"
        return route_string
