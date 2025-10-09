import numpy as np
import osmnx as ox
import logging
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
# Local modules
from . import alignment
from . import street_network_analysis
from . import geo_util
from . import od_pair_analysis
from .route import route
from . import orientation_plotting
from . import map_plotting
from . import route_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="app.log",
    filemode="w",
)


class od_pair:

    def __init__(self, G, origin, destination):
        """Initialize an OD pair with origin and destination nodes."""
        self.graph = G
        self._initialize_basics(origin, destination)
        self.path = None  # Initialize path as None
        self.path_map_bbox = None  # Initialize path map bbox as None
        self._analyze_environment()
        self._calculate_differences()

    @classmethod
    def from_route(cls, G, route_nodes, weightstring):

        instance = cls.__new__(cls)
        instance.graph = G
        # Set up the path object (unique to from_route)
        instance.path = route.from_nodes(G, route_nodes, weightstring=weightstring)
        instance.path_map_bbox = instance.path.map_bbox
        print(f"Path map bbox: {instance.path_map_bbox}")
        # Reuse existing helpers where possible
        instance._initialize_basics(route_nodes[0], route_nodes[-1])

        # Analyze environment with bbox instead of polygon
        instance._analyze_environment()

        return instance

    def _initialize_basics(self, origin, destination):
        """Set up basic attributes for the OD pair."""
        self.city_name = self.graph.graph["city_name"]
        self.origin_node = origin
        self.destination_node = destination
        self.id = (
            self.city_name + "-" + str(self.origin_node) + "-" + str(self.destination_node)
        )
        self.origin_point = (
            self.graph.nodes[self.origin_node]["y"],
            self.graph.nodes[self.origin_node]["x"],
        )
        self.destination_point = (
            self.graph.nodes[self.destination_node]["y"],
            self.graph.nodes[self.destination_node]["x"],
        )
        # Calculate the great circle distance between the origin and destination
        self.od_distance = float(
            ox.distance.great_circle(
                lat1=self.origin_point[0],
                lon1=self.origin_point[1],
                lat2=self.destination_point[0],
                lon2=self.destination_point[1],
            )
        )
        self.shortest_path = route(
            self.graph,
            origin=self.origin_node,
            destination=self.destination_node,
            weightstring="length",
        )
        self.simplest_path = route(
            self.graph,
            origin=self.origin_node,
            destination=self.destination_node,
            weightstring="decision_complexity",
        )

        self.shape_dict = geo_util.get_od_pair_polygon(
            self.origin_point, self.destination_point
        )
        self.polygon = self.shape_dict["polygon"]
        self.bbox = self.shape_dict["wsen_bbox"]
        self.bbox_polygon = self.shape_dict["bbox_polygon"]

        self.cardinal_direction = od_pair_analysis.get_od_cardinal_direction(
            G=self.graph, origin=self.origin_node, destination=self.destination_node
        )

    def get_subgraph(self):
        """Get the subgraph for the OD pair."""
        if self.path:
            return od_pair_analysis.get_od_pair_subgraph(
                G=self.graph, bbox=self.path_map_bbox
            )
        else:
            return od_pair_analysis.get_od_pair_subgraph(
                G=self.graph, polygon=self.polygon
            )

    def _analyze_environment(self):
        # Choose subgraph method based on parameters
        if self.path:
            subgraph = od_pair_analysis.get_od_pair_subgraph(
                G=self.graph, bbox=self.path_map_bbox
            )
            self.area = geo_util.calculate_bbox_area_with_utm(self.path_map_bbox)
        else:
            subgraph = od_pair_analysis.get_od_pair_subgraph(
                G=self.graph, polygon=self.polygon
            )
            self.area = geo_util.calculate_area_with_utm(self.polygon)

        undirected_subgraph = ox.convert.to_undirected(subgraph)
        self.subgraph_stats = ox.stats.basic_stats(subgraph, area=self.area)
        betweenness_centrality = nx.betweenness_centrality(subgraph, normalized=True)
        nx.set_node_attributes(subgraph, betweenness_centrality, "betweenness_centrality")
        avg_node_betweenness = street_network_analysis.get_node_avg(subgraph, 'betweenness_centrality')
        # Add avg_node_betweenness to subgraph_stats_dict
        self.subgraph_stats['avg_node_betweenness'] = avg_node_betweenness

        route_avg_betweenness = route_analysis.get_nodes_avg(subgraph,self.path.nodes,weightstring="betweenness_centrality")
        self.subgraph_stats['avg_routenode_betweenness'] = route_avg_betweenness
        # Environment bearing data

        env_undirected_bearings, env_undirected_weights = (
            street_network_analysis.extract_edge_bearings(
                G=undirected_subgraph, min_length=10, weight="length"
            )
        )

        

        env_directed_bearings, env_directed_weights = (
            street_network_analysis.extract_edge_bearings(
                G=subgraph, min_length=10, weight="length"
            )
        )

        env_undirected_bearing_dist_weighted, _ = street_network_analysis.bearings_distribution(
            G=undirected_subgraph, num_bins=36, min_length=10, weight="length"
        )
        env_undirected_bearing_dist, _ = street_network_analysis.bearings_distribution(
            G=undirected_subgraph, num_bins=36, min_length=10, weight=None
        )
        env_directed_bearing_dist_weighted, _ = street_network_analysis.bearings_distribution(
            G=subgraph, num_bins=36, min_length=10, weight="length"
        )
        env_directed_bearing_dist, _ = street_network_analysis.bearings_distribution(
            G=subgraph, num_bins=36, min_length=10, weight=None
        )

        od_undirected_bearing_dist = od_pair_analysis.get_od_pair_bearing_dist(
            self.shape_dict["fwd_bearing"], self.shape_dict["bwd_bearing"]
        )
        od_undirected_bearing_perp_dist = od_pair_analysis.get_od_pair_bearing_dist(
            fwd=self.shape_dict["fwd_bearing"], bwd=self.shape_dict["bwd_bearing"], 
            perp_bwd= self.shape_dict["perpendicular_bwd_bearing"], perp_fwd=self.shape_dict["perpendicular_fwd_bearing"]
        )
        od_directed_bearing_dist = od_pair_analysis.get_od_pair_bearing_dist(
            self.shape_dict["fwd_bearing"]
        )

        od_bearings_array = np.array(
            [self.shape_dict["fwd_bearing"],self.shape_dict["bwd_bearing"],self.shape_dict["perpendicular_fwd_bearing"], self.shape_dict["perpendicular_bwd_bearing"]]
        )

        self.cot_alignment = alignment.COT_sample_distance(route_bearings=od_bearings_array,env_bearings= env_undirected_bearings)

        p_od_undirected_bearing_perp_dist = od_undirected_bearing_perp_dist / sum(od_undirected_bearing_perp_dist)
        p_env_undirected_bearing_dist = env_undirected_bearing_dist / sum(env_undirected_bearing_dist)
        self.cot_distribution_alignment = alignment.COT_distribution_distance(route_dist=p_env_undirected_bearing_dist,env_dist= p_od_undirected_bearing_perp_dist)

        self.bearing_data = {
            "undirected": env_undirected_bearings,
            "undirected_weights": env_undirected_weights,
            "directed": env_directed_bearings,
            "directed_weights": env_directed_weights,
            "od_fwd": self.shape_dict["fwd_bearing"],
            "od_bwd": self.shape_dict["bwd_bearing"],
            "od_perp_fwd": self.shape_dict["perpendicular_fwd_bearing"],
            "od_perp_bwd": self.shape_dict["perpendicular_bwd_bearing"],
        }
        self.bearing_dist_data = {
            "env_directed_dist_weighted": env_directed_bearing_dist_weighted,
            "env_directed_dist": env_directed_bearing_dist,
            "env_undirected_dist_weighted": env_undirected_bearing_dist_weighted,
            "env_undirected_dist": env_undirected_bearing_dist,
            "od_undirected_dist": od_undirected_bearing_dist,
            "od_directed_dist": od_directed_bearing_dist,
            "od_undirected_dist_perp": od_undirected_bearing_perp_dist,
        }
        # Distribution properties of undirected environment bearings
        environment_orientation_undirected_entropy_weighted = (
            street_network_analysis.orientation_entropy(
                undirected_subgraph, num_bins=36, weight="length"
            )
        )
        environment_orientation_undirected_entropy = (
            street_network_analysis.orientation_entropy(
                undirected_subgraph, num_bins=36
            )
        )
        order_undirected_weighted = street_network_analysis.get_orientation_order(
            environment_orientation_undirected_entropy_weighted
        )

        order_undirected = street_network_analysis.get_orientation_order(
            environment_orientation_undirected_entropy
        )

        # Distribution properties of undirected environment bearings
        environment_orientation_directed_entropy_weighted = (
            street_network_analysis.orientation_entropy(
                subgraph, num_bins=36, weight="length"
            )
        )
        environment_orientation_directed_entropy = (
            street_network_analysis.orientation_entropy(
                subgraph, num_bins=36
            )
        )
        order_directed_weighted = street_network_analysis.get_orientation_order(
            environment_orientation_directed_entropy_weighted
        )

        order_directed = street_network_analysis.get_orientation_order(
            environment_orientation_directed_entropy
        )



        self.bearing_dist_properties = {
            ""
            "environment_orientation_entropy_weighted": environment_orientation_undirected_entropy_weighted,
            "environment_orientation_entropy": environment_orientation_undirected_entropy,
            "environment_orientation_order_weighted": order_undirected_weighted,
            "environment_orientation_order": order_undirected,
            "environment_orientation_directed_entropy_weighted": environment_orientation_directed_entropy_weighted,
            "environment_orientation_directed_entropy": environment_orientation_directed_entropy,
            "environment_orientation_order_directed_weighted": order_directed_weighted,
            "environment_orientation_order_directed": order_directed,
        }

    def _calculate_differences(self):
        """Calculate differences between routes and OD distance."""
        self.length_diff = self.simplest_path.length - self.shortest_path.length
        self.complexity_diff = int(self.shortest_path.complexity) - int(self.simplest_path.complexity)
        self.shortest_diff = self.shortest_path.length - self.od_distance
        try:
            # Safely access hausdorff_distance method using getattr
            hausdorff_method = getattr(self.shortest_path.route_linestring, 'hausdorff_distance', None)
            if hausdorff_method and callable(hausdorff_method):
                self.hausdorff_diff = hausdorff_method(self.simplest_path.route_linestring)
            else:
                self.hausdorff_diff = 0.0
        except Exception:
            # Handle any errors with geometric calculations
            self.hausdorff_diff = 0.0

    def create_orientation_plot(self, filepath):
        """Create an orientation plot showing environment and route distributions."""
        env_dist = self.bearing_dist_data["env_undirected_dist"]
        env_dist = env_dist / env_dist.sum()
        print(f"Environment distribution: {env_dist}")
        fig, ax = orientation_plotting.plot_orientation(env_dist)
        r_dist = self.bearing_dist_data["od_undirected_dist_perp"]
        r_dist = r_dist / r_dist.sum()
        orientation_plotting._plot_overlaid_distribution(
            ax=ax, new_distribution=r_dist, num_bins=36
        )
        fig.savefig(filepath)

    def create_alignment_plot(self, filepath):
        """Create an alignment plot showing route and environment correlation."""
        # Get normalized distributions for plotting
        env_dist = self.bearing_dist_data["env_undirected_dist_weighted"]
        env_dist = alignment.wrap_dist(env_dist)
        env_dist = env_dist / np.sum(env_dist)

        route_dist = self.bearing_dist_data["od_undirected_dist_perp"]
        route_dist = alignment.wrap_dist(route_dist)
        route_dist = route_dist / np.sum(route_dist)

        # Create the plot
        fig, ax = orientation_plotting.plot_alignment_orientation(
            bin_counts=env_dist
        )
        orientation_plotting._plot_overlaid_alignment_distribution(
            ax,
            route_dist,
            0,  # strongest_env_index
            0,  # closest_env_index
            num_bins=18,
        )
        fig.savefig(filepath)

    def plot_on_map(self,html_path,graph_plot_path):
        route_gdf = ox.routing.route_to_gdf(self.graph, self.shortest_path.nodes, weight='length')
        filepath = html_path
        polygon = self.polygon
        map_plotting.plot_route_gdf(G=self.graph, start_node=self.origin_node,
                                        end_node=self.destination_node,
                                        route_gdf=route_gdf,
                                        map_tiles="OpenStreetMap.Mapnik",
                                        file_path=filepath,
                                        truncation_polygon=polygon,
                                        cot_dist=self.cot_distribution_alignment)

        if self.path:
            subgraph = od_pair_analysis.get_od_pair_subgraph(
                G=self.graph, bbox=self.path_map_bbox
            )
            self.area = geo_util.calculate_bbox_area_with_utm(self.path_map_bbox)
        else:
            subgraph = od_pair_analysis.get_od_pair_subgraph(
                G=self.graph, polygon=self.polygon
            )
            self.area = geo_util.calculate_area_with_utm(self.polygon)

        try:
        #undirected_subgraph = ox.convert.to_undirected(subgraph)
            fig,_ = ox.plot_graph_route(
                subgraph,
                self.shortest_path.nodes,
                node_size=5,
                bgcolor='white',
                route_color='blue',
                edge_color="black",
                node_color="black",
                route_linewidth=4,
                edge_linewidth=0.5,
                show=False,
                close=False
            )
            graph_plot_path
            fig.savefig(graph_plot_path, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"error {e}")

    def _calculate_alignment_metrics(self, directed=True, weighted=True):
        """Calculate all alignment metrics between route and environment."""
        
        # Select the appropriate distributions based on parameters
        if directed:
            route_dist = self.bearing_dist_data["od_directed_dist"]
            if weighted:
                env_dist= self.bearing_dist_data["env_directed_dist_weighted"]
            else:
                env_dist = self.bearing_dist_data["env_directed_dist"]

            # Process directed distributions
            max_index = np.argmax(route_dist)
            route_dist = alignment.center_around_index(route_dist, max_index)
            env_dist = alignment.center_around_index(env_dist, max_index)
            route_dist = route_dist / np.sum(route_dist)
            env_dist = env_dist / np.sum(env_dist)
            
        else:
            route_dist = self.bearing_dist_data["od_undirected_dist"]
            if weighted:
                env_dist = self.bearing_dist_data["env_undirected_dist_weighted"]
            else:
                env_dist = self.bearing_dist_data["env_undirected_dist"]
            
            # Process undirected distributions
            max_index = np.argmax(route_dist)
            route_dist = alignment.wrap_dist(route_dist)
            env_dist = alignment.wrap_dist(env_dist)
            route_dist = alignment.center_around_index(route_dist, max_index)
            env_dist = alignment.center_around_index(env_dist, max_index)
            route_dist = route_dist / np.sum(route_dist)
            env_dist = env_dist / np.sum(env_dist)

        # Calculate alignment metrics using the selected distributions
        strongest_correlation, closest_strongest_correlation = (
            alignment.crosscorrelate_alignment(route_dist, env_dist)
        )

        return {
            "strongest_correlation": strongest_correlation,
            "closest_strongest_correlation": closest_strongest_correlation
        }

    def get_odpair_df(self):
        # 1 Basics
        basic_dict = {
            "id": self.id,
            "city_name": self.city_name,
            "origin_node": self.origin_node,
            "origin_point": str(self.origin_point),
            "destination_node": self.destination_node,
            "destination_point": str(self.destination_point),
            "od_distance": self.od_distance,
            "od_cardinal_direction": self.cardinal_direction,
            "cot_alignment": self.cot_alignment,
            "cot_dist_alignment": self.cot_distribution_alignment
        }


        # 3 Bearing distribution data
        bearing_dist_dict = {}
        for key, value in self.bearing_dist_data.items():
            if isinstance(value, (int, float, str, bool, np.integer, np.floating)):
                bearing_dist_dict[f"bearings_dist_{key}"] = value
            elif isinstance(value, list):
                bearing_dist_dict[f"bearings_dist_{key}"] = value
            else:
                bearing_dist_dict[f"bearings_dist_{key}"] = value.tolist()


        # 4 Bearing distribution properties
        distribution_properties_dict = {}
        for key, value in self.bearing_dist_properties.items():
            if isinstance(value, (int, float, str, bool, np.integer, np.floating)):
                distribution_properties_dict[f"bearing_dist_prop_{key}"] = value
            elif isinstance(value, list):
                distribution_properties_dict[f"bearing_dist_prop_{key}"] = value
            else:
                distribution_properties_dict[f"bearing_dist_prop_{key}"] = value.tolist()
                

        # 5 Route dict
        path_dict = self._build_paths_dict()

        # 6 subgraph stats dict
        subgraph_stats_dict = {}
        for key, value in self.subgraph_stats.items():
            if isinstance(value, (int, float, str, bool, np.integer, np.floating)):
                subgraph_stats_dict[f"subgraph_stats_{key}"] = value
            elif isinstance(value, dict):
                subgraph_stats_dict[f"subgraph_stats_{key}"] = str(value)
            else:
                subgraph_stats_dict[f"subgraph_stats_{key}"] = value.tolist()

        

        odpair_dict = {}
        odpair_dict.update(basic_dict)
        #odpair_dict.update(bearing_dict)
        odpair_dict.update(bearing_dist_dict)
        odpair_dict.update(distribution_properties_dict)
        odpair_dict.update(subgraph_stats_dict)
        odpair_dict.update(path_dict) # type: ignore

        odpair_df = pd.DataFrame([odpair_dict]).set_index('id')
        
        return odpair_df

    def _build_paths_dict(self):
        if self.path:
            path_dict = {}
            single_path_dict = self.path.get_route_dict()
            for key, value in single_path_dict.items():
                if isinstance(value, (int, float, str, bool, np.integer, np.floating)):
                    path_dict[f"path_{key}"] = value
                elif isinstance(value, list):
                    path_dict[f"path_{key}"] = value
                else:
                    path_dict[f"path_{key}"] = str(value)



            return path_dict

        else:
            path_dict = {}
            shortest_path_dict = self.shortest_path.get_route_dict()
            for key, value in shortest_path_dict.items():
                if isinstance(value, (int, float, str, bool, np.integer, np.floating)):
                    path_dict[f"shortest_{key}"] = value
                elif isinstance(value, list):
                    path_dict[f"shortest_{key}"] = value
                else:
                    path_dict[f"shortest_{key}"] = value.tolist()

            simplest_path_dict = self.simplest_path.get_route_dict()
            for key, value in simplest_path_dict.items():
                if isinstance(value, (int, float, str, bool, np.integer, np.floating)):
                    path_dict[f"simplest_{key}"] = value
                elif isinstance(value, list):
                    path_dict[f"simplest_{key}"] = value
                else:
                    path_dict[f"simplest_{key}"] = value.tolist()

            differences = {
            "length_diff": self.length_diff,
            "complexity_diff": self.complexity_diff,
            "shortest_diff": self.shortest_diff,
            "hausdorff_diff": self.hausdorff_diff,
            }

            path_dict.update(differences)

            return path_dict

    def get_geometry_dict(self):
        """Build the geometry and area dictionary."""
        bearing_dict = {}
        for key, value in self.bearing_data.items():
            if isinstance(value, (int, float, str, bool, np.integer, np.floating)):
                bearing_dict[f"bearings_{key}"] = value
            elif isinstance(value, list):
                bearing_dict[f"bearings_{key}"] = value
            else:
                bearing_dict[f"bearings_{key}"] = value.tolist()

        if not self.path:
            geom_dict = {
                "id": self.id,
                "city_name": self.city_name,
                "origin_node": self.origin_node,
                "destination_node": self.destination_node,
                "origin_point": str(self.origin_point),
                "destination_point": str(self.destination_point),
                "shortest_route_nodes": str(self.shortest_path.nodes),
                "shortest_route_linestring": str(self.shortest_path.route_linestring),
                "simplest_route_nodes": str(self.simplest_path.nodes),
                "simplest_route_linestring": str(self.simplest_path.route_linestring),
                "bbox": str(self.bbox),
                "diamond": str(self.polygon.wkt),
                "area": self.area,
            }
            geom_dict.update(bearing_dict)
            return geom_dict
        else:
            geom_dict =  {
                "id": self.id,
                "city_name": self.city_name,
                "origin_node": self.origin_node,
                "destination_node": self.destination_node,
                "origin_point": str(self.origin_point),
                "destination_point": str(self.destination_point),
                "route_nodes": str(self.path.nodes),
                "route_linestring": str(self.path.route_linestring),
                "bbox": str(self.path.map_bbox),
                "diamond": str(self.polygon.wkt),
                "area": self.area,
            }
            geom_dict.update(bearing_dict)
            return geom_dict
