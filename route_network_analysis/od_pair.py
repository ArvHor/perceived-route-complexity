import math
import numpy as np
import networkx as nx
import osmnx as ox
import logging
import hashlib
from matplotlib.projections.polar import PolarAxes
import matplotlib.pyplot as plt
from scipy.signal import correlate

# Local modules
from . import alignment
from . import street_network_analysis
from . import geo_util
from . import route_analysis
from . import od_pair_analysis
from . import map_plotting
from .route import route
from . import orientation_plotting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="app.log",
    filemode="w",
)


class od_pair:
    def __init__(self, G, origin, destination):
        # Set the basic attributes
        graph = G
        self.city_name = graph.graph["city_name"]
        self.origin_node = origin
        self.destination_node = destination
        self.origin_point = (
            graph.nodes[self.origin_node]["y"],
            graph.nodes[self.origin_node]["x"],
        )
        self.destination_point = (
            graph.nodes[self.destination_node]["y"],
            graph.nodes[self.destination_node]["x"],
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

        # Find the simplest and shortest route from the origin to the destination
        self.shortest_path = route(
            graph,
            origin=self.origin_node,
            destination=self.destination_node,
            weightstring="length",
        )
        self.simplest_path = route(
            graph,
            origin=self.origin_node,
            destination=self.destination_node,
            weightstring="decision_complexity",
        )

        # Get geometric properties of the origin and destination
        self.shape_dict = geo_util.get_od_pair_polygon(
            self.origin_point, self.destination_point
        )
        self.polygon = self.shape_dict[
            "polygon"
        ]  # Square origin and destination as the diagonal of a square
        self.bbox = self.shape_dict[
            "wsen_bbox"
        ]  # Bounding box as `(left, bottom, right, top)`.
        self.bbox_polygon = self.shape_dict["bbox_polygon"]

        self.cardinal_direction = od_pair_analysis.get_od_cardinal_direction(
            G=graph, origin=self.origin_node, destination=self.destination_node
        )

        # self.map_bbox = self.shortest_path.map_bbox

        subgraph = od_pair_analysis.get_od_pair_subgraph(G=graph, polygon=self.polygon)
        undirected_subgraph = ox.convert.to_undirected(subgraph)
        self.area = geo_util.calculate_area_with_utm(self.polygon)
        self.subgraph_stats = ox.stats.basic_stats(subgraph, area=self.area)

        logging.info(
            f"Creating od_pair for graph {graph.graph['city_name']} with n self.subgraph edges: {len(subgraph.edges)}, {len(undirected_subgraph.edges)}"
        )

        self.env_bearing_dist_weighted, _ = (
            street_network_analysis.bearings_distribution(
                G=undirected_subgraph, num_bins=36, min_length=10, weight="length"
            )
        )
        self.env_bearing_dist, _ = street_network_analysis.bearings_distribution(
            G=undirected_subgraph, num_bins=36, min_length=10, weight=None
        )

        self.route_direction_bearing_dist = od_pair_analysis.get_od_pair_bearing_dist(
            self.shape_dict["fwd_bearing"], self.shape_dict["bwd_bearing"]
        )

        self.environment_orientation_entropy_weighted = (
            street_network_analysis.orientation_entropy(
                undirected_subgraph, num_bins=36, weight="length"
            )
        )
        self.environment_orientation_entropy = (
            street_network_analysis.orientation_entropy(
                undirected_subgraph, num_bins=36
            )
        )

        self.order_weighted = street_network_analysis.get_orientation_order(
            self.environment_orientation_entropy_weighted
        )
        self.order = street_network_analysis.get_orientation_order(
            self.environment_orientation_entropy
        )

        self.length_diff = self.simplest_path.length - self.shortest_path.length
        self.complexity_diff = int(self.simplest_path.complexity) - int(
            self.shortest_path.complexity
        )
        self.shortest_diff = self.shortest_path.length - self.od_distance

    @classmethod
    def from_route(cls, G, route_nodes, weightstring):
        # Create instance without calling __init__
        instance = cls.__new__(cls)

        instance.path = route.from_nodes(G, route_nodes, weightstring=weightstring)
        instance.path_map_bbox = instance.path.map_bbox

        # Set the basic attributes
        graph = G
        instance.city_name = G.graph["city_name"]
        instance.origin_node = route_nodes[0]
        instance.destination_node = route_nodes[-1]
        instance.origin_point = (
            graph.nodes[instance.origin_node]["y"],
            graph.nodes[instance.origin_node]["x"],
        )
        instance.destination_point = (
            graph.nodes[instance.destination_node]["y"],
            graph.nodes[instance.destination_node]["x"],
        )

        # Calculate OD distance
        instance.od_distance = float(
            ox.distance.great_circle(
                lat1=instance.origin_point[0],
                lon1=instance.origin_point[1],
                lat2=instance.destination_point[0],
                lon2=instance.destination_point[1],
            )
        )

        # Generate geometry
        instance.shape_dict = geo_util.get_od_pair_polygon(
            instance.origin_point, instance.destination_point
        )
        instance.polygon = instance.shape_dict["polygon"]
        instance.bbox = instance.shape_dict["wsen_bbox"]
        instance.bbox_polygon = instance.shape_dict["bbox_polygon"]

        # instance.map_bbox = instance.path.map_bbox

        instance.cardinal_direction = od_pair_analysis.get_od_cardinal_direction(
            G=graph,
            origin=instance.origin_node,
            destination=instance.destination_node,
        )

        subgraph = od_pair_analysis.get_od_pair_subgraph(
            G=graph, bbox=instance.path_map_bbox
        )
        undirected_subgraph = ox.convert.to_undirected(subgraph)
        instance.area = geo_util.calculate_bbox_area_with_utm(instance.path_map_bbox)
        subgraph_stats = ox.stats.basic_stats(subgraph, area=instance.area)

        instance.env_bearing_dist_weighted, _ = (
            street_network_analysis.bearings_distribution(
                G=undirected_subgraph,
                num_bins=36,
                min_length=10,
                weight="length",
            )
        )
        instance.env_bearing_dist, _ = street_network_analysis.bearings_distribution(
            G=undirected_subgraph, num_bins=36, min_length=10, weight=None
        )

        instance.route_direction_bearing_dist = (
            od_pair_analysis.get_od_pair_bearing_dist(
                instance.shape_dict["fwd_bearing"], instance.shape_dict["bwd_bearing"]
            )
        )

        instance.environment_orientation_entropy_weighted = (
            street_network_analysis.orientation_entropy(
                undirected_subgraph, num_bins=36, weight="length"
            )
        )
        instance.environment_orientation_entropy = (
            street_network_analysis.orientation_entropy(
                undirected_subgraph, num_bins=36
            )
        )

        instance.order_weighted = street_network_analysis.get_orientation_order(
            instance.environment_orientation_entropy_weighted
        )
        instance.order = street_network_analysis.get_orientation_order(
            instance.environment_orientation_entropy
        )

        instance.stats_edge_count = subgraph_stats["m"]
        instance.stats_node_count = subgraph_stats["n"]
        instance.stats_street_segment_count = subgraph_stats["street_segment_count"]
        instance.stats_streets_per_node_avg = subgraph_stats["streets_per_node_avg"]
        instance.stats_streets_per_node_counts = subgraph_stats[
            "streets_per_node_counts"
        ]
        instance.stats_intersection_density_km = subgraph_stats[
            "intersection_density_km"
        ]
        instance.stats_intersection_count = subgraph_stats["intersection_count"]
        instance.stats_k_avg = subgraph_stats["k_avg"]
        instance.stats_street_length_total = subgraph_stats["street_length_total"]
        instance.stats_street_length_avg = subgraph_stats["street_length_avg"]
        instance.stats_circuity_avg = subgraph_stats["circuity_avg"]
        instance.stats_node_density_km = subgraph_stats["node_density_km"]

        return instance

    def generate_identifier(self):
        identifier = str(self.origin_node) + "-" + str(self.destination_node)

        hash_object = hashlib.sha256(identifier.encode())
        hex_dig = hash_object.hexdigest()
        identifier = hex_dig
        return identifier

    def get_subgraph(self, graph):
        subgraph = od_pair_analysis.get_od_pair_subgraph(G=graph, polygon=self.polygon)
        return subgraph
    
    def create_orientation_plot(self, filepath):
            env_dist = self.env_bearing_dist_weighted / self.env_bearing_dist_weighted.sum()
            fig, ax = orientation_plotting.plot_orientation(env_dist)
            r_dist = self.route_direction_bearing_dist / self.route_direction_bearing_dist.sum()
            self._plot_overlaid_distribution(ax, r_dist, num_bins=36)
            fig.savefig(filepath)


    def _plot_overlaid_distribution(self,
                                    ax: plt.PolarAxes,
                                    new_distribution: np.ndarray,
                                    num_bins: int,
                                    ) -> None:
        bin_centers = 360 / num_bins * np.arange(num_bins)
        positions = np.radians(bin_centers)
        width = 2 * np.pi / num_bins

        # Normalize the new distribution to calculate height/area
        new_bin_frequency = new_distribution / new_distribution.sum()

        new_radius = new_bin_frequency

        # Plot the histogram
        ax.bar(
            positions,
            height=new_radius,
            width=width,
            align="center",
            bottom=0,
            zorder=4, 
            edgecolor="k",
            linewidth=0.5,
            facecolor="none", 
            alpha=1,  
            hatch=".",
            label="Route",
        )
        ax.bar(
            positions,
            height=new_radius,
            width=width,
            align="center",
            bottom=0,
            zorder=4, 
            edgecolor="k",
            linewidth=0.5,
            facecolor="blue", 
            alpha=0.2,
        )

        # Set the radial limits to fit the data
        ax.set_ylim(0, new_radius.max() * 1.1)

        # Set radial ticks to indicate each 10%, and label them accordingly
        ax.set_yticks([i * 0.1 for i in range(6)])
        ax.set_yticklabels([f"{int(i * 10)}%" for i in range(6)])

        # Set the angular range to show the full circle
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)


    def create_alignment_plot(self, filepath,peaks=True):
        #subgraph = self.get_subgraph(graph)
        #undirected_subgraph = ox.convert.to_undirected(subgraph)
        
        env_dist = self.env_bearing_dist_weighted
        route_dist = self.route_direction_bearing_dist
        
        strongest_crosscorr, closest_strongest_correlation = alignment.find_optimal_correlation(route_dist, env_dist)
        closest_lag = closest_strongest_correlation["lag"]
        strongest_lag = strongest_crosscorr["lag"]

        closest_lag = closest_lag - len(route_dist) // 2
        strongest_lag = strongest_lag - len(route_dist) // 2

        peak_alignment = alignment.find_peaks_alignment(
            route_dist, env_dist
        )
        strongest_peak = peak_alignment["strongest_env_peak"]
        closest_peak = peak_alignment["closest_env_peak"]

        print("strongest_lag: ", strongest_lag)
        print("closest_lag: ", closest_lag) 
        print("strongest_peak: ", strongest_peak)
        print("closest_peak: ", closest_peak)

        # prepare the distributions for plotting
        env_dist = alignment.fold_dist(self.env_bearing_dist_weighted)
        route_dist = alignment.fold_dist(self.route_direction_bearing_dist)
        #print("len env_dist: ", len(env_dist))
        #print("len route_dist: ", len(route_dist))
        route_dist = route_dist / route_dist.sum()
        env_dist = env_dist / env_dist.sum()

        max_index = np.argmax(route_dist)

        route_dist = alignment.roll_to_max(route_dist, max_index)
        env_dist = alignment.roll_to_max(env_dist, max_index)

        # Plot the distributions
        fig, ax = orientation_plotting.plot_alignment_orientation(env_dist)
        if peaks:
            self._plot_overlaid_alignment_distribution(ax, route_dist, strongest_peak, closest_peak, num_bins=18)
        else:
            self._plot_overlaid_alignment_distribution(ax, route_dist, strongest_lag, closest_lag, num_bins=18)
        fig.savefig(filepath)

    def _plot_overlaid_alignment_distribution(
        self,
        ax: plt.PolarAxes,
        new_distribution: np.ndarray,
        strongest_peak: int,
        closest_peak: int,
        num_bins: int,
    ) -> None:
        # Calculate bin centers from 0 to 180 degrees
        bin_centers = np.arange(0, 180, 180 / num_bins) 
        positions = np.radians(bin_centers)
        width = 2 * np.pi / 36  # Each bin is 10 degrees wide

        new_radius = new_distribution

        # Plot the histogram
        ax.bar(
            positions,
            height=new_radius,
            width=width,
            align="center",
            bottom=0,
            zorder=4, 
            edgecolor="k",
            linewidth=0.5,
            facecolor="none", 
            alpha=1,  
            hatch=".",
            label="Route",
        )
        ax.bar(
            positions,
            height=new_radius,
            width=width,
            align="center",
            bottom=0,
            zorder=4, 
            edgecolor="k",
            linewidth=0.5,
            facecolor="blue", 
            alpha=0.2,
        )

        # Second bar: transparent face, opaque hatch
        ax.bar(
            positions[strongest_peak],
            height=1,
            width=width,
            align="center",
            bottom=0,
            zorder=5,
            edgecolor="k",
            linewidth=0.5,
            facecolor="none",
            hatch="--",
            alpha=1,
            label="Strongest Peak",
        )

        ax.bar(
            positions[closest_peak],
            height=1,
            width=width,
            align="center",
            bottom=0,
            zorder=4,
            edgecolor="k",
            linewidth=0.5,
            facecolor="none", 
            alpha=1,
            hatch="||",
            label="Closest Peak",
        )
        
        ax.bar(
            positions[closest_peak],
            height=1,
            width=width,
            align="center",
            bottom=0,
            zorder=4,
            edgecolor="k",
            linewidth=0.5,
            facecolor="yellow",      #"#ff7f0e", # orange
            alpha=0.3,
            label="Closest Peak",
        )
        # Add a transparent red bar at the bin corresponding to the lag
        ax.bar(
            positions[strongest_peak],
            height=1,
            width=width,
            align="center",
            bottom=0,
            zorder=4,
            edgecolor="none",
            linewidth=0.5,
            facecolor="red",      #"#d62728", # red
            alpha=0.3,
            label=None,
        )
        
        # Set the radial limits to 50%
        ax.set_ylim(0, 0.6)

        # Set radial ticks to indicate each 10%, and label them accordingly
        ax.set_yticks([i * 0.1 for i in range(7)])
        ax.set_yticklabels([f"{int(i * 10)}%" for i in range(7)])

        # Set angular (x) ticks from -90 to +90 degrees
        xticks_deg = np.arange(-90, 91, 10)
        ax.set_xticks(np.radians(xticks_deg + 90))  # shift so 0 is at center
        ax.set_xticklabels([f"{int(deg)}°" for deg in xticks_deg])

        # Set the theta limits to display only from 0 to 180 degrees
        ax.set_theta_zero_location("W")
        ax.set_theta_direction(-1)  # Set the direction to counter-clockwise
        ax.set_thetamin(-5)
        ax.set_thetamax(175)

    def get_comparison_dict(self):
        env_dist = self.env_bearing_dist
        route_dist = self.route_direction_bearing_dist
        env_dist_weighted = self.env_bearing_dist_weighted

        # Circular cross-correlation to find the strongest and closest correlation
        strongest_correlation, closest_strongest_correlation = (
            alignment.find_optimal_correlation(route_dist, env_dist_weighted)
        )

        # Cosine similarity
        cosine_similarity_weighted = alignment.get_cosine_similarity_alignment(
            route_dist, env_dist_weighted
        )

        # Wasserstein distance or Earth Mover's Distance
        wasserstein_distance = alignment.get_EMD_alignment(
            route_dist, env_dist_weighted
        )

        peak_alignment = alignment.find_peaks_alignment(
            route_dist, env_dist_weighted
        )
        route_dist = route_dist / np.sum(route_dist)
        env_dist = env_dist / np.sum(env_dist)
        env_dist_weighted = env_dist_weighted / np.sum(env_dist_weighted)

        comparison_dict = {
            # od pair values
            "id": f"{self.origin_node}-{self.destination_node}",
            "city_name": self.city_name,
            "origin_node": self.origin_node,
            "origin_point": self.origin_point,
            "destination_node": self.destination_node,
            "destination_point": self.destination_point,
            "od_distance": self.od_distance,
            "od_cardinal_direction": self.cardinal_direction,
            # Difference values
            "shortest_simplest_hausdorff_distance": self.shortest_path.route_linestring.hausdorff_distance(self.simplest_path.route_linestring),
            # Alignment values
            "zero_lag_strongest": strongest_correlation["zero_lag"],
            "closest_strongest_lag": closest_strongest_correlation["lag"],
            "closest_strongest_correlation": closest_strongest_correlation["strength"],
            "closest_strongest_crosscorr": closest_strongest_correlation[
                "cross_correlation"
            ],
            "strongest_correlation_lag": strongest_correlation["lag"],
            "strongest_correlation": strongest_correlation["strength"],
            "strongest_crosscorr": strongest_correlation["cross_correlation"],
            "cosine_distance": closest_strongest_correlation["cosine_distance"],
            "euclidean_distance": closest_strongest_correlation["euclidean_distance"],
            "shifted_cosine_distance": closest_strongest_correlation[
                "shifted_cosine_distance"
            ],
            "shifted_euclidean_distance": closest_strongest_correlation[
                "shifted_euclidean_distance"
            ],
            "wasserstein_distance": wasserstein_distance,
            "_cosine_similarity": cosine_similarity_weighted,
            # Street orientation values
            "orientation_entropy": self.environment_orientation_entropy,
            "orientation_entropy_weighted": self.environment_orientation_entropy_weighted,
            "environment_orientation_order": self.order_weighted,
            "route_bearings_distribution": route_dist.tolist(),  # remove this
            "route_bearings": [
                str(self.shape_dict["fwd_bearing"]),
                str(self.shape_dict["bwd_bearing"]),
            ],
            "environment_bearings_distribution": env_dist.tolist(),
            "environment_bearings_distribution_weighted": env_dist_weighted.tolist(),
            # od pair shape values
            "bbox": self.bbox,
            "diamond": self.polygon,
            "area": self.area,
            # subgraph/environment values
            "edge_count": self.subgraph_stats["m"],
            "node_count": self.subgraph_stats["n"],
            "street_segment_count": self.subgraph_stats["street_segment_count"],
            "streets_per_node_avg": self.subgraph_stats["streets_per_node_avg"],
            "streets_per_node_counts": self.subgraph_stats["streets_per_node_counts"],
            "intersection_density_km": self.subgraph_stats["intersection_density_km"],
            "intersection_count": self.subgraph_stats["intersection_count"],
            "k_avg": self.subgraph_stats["k_avg"],
            "street_length_total": self.subgraph_stats["street_length_total"],
            "street_length_avg": self.subgraph_stats["street_length_avg"],
            "circuity_avg": self.subgraph_stats["circuity_avg"],
            "node_density_km": self.subgraph_stats["node_density_km"],
            # Peak alignment values
            "peak_route_main": peak_alignment["route_main_peak"],
            "peak_closest_env": peak_alignment["closest_env_peak"],
            "peak_strongest_env": peak_alignment["strongest_env_peak"],
            "peak_closest_env_value": peak_alignment["closest_env_peak_value"],
            "peak_strongest_env_value": peak_alignment["strongest_env_peak_value"],
            "peak_distance_to_closest": peak_alignment["distance_to_closest"],
            "peak_distance_to_strongest": peak_alignment["distance_to_strongest"],
        }
        # print("now adding the route dicts")
        shortest_path_dict = {
            f"shortest_{key}": value for key, value in vars(self.shortest_path).items()
        }
        simplest_path_dict = {
            f"simplest_{key}": value for key, value in vars(self.simplest_path).items()
        }

        comparison_dict.update(shortest_path_dict)
        comparison_dict.update(simplest_path_dict)
        return comparison_dict

    def get_comparison_dict_single_path(self):
        env_dist = self.env_bearing_dist
        route_dist = self.route_direction_bearing_dist
        env_dist_weighted = self.env_bearing_dist_weighted

        # Circular cross-correlation to find the strongest and closest correlation
        strongest_correlation, closest_strongest_correlation = (
            alignment.find_optimal_correlation(route_dist, env_dist_weighted)
        )

        # Cosine similarity
        cosine_similarity_weighted = alignment.get_cosine_similarity_alignment(
            route_dist, env_dist_weighted
        )

        # Wasserstein distance or Earth Mover's Distance
        wasserstein_distance = alignment.get_EMD_alignment(
            route_dist, env_dist_weighted
        )
        peak_alignment = alignment.find_peaks_alignment(
            route_dist, env_dist_weighted
        )


        route_dist = route_dist / np.sum(route_dist)
        env_dist = env_dist / np.sum(env_dist)
        env_dist_weighted = env_dist_weighted / np.sum(env_dist_weighted)

        comparison_dict = {
            # od pair values
            "id": f"{self.origin_node}-{self.destination_node}",
            "city_name": self.city_name,
            "origin_node": self.origin_node,
            "origin_point": self.origin_point,
            "destination_node": self.destination_node,
            "destination_point": self.destination_point,
            "od_distance": self.od_distance,
            "od_cardinal_direction": self.cardinal_direction,
            # Alignment values
            "closest_strongest_lag": closest_strongest_correlation["lag"],
            "closest_strongest_correlation": closest_strongest_correlation["strength"],
            "strongest_correlation_lag": strongest_correlation["lag"],
            "strongest_correlation": strongest_correlation["strength"],
            "cosine_distance": closest_strongest_correlation["cosine_distance"],
            "euclidean_distance": closest_strongest_correlation["euclidean_distance"],
            "shifted_cosine_distance": closest_strongest_correlation[
                "shifted_cosine_distance"
            ],
            "shifted_euclidean_distance": closest_strongest_correlation[
                "shifted_euclidean_distance"
            ],
            "wasserstein_distance": wasserstein_distance,
            "cosine_similarity": cosine_similarity_weighted,
            # Peak alignment values
            "peak_route_main": peak_alignment["route_main_peak"],
            "peak_closest_env": peak_alignment["closest_env_peak"],
            "peak_strongest_env": peak_alignment["strongest_env_peak"],
            "peak_closest_env_value": env_dist[peak_alignment["closest_env_peak"]],
            "peak_strongest_env_value": env_dist[peak_alignment["strongest_env_peak"]],
            "peak_distance_to_closest": peak_alignment["distance_to_closest"],
            "peak_distance_to_strongest": peak_alignment["distance_to_strongest"],
            # Street orientation values
            "orientation_entropy": self.environment_orientation_entropy,
            "orientation_entropy_weighted": self.environment_orientation_entropy_weighted,
            "environment_orientation_order": self.order_weighted,
            "route_bearings_distribution": route_dist.tolist(),
            "route_bearings": [
                str(self.shape_dict["fwd_bearing"]),
                str(self.shape_dict["bwd_bearing"]),
            ],
            "environment_bearings_distribution": env_dist.tolist(),
            "environment_bearings_distribution_weighted": env_dist_weighted.tolist(),
            # od pair shape values
            "bbox": self.bbox,
            "diamond": self.polygon,
            "area": self.area,
            # subgraph/environment values
            "edge_count": self.subgraph_stats["m"],
            "node_count": self.subgraph_stats["n"],
            "street_segment_count": self.subgraph_stats["street_segment_count"],
            "streets_per_node_avg": self.subgraph_stats["streets_per_node_avg"],
            "streets_per_node_counts": self.subgraph_stats["streets_per_node_counts"],
            "intersection_density_km": self.subgraph_stats["intersection_density_km"],
            "intersection_count": self.subgraph_stats["intersection_count"],
            "k_avg": self.subgraph_stats["k_avg"],
            "street_length_total": self.subgraph_stats["street_length_total"],
            "street_length_avg": self.subgraph_stats["street_length_avg"],
            "circuity_avg": self.subgraph_stats["circuity_avg"],
            "node_density_km": self.subgraph_stats["node_density_km"],

        }
        path_dict = {f"route_{key}": value for key, value in vars(self.path).items()}

        comparison_dict.update(path_dict)
        return comparison_dict
