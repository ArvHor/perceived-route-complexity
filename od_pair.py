import math
import networkx as nx
import numpy as np
import pandas as pd
from route import route
import osmnx as ox
import geo_utilities
from scipy.stats import wasserstein_distance
import logging
from scipy.signal import correlate2d
from scipy.signal import find_peaks, correlate2d
import hashlib
from matplotlib.projections.polar import PolarAxes

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s',filename='app.log', filemode='w')

class od_pair:
    def __init__(self,G, origin, destination):
        self.graph = G
        self.origin_node = origin
        self.destination_node = destination
        self.origin_point = (self.graph.nodes[self.origin_node]['y'], self.graph.nodes[self.origin_node]['x'])
        self.destination_point = (self.graph.nodes[self.destination_node]['y'], self.graph.nodes[self.destination_node]['x'])
        self.od_distance = float(ox.distance.great_circle(lat1=self.origin_point[0], lon1=self.origin_point[1], lat2=self.destination_point[0], lon2=self.destination_point[1]))
        self.shortest_path = route(self.graph,origin=self.origin_node,destination=self.destination_node,weighstring='length')
        self.simplest_path = route(self.graph,origin=self.origin_node,destination=self.destination_node,weighstring='decision_complexity')
        self.shape_dict = geo_utilities.get_od_pair_polygon(self.origin_point, self.destination_point)
        self.polygon = self.shape_dict["polygon"]
        self.bbox = self.shape_dict["bbox"]
        self.length_diff = self.simplest_path.length - self.shortest_path.length
        self.complexity_diff = self.simplest_path.complexity - self.shortest_path.complexity
        self.shortest_diff = self.shortest_path.length - self.od_distance
        self.environment_bearing_dist = None
        self.route_direction_bearing_dist = None
        self.environment_orientation_entropy = None
        self.subgraph = None
        self.area = None
        self.od_pair_polygon_ug = None
        self.subgraph_stats = None


    def od_pair_polygon_subgraph(self):
        if self.subgraph is None:
            subgraph = ox.truncate.truncate_graph_polygon(G=self.graph, polygon=self.polygon, truncate_by_edge=True)
            self.area = geo_utilities.calculate_area_with_utm(self.polygon)
            self.subgraph_stats = ox.stats.basic_stats(subgraph, area=self.area)
            self.subgraph = subgraph
        else:
            subgraph = self.subgraph
        logging.info(f"Number of edges in the subgraph: {subgraph.number_of_edges()}")
        return subgraph
    
    def get_environment_bearing_dist(self):
        od_pair_polygon_g = self.od_pair_polygon_subgraph()
        od_pair_polygon_g = ox.convert.to_undirected(od_pair_polygon_g)
        od_pair_polygon_g = ox.bearing.add_edge_bearings(od_pair_polygon_g)

        env_bearing_distribution_weighted,_ = ox.bearing._bearings_distribution(G=od_pair_polygon_g,num_bins=36,min_length=10,weight="length")
        env_bearing_distribution,_ = ox.bearing._bearings_distribution(G=od_pair_polygon_g,num_bins=36,min_length=10,weight=None)
        if self.environment_orientation_entropy is None:
            self.od_pair_polygon_ug = od_pair_polygon_g
            self.environment_orientation_entropy_weighted = ox.bearing.orientation_entropy(od_pair_polygon_g,num_bins=36,weight="length",min_length=10)
            self.environment_orientation_entropy = ox.bearing.orientation_entropy(od_pair_polygon_g,num_bins=36,min_length=10)
        
        print(f"Environment bearing distribution: {env_bearing_distribution}, weighted: {env_bearing_distribution_weighted}")
        return env_bearing_distribution, env_bearing_distribution_weighted
    
    def create_orientation_plot(self,filepath):
        fig,ax = ox.plot_orientation(self.od_pair_polygon_g,filepath=filepath,weight="length",min_length=10,save=True,show=False)

        r_dist = self.get_route_direction_bearing_dist()

        self.plot_overlaid_distribution(ax, r_dist, num_bins=36, area=True)
        fig.savefig(filepath)

    def plot_overlaid_distribution(
        ax: PolarAxes,
        new_distribution: np.ndarray,
        num_bins: int,
        area: bool
    ) -> None:


        bin_centers = 360 / num_bins * np.arange(num_bins)
        positions = np.radians(bin_centers)
        width = 2 * np.pi / num_bins
        
        # Normalize the new distribution to calculate height/area
        new_bin_frequency = new_distribution / new_distribution.sum()
        new_radius = np.sqrt(new_bin_frequency) if area else new_bin_frequency

        ax.bar(
            positions,
            height=new_radius,
            width=width,
            align="center",
            bottom=0,
            zorder=3,  # Ensure red bars are on top
            color="red",
            edgecolor="k",
            linewidth=0.5,
            alpha=0.5,
        )
    def generate_identifier(self):
        identifier = str(self.origin_node) + '-' + str(self.destination_node)

        hash_object = hashlib.sha256(identifier.encode())
        hex_dig = hash_object.hexdigest()
        identifier = hex_dig
        return identifier 

    def get_route_direction_bearing_dist(self,with_perpendicular=False):
        fwd = self.shape_dict["fwd_bearing"]
        bwd = self.shape_dict["bwd_bearing"]

        if with_perpendicular:
            perpendicular_fwd_bearing = self.shape_dict["perpendicular_fwd_bearing"]
            perpendicular_bwd_bearing = self.shape_dict["perpendicular_bwd_bearing"]
            bearings = [fwd,bwd,perpendicular_fwd_bearing,perpendicular_bwd_bearing]
        else:
            bearings = [fwd,bwd]
        num_bins = 36
        num_split_bins = num_bins * 2
        split_bin_edges = np.arange(num_split_bins + 1) * 360 / num_split_bins

        split_bin_counts, split_bin_edges = np.histogram(bearings, bins=split_bin_edges)
        split_bin_counts = np.roll(split_bin_counts, 1)
        bin_counts = split_bin_counts[::2] + split_bin_counts[1::2]
        bin_centers = split_bin_edges[range(0, num_split_bins - 1, 2)]
        print(f"Route Bin counts: {bin_counts}, number of bins: {len(bin_counts)}")
        return bin_counts

    def get_environment_orientation_order(self,env_entropy,num_bins=36, ):
        max_nats = math.log(num_bins)
        min_nats = math.log(4)
        orientation_order = 1 - ((env_entropy - min_nats) / (max_nats - min_nats))**2
        return orientation_order
    
    def get_EMD_alignment(self, route_dist, env_dist):
        route_dist = route_dist / np.sum(route_dist)
        env_dist = env_dist / np.sum(env_dist)
        EMD_alignment = wasserstein_distance(env_dist, route_dist)
        return EMD_alignment
    
    
    def circular_shift(self,arr, shift):
        """Circularly shifts a 1D array."""
        return np.roll(arr, shift)
    
        
    def circular_cross_correlation(self,a, b):
        """Calculates the circular cross-correlation using FFT."""
        a = np.asarray(a)
        b = np.asarray(b)
        result = np.fft.ifft(np.fft.fft(a) * np.fft.fft(np.conj(b))[::-1]).real
        return np.roll(result, -(len(b) // 2))
    
    def find_circular_max_correlation_and_lag(self,a, b):
        """Finds the maximum correlation and lag in circular cross-correlation."""
        max_len = max(len(a), len(b))

        circ_cross_corr = self.circular_cross_correlation(a, b)
        lag = np.argmax(circ_cross_corr)
        if lag >= max_len // 2:
            lag -= max_len
        max_correlation = circ_cross_corr[lag] if lag >= 0 else circ_cross_corr[lag + max_len]
        return max_correlation, lag
    
    def find_strongest_and_closest_correlation(self,route_dist,env_dist):
        route_dist = route_dist / np.sum(route_dist)
        env_dist = env_dist / np.sum(env_dist)
        max_len = max(len(route_dist), len(env_dist))

        circ_cross_corr = self.circular_cross_correlation(route_dist, env_dist)

        normalized_circ_cross_corr = circ_cross_corr / np.max(circ_cross_corr)

        lag_range = np.arange(len(normalized_circ_cross_corr))

        scores = []
        max_lag = max_len//2
        for i, lag_index in enumerate(lag_range):
            lag = lag_index if lag_index < max_len // 2 else lag_index - max_len  # Adjust lag
            correlation_strength = normalized_circ_cross_corr[i]
            score = correlation_strength - (abs(lag) / max_lag)
            scores.append(score)

        # 1. Find the strongest correlation (and its lag)
        strongest_correlation_index = np.argmax(normalized_circ_cross_corr)
        max_correlation = normalized_circ_cross_corr[strongest_correlation_index]
        best_lag = lag_range[strongest_correlation_index]
        if best_lag >= max_len // 2:
            best_lag -= max_len

        # 2. Find the lag with the best combined score
        best_score_index = np.argmax(scores)
        best_score = scores[best_score_index]
        best_score_lag = lag_range[best_score_index]
        if best_score_lag >= max_len // 2:
            best_score_lag -= max_len

        return max_correlation, best_lag, best_score, best_score_lag
    
    def combined_alignment_score(correlation_strength, lag, max_lag):
        """
        Combines correlation strength and lag into a single score.

        Args:
            correlation_strength: Normalized correlation at best lag (e.g., from -1 to 1).
            lag: Lag at the best alignment.
            max_lag: Maximum possible lag (e.g., num_bins // 2 for circular data).

        Returns:
            A combined score.
        """
        abs_lag = abs(lag)
        score = correlation_strength / (abs_lag + 1)
        return score
    
    def get_circular_crosscorrelation_alignment(self,route_dist,env_dist):
        """
        Calculates alignment using circular cross-correlation and finds peak distances.
        """
        route_dist = route_dist / np.sum(route_dist)
        env_dist = env_dist / np.sum(env_dist)
        max_correlation, lag = self.find_circular_max_correlation_and_lag(env_dist, route_dist)

        # Normalize the correlation
        normalized_max_correlation = max_correlation / np.max(env_dist)

        # Align distributions
        aligned_route_dist = self.circular_shift(route_dist, lag)
        
        # Calculate the cosine similarity after aligning the distributions
        cosine_sim = np.dot(env_dist, aligned_route_dist) / (np.linalg.norm(env_dist) * np.linalg.norm(aligned_route_dist))


        return normalized_max_correlation, lag, cosine_sim

    def get_crosscorrelation_alignment(self, route_dist, env_dist):
        route_dist = route_dist / np.sum(route_dist)
        env_dist = env_dist / np.sum(env_dist)
        cross_correlation = np.correlate(env_dist, route_dist, mode='full')

        lag = np.argmax(cross_correlation) - (len(route_dist) - 1)
        max_correlation = cross_correlation[lag + (len(route_dist) - 1)]

        return cross_correlation, abs(lag), max_correlation
    
    def get_cosine_similarity_alignment(self, route_dist, env_dist):
        route_dist = route_dist / np.sum(route_dist)
        env_dist = env_dist / np.sum(env_dist)
        cosine_similarity = np.dot(env_dist, route_dist) / (np.linalg.norm(env_dist) * np.linalg.norm(route_dist))
        return cosine_similarity
    
    def get_comparison_dict(self):
        env_dist, env_dist_weighted = self.get_environment_bearing_dist()
        route_dist = self.get_route_direction_bearing_dist()
        logging.info(f"Route distribution: {route_dist}")
        logging.info(f"Environment distribution: {env_dist}")

        try:
            cross_correlation_dist, lag, max_correlation = self.get_crosscorrelation_alignment(route_dist, env_dist)
        except Exception as e:
            logging.error(f"Error in get_crosscorrelation_alignment: {e}")
            return

        try:
            cross_correlation_dist_weighted, weighted_lag, weighted_max_correlation = self.get_crosscorrelation_alignment(route_dist, env_dist_weighted)
        except Exception as e:
            logging.error(f"Error in get_crosscorrelation_alignment (weighted): {e}")
            return

        try:
            cosine_similarity = self.get_cosine_similarity_alignment(route_dist, env_dist)
        except Exception as e:
            logging.error(f"Error in get_cosine_similarity_alignment: {e}")
            return

        try:
            cosine_similarity_weighted = self.get_cosine_similarity_alignment(route_dist, env_dist_weighted)
        except Exception as e:
            logging.error(f"Error in get_cosine_similarity_alignment (weighted): {e}")
            return

        try:
            max_correlation, best_lag, best_score, best_score_lag = self.find_strongest_and_closest_correlation(route_dist, env_dist_weighted)	
        except Exception as e:
            logging.error(f"Error in get_cosine_similarity_alignment (weighted): {e}")
            return
        try:
            max_circular_correlation_weighted, circ_lag_weighted, circular_cosine_similarity_weighted = self.get_circular_crosscorrelation_alignment(route_dist, env_dist_weighted)
        except Exception as e:
            logging.error(f"Error in get_circular_crosscorrelation_alignment (weighted): {e}")
            return
        route_dist = route_dist / np.sum(route_dist)
        env_dist = env_dist / np.sum(env_dist)
        env_dist_weighted = env_dist_weighted / np.sum(env_dist_weighted)
        alignment = abs(best_score_lag) % 9
        comparison_dict = {
            # od pair values
            '0_id': f"{self.origin_node}-{self.destination_node}",  
            "0_city_name": self.graph.graph['city_name'],
            '0_origin_node': self.origin_node,
            "0_origin_point": self.origin_point,
            "0_destination_point": self.destination_point,
            '0_destination_node': self.destination_node,
            '0_od_distance': self.od_distance,
            # path values
            "1_simplest_path_nodes": self.simplest_path.nodes,
            "1_shortest_path_nodes": self.shortest_path.nodes,
            '1_simplest_path_length': self.simplest_path.length,
            '1_shortest_path_length': self.shortest_path.length,
            '1_simplest_path_complexity': self.simplest_path.complexity,
            '1_shortest_path_complexity': self.shortest_path.complexity,
            '1_simplest_path_turn_labels': self.simplest_path.edges,
            '1_shortest_path_turn_labels': self.simplest_path.edges,
            "1_simplest_path_n_nodes": self.simplest_path.n_nodes,
            "1_shortest_path_n_nodes": self.shortest_path.n_nodes,
            "1_simplest_path_deviation_from_prototypical": self.simplest_path.sum_deviation_from_prototypical,
            "1_shortest_path_deviation_from_prototypical": self.shortest_path.sum_deviation_from_prototypical,
            "1_simplest_path_instruction_equivalent": self.simplest_path.sum_instruction_equivalent,
            "1_shortest_path_instruction_equivalent": self.shortest_path.sum_instruction_equivalent,
            "1_simplest_path_node_degree": self.simplest_path.sum_node_degree,
            "1_shortest_path_node_degree": self.shortest_path.sum_node_degree,
            "1_simplest_path_geometry": self.simplest_path.route_geometry,
            "1_shortest_path_geometry": self.shortest_path.route_geometry,
            # basic alignment values
            '2_cross_correlation_dist': lag,
            '2_cosine_similarity': cosine_similarity,
            '2_EMD_alignment_weighted': self.get_EMD_alignment(route_dist, env_dist_weighted),
            # Alignment values
            "3_alignment": alignment,
            "3_A_strongest_correlation_lag": best_lag,
            "3_A_closest_strongest_lag": best_score_lag,
            "3_A_closest_strongest_score": best_score,
            '3_B_circular_crosscorrelation_dist_weighted': circ_lag_weighted,
            "3_B_max_circular_correlation_weighted": max_circular_correlation_weighted, 
            '3_c_cross_correlation_dist_weighted': weighted_lag,
            "3_c_max_correlation_weighted": weighted_max_correlation,
            '3_c_cosine_similarity_weighted': cosine_similarity_weighted,
            # Orientation values
            "4_orientation_entropy": self.environment_orientation_entropy,
            "4_orientation_entropy_weighted": self.environment_orientation_entropy_weighted,
            "4_environment_orientation_order": self.get_environment_orientation_order(self.environment_orientation_entropy),
            "4_route_bearings_distribution": str(route_dist),
            "4_route_bearings": [str(self.shape_dict["fwd_bearing"]), str(self.shape_dict["bwd_bearing"])],
            "4_environment_bearings_distribution": str(env_dist),
            "4_environment_bearings_distribution_weighted": str(env_dist_weighted),
            # od pair shape values
            "5_bbox": self.bbox,
            "5_diamond": self.polygon,
            "5_area": self.area,
            # subgraph or environment values
            "6_edge_count": self.subgraph_stats['m'],
            "6_node_count": self.subgraph_stats['n'],
            "6_street_segment_count": self.subgraph_stats['street_segment_count'],
            "6_streets_per_node_avg": self.subgraph_stats['streets_per_node_avg'],
            "6_streets_per_node_counts": self.subgraph_stats['streets_per_node_counts'],
            "6_intersection_density_km": self.subgraph_stats['intersection_density_km'],
            "6_intersection_count": self.subgraph_stats['intersection_count'],
            "6_k_avg": self.subgraph_stats['k_avg'],
            "6_street_length_total": self.subgraph_stats['street_length_total'],
            "6_street_length_avg": self.subgraph_stats['street_length_avg'],
            "6_circuity_avg": self.subgraph_stats['circuity_avg'],
            "6_node_density_km": self.subgraph_stats['node_density_km'],
        }
        return comparison_dict

    def to_dict(self):
        """Converts the OD pair data and its routes to a dictionary."""
        od_dict = {
            'od_pair_id': f"{self.origin_node}-{self.destination_node}",  # You might want a more robust ID
            'origin_node': self.origin_node,
            'destination_node': self.destination_node,
            'shortest_path': {
                'route_length': self.shortest_path.length,
                'dijkstra_weight': self.shortest_path.weightstring,
                'nodes': self.shortest_path.nodes,
                'edges': self.shortest_path.edges,
                'subgraph_edges': list(self.shortest_path.subgraph.edges)  # Convert subgraph to a serializable format
            },
            'simplest_path': {
                'route_length': self.simplest_path.length,
                'dijkstra_weight': self.simplest_path.weightstring,
                'nodes': self.simplest_path.nodes,
                'edges': self.simplest_path.edges,
                'subgraph_edges': list(self.simplest_path.subgraph.edges)  # Convert subgraph to a serializable format
            }
        }
        return od_dict