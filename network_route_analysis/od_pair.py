import math
import numpy as np
from route import route
import osmnx as ox
import geo_util
import logging
import hashlib
import alignment as alignment
import network_analysis


logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s', filename='../app.log', filemode='w')

class od_pair:
    def __init__(self,G, origin, destination):
        # Set the basic attributes
        self.graph = G
        self.origin_node = origin
        self.destination_node = destination
        self.origin_point = (self.graph.nodes[self.origin_node]['y'], self.graph.nodes[self.origin_node]['x'])
        self.destination_point = (self.graph.nodes[self.destination_node]['y'], self.graph.nodes[self.destination_node]['x'])

        # Calculate the great circle distance between the origin and destination
        self.od_distance = float(ox.distance.great_circle(lat1=self.origin_point[0], lon1=self.origin_point[1], lat2=self.destination_point[0], lon2=self.destination_point[1]))

        # Find the simplest and shortest route from the origin to the destination
        self.shortest_path = route(self.graph,origin=self.origin_node,destination=self.destination_node,weighstring='length')
        self.simplest_path = route(self.graph,origin=self.origin_node,destination=self.destination_node,weighstring='decision_complexity')

        # Get geometric properties of the origin and destination

        self.shape_dict = geo_utilities.get_od_pair_polygon(self.origin_point, self.destination_point)
        self.polygon = self.shape_dict["polygon"] # Square origin and destination as the diagonal of a square
        self.bbox = self.shape_dict["bbox"] # Bounding box as `(left, bottom, right, top)`.
        self.bbox_polygon = self.shape_dict["bbox_polygon"]


        self.map_bbox = self.shortest_path.map_bbox

        self.subgraph = self.get_subgraph(bbox=False)
        self.undirected_subgraph = ox.convert.to_undirected(self.subgraph)
        self.area = geo_utilities.calculate_area_with_utm(self.polygon)
        self.subgraph_stats = ox.stats.basic_stats(self.subgraph, area=self.area)


        logging.error(f"Creating od_pair for graph {self.graph.graph['city_name']} with n subgraph edges: {len(self.subgraph.edges)}, {len(self.undirected_subgraph.edges)}")
        self.env_bearing_dist_weighted, _ = ox.bearing._bearings_distribution(G=self.undirected_subgraph , num_bins=36,min_length=10, weight="length")
        self.env_bearing_dist, _ = ox.bearing._bearings_distribution(G=self.undirected_subgraph , num_bins=36, min_length=10,weight=None)

        self.route_direction_bearing_dist = self.get_route_direction_bearing_dist()

        self.environment_orientation_entropy_weighted = ox.bearing.orientation_entropy(self.undirected_subgraph , num_bins=36, weight="length")
        self.environment_orientation_entropy = ox.bearing.orientation_entropy(self.undirected_subgraph , num_bins=36)




        self.order_weighted = self.get_environment_orientation_order(self.environment_orientation_entropy_weighted)
        self.order = self.get_environment_orientation_order(self.environment_orientation_entropy)


        self.length_diff = self.simplest_path.length - self.shortest_path.length
        self.complexity_diff = int(self.simplest_path.complexity) - int(self.shortest_path.complexity)
        self.shortest_diff = self.shortest_path.length - self.od_distance


    @classmethod
    def from_route(cls, G, route_nodes,weightstring):
        # Create instance without calling __init__
        instance = cls.__new__(cls)

        instance.path = route.from_nodes(G,route_nodes,weightstring=weightstring)

        # Set the basic attributes
        instance.graph = G
        instance.origin_node = instance.path.origin_node
        instance.destination_node = instance.path.destination_node
        instance.origin_point = (instance.graph.nodes[instance.origin_node]['y'], instance.graph.nodes[instance.origin_node]['x'])
        instance.destination_point = (instance.graph.nodes[instance.destination_node]['y'], instance.graph.nodes[instance.destination_node]['x'])
        
        # Calculate OD distance
        instance.od_distance = float(ox.distance.great_circle(lat1=instance.origin_point[0], lon1=instance.origin_point[1], 
                                                            lat2=instance.destination_point[0], lon2=instance.destination_point[1]))

        instance.graph = network_analysis.add_origin_destination_betweenness_centrality(instance.graph, origin=instance.origin_node,
                                                                                        destination=instance.destination_node,
                                                                                        weightstring="length")

        # Generate geometry
        instance.shape_dict = geo_utilities.get_od_pair_polygon(instance.origin_point, instance.destination_point)
        instance.polygon = instance.shape_dict["polygon"]
        instance.bbox = instance.shape_dict["bbox"]
        instance.bbox_polygon = instance.shape_dict["bbox_polygon"]
        instance.map_bbox = instance.path.map_bbox


        instance.subgraph = instance.get_subgraph(bbox=True)
        instance.undirected_subgraph = ox.convert.to_undirected(instance.subgraph)
        instance.area = geo_utilities.calculate_area_with_utm(instance.polygon)
        instance.subgraph_stats = ox.stats.basic_stats(instance.subgraph, area=instance.area)

        instance.env_bearing_dist_weighted, _ = ox.bearing._bearings_distribution(G=instance.undirected_subgraph, num_bins=36,min_length=10, weight="length")
        instance.env_bearing_dist, _ = ox.bearing._bearings_distribution(G=instance.undirected_subgraph, num_bins=36,min_length=10, weight=None)

        instance.route_direction_bearing_dist = instance.get_route_direction_bearing_dist()

        instance.environment_orientation_entropy_weighted = ox.bearing.orientation_entropy(instance.undirected_subgraph, num_bins=36,weight="length")
        instance.environment_orientation_entropy = ox.bearing.orientation_entropy(instance.undirected_subgraph, num_bins=36)

        instance.order_weighted = instance.get_environment_orientation_order(instance.environment_orientation_entropy_weighted)
        instance.order = instance.get_environment_orientation_order(instance.environment_orientation_entropy)

        #instance.subgraph_stats = None
        
        # Handle elevation if available

        #instance.elevation_origin = instance.graph.nodes[instance.origin_node]['elevation']
        #instance.elevation_destination = instance.graph.nodes[instance.destination_node]['elevation']
        #instance.elevation_difference = instance.elevation_origin - instance.elevation_destination

        return instance

    def generate_identifier(self):
        identifier = str(self.origin_node) + '-' + str(self.destination_node)

        hash_object = hashlib.sha256(identifier.encode())
        hex_dig = hash_object.hexdigest()
        identifier = hex_dig
        return identifier 


    def get_comparison_dict(self):
        env_dist = self.env_bearing_dist
        route_dist = self.route_direction_bearing_dist
        env_dist_weighted = self.env_bearing_dist_weighted


        # Check for None distributions and print their contents if any
        if self.env_bearing_dist is None:
            logging.error(f"env_bearing_dist is None. Number of edges in subgraph: {len(self.subgraph.edges)}")
        if self.route_direction_bearing_dist is None:
            logging.error(f"route_direction_bearing_dist is None. Distribution: {self.route_direction_bearing_dist}, Number of edges in subgraph: {len(self.subgraph.edges)}")
        if self.env_bearing_dist_weighted is None:
            logging.error(f"env_bearing_dist_weighted is None. Distribution: {self.env_bearing_dist_weighted}, Number of edges in subgraph: {len(self.subgraph.edges)}")


        # Circular cross-correlation to find the strongest and closest correlation
        max_correlation, best_lag, best_score, best_score_lag = alignment.find_strongest_and_closest_correlation(route_dist, env_dist_weighted)

        # Cosine similarity
        cosine_similarity_weighted = alignment.get_cosine_similarity_alignment(route_dist, env_dist_weighted)

        # Wasserstein distance or Earth Mover's Distance
        wasserstein_distance = alignment.get_EMD_alignment(route_dist, env_dist_weighted)

        route_dist = route_dist / np.sum(route_dist)
        env_dist = env_dist / np.sum(env_dist)
        env_dist_weighted = env_dist_weighted / np.sum(env_dist_weighted)


        comparison_dict = {
            # od pair values
            'id': f"{self.origin_node}-{self.destination_node}",  
            "city_name": self.graph.graph['city_name'],
            'origin_node': self.origin_node,
            "origin_point": self.origin_point,
            'destination_node': self.destination_node,
            "destination_point": self.destination_point,
            'od_distance': self.od_distance,

            # Difference values
            "shortest_simplest_hausdorff_distance": self.shortest_path.route_geometry.hausdorff_distance(self.simplest_path.route_geometry),

            # Alignment values
            "closest_strongest_lag": best_score_lag,
            "closest_strongest_correlation": best_score,
            "strongest_correlation_lag": best_lag,
            'wasserstein_distance': wasserstein_distance,
            'cosine_similarity': cosine_similarity_weighted,

            # Street orientation values
            "orientation_entropy": self.environment_orientation_entropy,
            "orientation_entropy_weighted": self.environment_orientation_entropy_weighted,
            "environment_orientation_order_order": self.order_weighted,
            "route_bearings_distribution": route_dist.tolist(),
            "route_bearings": [str(self.shape_dict["fwd_bearing"]), str(self.shape_dict["bwd_bearing"])],
            "environment_bearings_distribution": env_dist.tolist(),
            "environment_bearings_distribution_weighted": env_dist_weighted.tolist(),

            # od pair shape values
            "bbox": self.bbox,
            "diamond": self.polygon,
            "area": self.area,

            # subgraph/environment values
            "edge_count": self.subgraph_stats['m'],
            "node_count": self.subgraph_stats['n'],
            "street_segment_count": self.subgraph_stats['street_segment_count'],
            "streets_per_node_avg": self.subgraph_stats['streets_per_node_avg'],
            "streets_per_node_counts": self.subgraph_stats['streets_per_node_counts'],
            "intersection_density_km": self.subgraph_stats['intersection_density_km'],
            "intersection_count": self.subgraph_stats['intersection_count'],
            "k_avg": self.subgraph_stats['k_avg'],
            "street_length_total": self.subgraph_stats['street_length_total'],
            "street_length_avg": self.subgraph_stats['street_length_avg'],
            "circuity_avg": self.subgraph_stats['circuity_avg'],
            "node_density_km": self.subgraph_stats['node_density_km'],
        }

        shortest_path_dict = vars(self.simplest_path)
        simplest_path_dict = vars(self.shortest_path)

        comparison_dict.update(shortest_path_dict)
        comparison_dict.update(simplest_path_dict)
        return comparison_dict

    def get_comparison_dict_single_path(self):
        env_dist = self.env_bearing_dist
        route_dist = self.route_direction_bearing_dist
        env_dist_weighted = self.env_bearing_dist_weighted

        # basic cross-correlation
        lag, max_correlation = alignment.get_crosscorrelation_alignment(route_dist, env_dist_weighted)

        # Circular cross-correlation to find the strongest and closest correlation
        max_correlation, best_lag, best_score, best_score_lag = alignment.find_strongest_and_closest_correlation(route_dist, env_dist_weighted)

        # Cosine similarity
        cosine_similarity_weighted = alignment.get_cosine_similarity_alignment(route_dist, env_dist_weighted)

        # Wasserstein distance or Earth Mover's Distance
        wasserstein_distance = alignment.get_EMD_alignment(route_dist, env_dist_weighted)

        route_dist = route_dist / np.sum(route_dist)
        env_dist = env_dist / np.sum(env_dist)
        env_dist_weighted = env_dist_weighted / np.sum(env_dist_weighted)

        comparison_dict = {
            # od pair values
            'id': f"{self.origin_node}-{self.destination_node}",  
            "city_name": self.graph.graph['city_name'],
            'origin_node': self.origin_node,
            "origin_point": self.origin_point,
           # 'origin_elevation': self.elevation_origin,
            'destination_node': self.destination_node,
            "destination_point": self.destination_point,
           # 'destination_elevation': self.elevation_destination,
           # 'elevation_difference': self.elevation_difference,
            'od_distance': self.od_distance,

            # Path values
            "path_nodes": self.path.nodes,
            'path_length': self.path.length,
            "path_circuity": self.path.length / self.od_distance,
            'path_complexity': self.path.complexity,
            'path_turn_labels': self.path.turn_types,
            'path_turn_count': self.path.turn_count,
            'path_turn_frequency': self.path.turn_frequency,
            "path_n_nodes": self.path.n_nodes,
            "simplest_path_avg_betweenness": self.path.avg_betweenness,
            "simplest_path_avg_od_betweenness": self.path.avg_od_betweenness,
            "path_deviation_from_prototypical": self.path.sum_deviation_from_prototypical,
            "path_instruction_equivalent": self.path.sum_instruction_equivalent,
            "path_node_degree": self.path.sum_node_degree,
            "path_geometry": self.path.route_geometry,

            # Path MAP values
            'path_map_intersection_count': self.path.map_intersection_count,
            'path_map_street_count': self.path.map_street_count,
            'path_map_road_length': self.path.map_road_length,

            # Path type
            'path_type': self.path.weightstring,  # Indicates whether this was a "length" or "decision_complexity" path

            # basic alignment values
            'basic_crosscorrelation_lag': lag,
            'basic_crosscorrelation_max_corr': max_correlation,
            'circular_crosscorrelation_lag': circ_lag_weighted,
            "circular_crosscorrelation_max_corr": max_circular_correlation_weighted, 
            'cosine_similarity': cosine_similarity_weighted,
            "cosine_similarity_at_best_lag": circular_cosine_similarity_weighted,
            'wasserstein_distance': wasserstein_distance,

            # Alignment values
            "closest_strongest_lag": best_score_lag,
            "closest_strongest_correlation": best_score,
            "strongest_correlation_lag": best_lag,

            # Street orientation values
            "orientation_entropy": self.environment_orientation_entropy,
            "orientation_entropy_weighted": self.environment_orientation_entropy_weighted,
            "environment_orientation_order": self.get_environment_orientation_order(self.environment_orientation_entropy),
            "route_bearings_distribution": route_dist.tolist(),
            "route_bearings": [str(self.shape_dict["fwd_bearing"]), str(self.shape_dict["bwd_bearing"])],
            "environment_bearings_distribution": env_dist.tolist(),
            "environment_bearings_distribution_weighted": env_dist_weighted.tolist(),

            # od pair shape values
            "bbox": self.bbox,
            "diamond": self.polygon,
            "area": self.area,

            # subgraph/environment values
            "edge_count": self.subgraph_stats['m'],
            "node_count": self.subgraph_stats['n'],
            "street_segment_count": self.subgraph_stats['street_segment_count'],
            "streets_per_node_avg": self.subgraph_stats['streets_per_node_avg'],
            "streets_per_node_counts": self.subgraph_stats['streets_per_node_counts'],
            "intersection_density_km": self.subgraph_stats['intersection_density_km'],
            "intersection_count": self.subgraph_stats['intersection_count'],
            "k_avg": self.subgraph_stats['k_avg'],
            "street_length_total": self.subgraph_stats['street_length_total'],
            "street_length_avg": self.subgraph_stats['street_length_avg'],
            "circuity_avg": self.subgraph_stats['circuity_avg'],
            "node_density_km": self.subgraph_stats['node_density_km'],
        }
        return comparison_dict