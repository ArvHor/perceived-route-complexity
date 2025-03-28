from collections import defaultdict
import pickle
import random
from typing import List, Optional
import logging
import numpy as np
import osmnx as ox
import pandas as pd
import weighting_algorithms as wa
from od_pair import od_pair
import networkx as nx
import ast

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s',filename='app.log', filemode='w')

class origin_graph:
    def __init__(self, origin_point:tuple, distance_from_point:int,city_name:str,
                  edge_attr_diff: Optional[str] = None,network_type: Optional[str] = 'drive',
                  simplify: Optional[bool] = False,remove_parallel: Optional[bool] = False):
        
        logging.info(f"Beginning creation of graph: {city_name} origin point {origin_point}")
        self.origin_point = origin_point
        self.city_name = city_name
        self.distance_from_point = distance_from_point
        self.edge_attr_diff = edge_attr_diff
        self.network_type = network_type
        self.edge_weights = []
        self.node_attributes = []
        self.removed_parallel_edges = []
        self.removed_inf_edges = []
        self.od_pairs = []

        self.graph = self.create_graph(simplify=simplify)
        if remove_parallel:
            self.graph = self.remove_parallel_edges()

        self.start_node = self.find_start_node()
        self.bbox_coords = self.calculate_graph_bounding_box()
        self.graph.graph['origin_point'] = self.origin_point
        self.graph.graph['distance_from_point'] = self.distance_from_point
        self.graph.graph['network_type'] = self.network_type
        self.graph.graph['edge_attr_diff'] = self.edge_attr_diff
        self.graph.graph['simplify'] = simplify
        self.graph.graph['remove_parallel'] = remove_parallel
        self.graph.graph['city_name'] = self.city_name
        self.graph.graph['start_node'] = self.start_node
        self.graph.graph['bbox_coords'] = self.bbox_coords
        self.graph.graph['edge_weights'] = self.edge_weights
        self.graph.graph['node_attributes'] = self.node_attributes
        logging.info(f"Created graph for city: {city_name}, start_node: {self.start_node}")

    @classmethod
    def from_graphml(cls, graphml_path: str):
        edge_data_types = {
            "length": float,
            "decision_complexity": float,
            "deviation_from_prototypical": float,
            "instruction_equivalent": float,
            "node_degree": float
        }
        
        instance = cls.__new__(cls)
        instance.graph = ox.load_graphml(graphml_path, edge_dtypes=edge_data_types)
        instance.graph_path = graphml_path
        
        # Define attributes with their expected types
        attr_types = {
            'origin_point': tuple,  # e.g., (lat, lon)
            'distance_from_point': float,
            'network_type': str,
            'simplify': bool,
            'remove_parallel': bool,
            'city_name': str,
            'start_node': int,
            'bbox_coords': tuple,  # e.g., (min_lat, max_lat, min_lon, max_lon)
            'edge_weights': list,
            'node_attributes': list
        }
        
        graph_attrs = instance.graph.graph
        
        # Check and convert attributes
        for attr, expected_type in attr_types.items():
            if attr not in graph_attrs:
                raise ValueError(f"Missing required graph attribute: {attr}")
                
            value = graph_attrs[attr]
            
            # Convert string representation to actual type
            if isinstance(value, str):
                if expected_type in (dict, list, tuple):
                    try:
                        value = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        raise ValueError(f"Could not parse {attr} as {expected_type}")
                elif expected_type == bool:
                    value = value.lower() == 'true'
                elif expected_type == float:
                    value = float(value)
                elif expected_type == int:
                    value = int(value)
                    
            # Verify type
            if not isinstance(value, expected_type):
                raise TypeError(f"Attribute {attr} should be {expected_type}, got {type(value)}")
                
            setattr(instance, attr, value)
        logging.info(f"loaded graph for city: {instance.city_name}, start_node: {instance.start_node}")
        return instance

        
    def create_graph(self, simplify=False):
        """
        Create a graph from the center point and distance.

        Output:
        - Returns a graph object created from the center point and distance.
        """
        # First, download the graph from OSM via osmnx using the specified network type
        if self.network_type not in ['walk', 'bike', 'drive', 'drive_service', 'all', 'all_private']:
            G = ox.graph_from_point(self.origin_point, dist=self.distance_from_point, custom_filter=self.network_type, simplify=False, truncate_by_edge=True)
            print('Using custom filter to create graph')        
        else:
            G = ox.graph_from_point(self.origin_point, dist=self.distance_from_point, network_type=self.network_type, simplify=False, truncate_by_edge=True)

        G = ox.distance.add_edge_lengths(G)
        self.edge_weights.append("length")
        G.graph['length_added'] = "True"


        # Then simplify the graph, retaining only the unique osmid if specified
        if self.edge_attr_diff == 'osmid' and simplify == True:
            G = ox.simplification.simplify_graph(G,edge_attrs_differ=['osmid'])
            print('Simplifying graph retaining unique osmid')
        else:
            G = ox.simplify_graph(G)

        G = ox.bearing.add_edge_bearings(G)
        self.node_attributes.append("bearing")
        return G

    
    def calculate_graph_bounding_box(self):
        """
        Calculate the bounding box of the graph.

        Output:
        - Returns the bounding box of the graph.
        """
        return ox.convert.graph_to_gdfs(self.graph, nodes=False).unary_union.bounds

    def find_start_node(self):
        """
        Find the nearest with node outgoing edges that is closest to the center point 

        Output:
        - Returns the start node of the graph.
        """
        nearest_node = ox.distance.nearest_nodes(self.graph, self.origin_point[1], self.origin_point[0])
        temp_graph = self.graph.copy()
        while self.graph.out_degree(nearest_node) == 0:
            temp_graph.remove_node(nearest_node)
            nearest_node = ox.distance.nearest_nodes(temp_graph, self.origin_point[1], self.origin_point[0])
        return nearest_node

    def remove_parallel_edges(self):
        """
        Remove parallel edges from the graph, keeping only the shortest one
        and ensuring the remaining edge has key 0.

        Output:
        - Modifies the graph by removing parallel edges.
        """
        G = self.graph
        edges_to_remove = []
        edges_to_add = {}  # (u,v) -> edge_data
        
        # Identify parallel edges and find shortest ones
        for u, v, k in list(G.edges(keys=True)):
            if (u, v) not in edges_to_add:
                parallel_edges = [(u, v, key) for key in G[u][v]]
                if len(parallel_edges) > 1:
                    # Find the shortest edge and store its data
                    shortest_edge = min(parallel_edges, key=lambda edge: G.edges[edge]['length'])
                    edges_to_add[(u, v)] = G.edges[shortest_edge].copy()
                    # Mark all edges between u,v for removal
                    edges_to_remove.extend(parallel_edges)
        
        # Remove all parallel edges
        G.remove_edges_from(edges_to_remove)
        
        # Add back shortest edges with key 0
        for (u, v), edge_data in edges_to_add.items():
            G.add_edge(u, v, key=0, **edge_data)
        
        return G                                        
    
    def remove_infinite_edges(self):
        G = self.graph
        edges_to_remove = []
        for (u, v, k, data) in G.edges(keys=True,data=True):
            if data['decision_complexity'] == float('inf'):
                edges_to_remove.append((u, v, k))
        G.remove_edges_from(edges_to_remove)
        self.graph = G
        self.removed_inf_edges = edges_to_remove

    def create_orientation_plot(self,filepath=None):
        """
        Create an orientation plot of the graph.

        Parameters:
        - filepath: str, path to save the orientation plot.

        Output:
        - Saves the orientation plot to the specified filepath.
        """
        ox.plot_graph_orientation(self.graph, bbox=self.bbox_coords, save=True, filepath=filepath)
    
    def add_simplest_paths_from_origin(self):
        if "decision_complexity" in self.edge_weights:
            logging.info(f"Simplest paths from origin already added {self.city_name}")
            return
        else:
            try:
                self.graph = wa.simplest_path_from_source(G=self.graph,start_node=self.start_node)
                self.remove_infinite_edges()
                self.edge_weights.append("decision_complexity")
                self.graph.graph['edge_weights'] = self.edge_weights
            except Exception as e:
                logging.info(f"Error finding simplest paths for {self.city_name}: {e}")


    def add_weights(self,weightstrings:List[str]):
        

        if  "deviation_from_prototypical" in weightstrings:
            if "deviation_from_prototypical" in self.edge_weights:
                logging.info(f"Deviation from prototypical already calculated in {self.city_name}")
            else:
                self.graph, self.max_deviation_from_prototypical = wa.add_deviation_from_prototypical_weights(G=self.graph)
                self.edge_weights.append("deviation_from_prototypical")

        if "instruction_equivalent" in weightstrings:
            self.graph, self.max_instruction_equivalent = wa.add_instruction_equivalent_weights(G=self.graph)
            self.edge_weights.append("instruction_equivalent")

        if "node_degree" in weightstrings:
            self.graph, self.max_node_degree = wa.add_node_degree_weights(G=self.graph)
            self.edge_weights.append("node_degree")

        self.graph.graph['edge_weights'] = self.edge_weights

    def save_graph(self, filepath):
        try:
            ox.save_graphml(self.graph, filepath)
            logging.info(f"Successfully saved to {filepath}")
        except Exception as e:
            logging.info(f"error {e} saving to {filepath}")

    def add_node_elevation(self,api_key=None):
        self.graph = ox.elevation.add_node_elevations_google(self.graph, api_key=api_key,pause=0.1)
        self.node_attributes.append("elevation")
        self.graph.graph['node_attributes'] = self.node_attributes

    def save_pickle(self, filepath):
        """
        Save the routing_graph object to a pickle file.

        Parameters:
        - filepath: str, path to save the pickle file.

        Output:
        - Saves the routing_graph object to the specified filepath.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def find_destinations(self,min_radius,max_radius,sample=None):

        possible_destinations = set()
        for node in self.graph.nodes:
            if node != self.start_node:
                unprojected_distance = ox.distance.great_circle(self.graph.nodes[self.start_node]['y'], self.graph.nodes[self.start_node]['x'], self.graph.nodes[node]['y'], self.graph.nodes[node]['x'])
                if unprojected_distance >= min_radius and unprojected_distance <= max_radius:
                    if nx.has_path(self.graph,self.start_node,node):
                        possible_destinations.add(node)
                       
        if sample:
            return self.sample_destinations(possible_destinations,n=sample)
        else:
            return possible_destinations

    def sample_destinations(self,destinations,n=144):
        destination_bearing_tuples = []

        # 1. Calculate bearings and create tuples
        for destination in destinations:
            bearing = ox.bearing.calculate_bearing(
                self.graph.nodes[self.start_node]['y'],
                self.graph.nodes[self.start_node]['x'],
                self.graph.nodes[destination]['y'],
                self.graph.nodes[destination]['x'],
            )
            destination_bearing_tuples.append((destination, bearing))
        
        # 2. Create a histogram of bearings with edge effect mitigation
        bearings = [t[1] for t in destination_bearing_tuples]
        num_bins = 36
        num_split_bins = num_bins * 2
        split_bin_edges = np.arange(num_split_bins + 1) * 360 / num_split_bins

        split_bin_counts, split_bin_edges = np.histogram(bearings, bins=split_bin_edges)
        split_bin_counts = np.roll(split_bin_counts, 1)


         # 3. Create a dictionary to store nodes in each bin
        bin_nodes = defaultdict(list)
        for dest, bearing in destination_bearing_tuples:
            bin_index = int(bearing // 10) % num_bins # Use modulo to wrap around for edge cases
            bin_nodes[bin_index].append(dest)

        # 4. Retrieve a sample of nodes from each bin
        sampled_nodes = set()
        max_empty_in_row = 18
        while len(sampled_nodes) < n:
            start_bin = random.randint(0, num_bins - 1)
            empty_in_row = 0
            for i in range(num_bins):
                current_bin = (start_bin + i) % num_bins
                if len(sampled_nodes) < n:
                    new_nodes = set(bin_nodes[current_bin]) - sampled_nodes
                    if new_nodes:
                        empty_in_row = 0
                        sampled_nodes.add(np.random.choice(list(new_nodes)))
                    else:
                        empty_in_row += 1
                        if empty_in_row >= max_empty_in_row:
                            break
                else:
                    break
            if empty_in_row >= max_empty_in_row:
                break
        return sampled_nodes


        
    def create_od_pairs(self,min_radius=3000,max_radius=3500,sample_size=144):
        destinations = self.find_destinations(min_radius=min_radius,max_radius=max_radius,sample=sample_size)
        od_pairs = []
        for destination in destinations:
            od_p = od_pair(G=self.graph,origin=self.start_node,destination=destination)
            od_pairs.append(od_p)
        self.od_pairs = od_pairs

    def get_od_pair_data(self):
        od_pair_data = []
        for od_pair in self.od_pairs:
            od_pair_dict = od_pair.get_comparison_dict()
            od_pair_dict["graph_path"] = self.graph_path
            od_pair_data.append(od_pair.get_comparison_dict())
        od_pair_data = pd.DataFrame(od_pair_data)
        return od_pair_data
    
    def ensure_data_types(self):
        for u,v, data in self.graph.edges(data=True):
            for key in data:
                if key in ['length','decision_complexity','deviation_from_prototypical','instruction_equivalent','node_degree']:
                    data[key] = float(data[key])
                elif key in ['turn_complexity']:
                    data[key] = str(data[key])
                else:
                    continue
    @staticmethod
    def load_pickle(filepath):
        """
        Load a routing_graph object from a pickle file.

        Parameters:
        - filepath: str, path to the pickle file.

        Output:
        - Returns the loaded routing_graph object.
        """
        with open(filepath, 'rb') as f:
            routing_graph = pickle.load(f)
        return routing_graph
       
    


