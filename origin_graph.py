from collections import defaultdict
import pickle
import random
from typing import List, Optional, Set, Dict
from dataclasses import dataclass
import logging
import numpy as np
import osmnx as ox
import pandas as pd
import weighting_algorithms as wa
from od_pair import od_pair
import networkx as nx
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s',filename='app.log', filemode='w')

class origin_graph:
    def __init__(self, origin_point:tuple, distance_from_point:int,origin_info:Dict,
                  edge_attr_diff: Optional[str] = None,network_type: Optional[str] = 'drive',
                  load_graphml: Optional[str] = None, remove_parallel: Optional[bool] = False,
                  ):
        self.origin_point = origin_point
        # {"city_name":None,"origin_name":None,"country_name":None,"region_name":None},
        self.origin_info = origin_info
        self.city_name = origin_info["city_name"]
        self.distance_from_point = distance_from_point
        self.edge_attr_diff = edge_attr_diff
        self.network_type = network_type
        self.weights_list = ["length"]
        self.removed_parallel_edges = []
        self.removed_inf_edges = []
        self.od_pairs = []
        if load_graphml:
            logging.info(f'Loading graph, {load_graphml}')
            self.graph = ox.load_graphml(load_graphml)
            logging.info(f'Loaded graph, {load_graphml}, with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges')
            self.graph_path = load_graphml
            #logging.info(f'Loaded graph, {load_graphml}')
        else:
            logging.info(f'Creating graph, {self.origin_info} with distance {self.distance_from_point} and network type {self.network_type}')
            self.graph = self.create_graph()
            logging.info(f'Created graph, {self.origin_info} with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges')


        if remove_parallel:
            self.graph = self.remove_parallel_edges()
            logging.info(f'Removed parallel edges from {self.origin_info["city_name"]}')
        else:
            logging.info(f'Not removing parallel edges from {self.origin_info["city_name"]}')
        self.start_node = self.find_start_node()
        self.bbox_coords = self.calculate_graph_bounding_box()

        
    def create_graph(self):
        """
        Create a graph from the center point and distance.

        Output:
        - Returns a graph object created from the center point and distance.
        """
        if self.network_type not in ['walk', 'bike', 'drive', 'drive_service', 'all', 'all_private']:
            G = ox.graph_from_point(self.origin_point, dist=self.distance_from_point, custom_filter=self.network_type, simplify=False, truncate_by_edge=True)
            print('Using custom filter to create graph')        
        else:
            G = ox.graph_from_point(self.origin_point, dist=self.distance_from_point, network_type=self.network_type, simplify=False, truncate_by_edge=True)

        if self.edge_attr_diff == 'osmid':
            G = ox.simplification.simplify_graph(G,edge_attrs_differ=['osmid'])
            print('Simplifying graph retaining unique osmid')
        else:
            G = ox.simplify_graph(G)

        G = ox.bearing.add_edge_bearings(G)
        G = ox.distance.add_edge_lengths(G)
        G.graph['city_name'] = self.city_name
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
        Remove parallel edges from the graph, keeping only the shortest one.

        Output:
        - Modifies the graph by removing parallel edges.
        """
        G = self.graph
        edges_to_remove = []
        for u, v, k in G.edges(keys=True):
            parallel_edges = [(u, v, key) for key in G[u][v]]
            if len(parallel_edges) > 1:
                shortest_edge = min(parallel_edges, key=lambda edge: G.edges[edge]['length'])
                for edge in parallel_edges:
                    if edge != shortest_edge:
                        edges_to_remove.append(edge)
        self.removed_parallel_edges = edges_to_remove
        G.remove_edges_from(edges_to_remove)
        return G                                                   
    
    def remove_infinite_edges(self):
        #logging.info(f'Removing infinite edges from {self.origin_info["city_name"]}')
        G = self.graph
        edges_to_remove = []
        for (u, v, k, data) in G.edges(keys=True,data=True):
            #logging.info(f'Checking edge {u} to {v} with decision complexity {data["decision_complexity"]}')
            if data['decision_complexity'] == float('inf'):
                edges_to_remove.append((u, v, k))
        G.remove_edges_from(edges_to_remove)
        print(f'Removed {len(edges_to_remove)} edges with infinite decision complexity')
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
        
    def add_weights(self,weightstring):
        
        if weightstring == "decision_complexity":
            #logging.info(f'Adding decision complexity weights to {self.origin_info["city_name"]}')
            
            try:
                self.graph = wa.simplest_path_weight_algorithm_optimized(G=self.graph,start_node=self.start_node)
            except:
                logging.error(f"Error adding decision complexity weights to {self.origin_info['city_name']}",exc_info=True)
            
            try:
                self.remove_infinite_edges()
            except:
                logging.error(f"Error removing infinite edges from {self.origin_info['city_name']}",exc_info=True)
            self.weights_list.append("decision_complexity")
            
        elif weightstring == "deviation_from_prototypical":
            self.graph, self.max_deviation_from_prototypical = wa.add_deviation_from_prototypical_weights(G=self.graph)
            self.weights_list.append("deviation_from_prototypical")
        elif weightstring == "instruction_equivalent":
            self.graph, self.max_instruction_equivalent = wa.add_instruction_equivalent_weights(G=self.graph)
            self.weights_list.append("instruction_equivalent")
        elif weightstring == "node_degree":
            self.graph, self.max_node_degree = wa.add_node_degree_weights(G=self.graph)
            self.weights_list.append("node_degree")
        elif weightstring == "complexity_combined":
            self.graph, self.max_decision_complexity = wa.simplest_path_weight_algorithm(G=self.graph, start_node=self.start_node)
            self.graph, self.max_deviation_from_prototypical = wa.add_deviation_from_prototypical_weights(G=self.graph)
            self.graph, self.max_instruction_equivalent = wa.add_instruction_equivalent_weights(G=self.graph)
            self.graph, self.max_node_degree = wa.add_node_degree_weights(G=self.graph)
            
            max_local_combined = 0
            for u,v,data in self.graph.edges(data=True):
                w1 = (data["decision_complexity"]/self.max_decision_complexity)
                w2 = (data["deviation_from_prototypical"]/self.max_deviation_from_prototypical)
                w3 = (data["instruction_equivalent"]/self.max_instruction_equivalent)
                w4 = (data["node_degree"]/self.max_node_degree)
                
                combined = (w1 + w2 + w3 + w4) / 4

                data["complexity combined"] = combined
            
            self.weights_list.append("all_local_combined")

    def save_graph(self, filepath=None):
        if filepath is not None:
            self.graph_path = filepath
        try:
            ox.save_graphml(self.graph, self.graph_path)
            logging.info(f'Saved graph to {self.graph_path}')
        except:
            logging.error(f'Error saving graph to {self.graph_path}',exc_info=True)

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
                        logging.info(f'In edges of destination {node}: {self.graph.in_edges(node)}')
        logging.info(f'Found {len(possible_destinations)} possible destinations in {self.origin_info["city_name"]}')

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
        bin_counts = split_bin_counts[::2] + split_bin_counts[1::2]
        bin_centers = split_bin_edges[range(0, num_split_bins, 2)] # Corrected to step by 2


         # 3. Create a dictionary to store nodes in each bin
        bin_nodes = defaultdict(list)
        for dest, bearing in destination_bearing_tuples:
            bin_index = int(bearing // 10) % num_bins # Use modulo to wrap around for edge cases
            bin_nodes[bin_index].append(dest)

        
        # 4. Retrieve a sample of nodes from each bin
        sampled_nodes = set()
        nodes_per_bin = n // num_bins
        lap = 1
        max_laps = 10
        min_nodes_per_lap = 18
        while len(sampled_nodes) < n and lap < max_laps:
            start_bin = random.randint(0, num_bins - 1)
            #logging.info(f"lap {lap} beginning at bin {start_bin}")
            lap += 1
            nodes_sampled_this_lap = 0

            for i in range(num_bins):
                current_bin = (start_bin + i) % num_bins
                if len(sampled_nodes) < n:
                    new_nodes = set(bin_nodes[current_bin]) - sampled_nodes
                    if new_nodes:
                        sampled_nodes.add(np.random.choice(list(new_nodes)))
                        nodes_sampled_this_lap += 1
                else:
                    break
            if nodes_sampled_this_lap < min_nodes_per_lap:
                break
            
        return sampled_nodes


        
    def create_od_pairs(self,min_radius=3000,max_radius=3500):
        destinations = self.find_destinations(min_radius=min_radius,max_radius=max_radius,sample=144)
        od_pairs = []
        for destination in destinations:
            od_p = od_pair(G=self.graph,origin=self.start_node,destination=destination)
            od_pairs.append(od_p)
        logging.info(f'Added {len(od_pairs)} od pairs to {self.origin_info["city_name"]}')
        self.od_pairs = od_pairs
        #return od_pairs

    def get_od_pair_data(self):
        od_pair_data = []
        for od_pair in self.od_pairs:
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
       
    


