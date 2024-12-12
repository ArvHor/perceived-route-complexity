import ast
import gc
import math
import os
import time
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import osmnx as ox
import weighting_algorithms as sp
import pandas as pd
import routing as rt
import map_plotting as mp
import pickle
import networkx as nx
import logging
from tqdm import tqdm
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.ERROR)
class routing_graph:
    def __init__(self, point, distance, city_name, origin_name, edge_attributes, order_category=None, network_type='walk', load_graphml=None):
        """
        Initialize a routing_graph object.

        Parameters:
        - point: tuple, coordinates of the center point.
        - distance: float, distance from the center point to create the graph.
        - city_name: str, name of the city.
        - origin_name: str, name of the origin.
        - order_category: str, optional, category of the order.
        - network_type: str, type of the network to create.
        - load_graphml: str, optional, path to load an existing graphml file.

        Output:
        - Initializes the routing_graph object with the provided parameters.
        """
        self.center_point = point
        self.distance_from_point = distance
        self.city_name = city_name
        self.origin_name = origin_name
        self.order_category = order_category
        self.network_type = network_type
        self.routes = None
        self.edge_attributes = edge_attributes
        self.filtered_routes = None
        self.order_grouped_routes = None
        self.removed_paralell_edges = None
        self.removed_inf_edges = None
        if load_graphml is not None:
            self.graph = ox.load_graphml(load_graphml)
        else:
            self.graph = self.create_graph()
        self.start_node = self.find_start_node()

    def create_graph(self):
        """
        Create a graph from the center point and distance.

        Output:
        - Returns a graph object created from the center point and distance.
        """
        if self.network_type not in ['walk', 'bike', 'drive', 'drive_service', 'all', 'all_private']:
            G = ox.graph_from_point(self.center_point, dist=self.distance_from_point, custom_filter=self.network_type, simplify=False, truncate_by_edge=True)
            print('Using custom filter to create graph')        
        else:
            G = ox.graph_from_point(self.center_point, dist=self.distance_from_point, network_type=self.network_type, simplify=False, truncate_by_edge=True)

        if self.edge_attributes == 'osmid':
            G = ox.simplification.simplify_graph(G,edge_attrs_differ=['osmid'])
            print('Simplifying graph retaining unique osmid')
        else:
            G = ox.simplify_graph(G)
        G = ox.bearing.add_edge_bearings(G)
        G = ox.distance.add_edge_lengths(G)
        G.graph['city_name'] = self.city_name
        G.graph['origin_name'] = self.origin_name
        G.graph['order_category'] = self.order_category
        return G
        print(self.graph)

    def find_start_node(self):
        """
        Find the nearest node to the center point that has outgoing edges.

        Output:
        - Returns the nearest node with outgoing edges.
        """
        print(self.graph)
        nearest_node = ox.distance.nearest_nodes(self.graph, self.center_point[1], self.center_point[0])
        print('centerpoints:', self.center_point[1], self.center_point[0])
        temp_G = self.graph
        while self.graph.out_degree(nearest_node) == 0:
            temp_G.remove_node(nearest_node)
            nearest_node = ox.distance.nearest_nodes(temp_G, self.center_point[1], self.center_point[0])
        self.graph.graph['start_node'] = nearest_node
        return nearest_node

    def plot_graph(self):
        """
        Plot the graph using osmnx.

        Output:
        - Plots the graph.
        """
        if self.graph is not None:
            ox.plot_graph(self.graph)

    def remove_parallel_edges(self):
        """
        Remove parallel edges from the graph, keeping only the shortest one.

        Output:
        - Modifies the graph by removing parallel edges.
        """
        edges_to_remove = []
        for u, v, k in self.graph.edges(keys=True):
            parallel_edges = [(u, v, key) for key in self.graph[u][v]]
            if len(parallel_edges) > 1:
                shortest_edge = min(parallel_edges, key=lambda edge: self.graph.edges[edge]['length'])
                for edge in parallel_edges:
                    if edge != shortest_edge:
                        edges_to_remove.append(edge)
        self.remove_parallel_edges = edges_to_remove
        self.graph.remove_edges_from(edges_to_remove)

    def remove_infinite_edges(self):
        G = self.graph
        edges_to_remove = []
        for (u, v, data) in G.edges(data=True):
            if float(G.edges[(u, v, 0)]['weight_decision_complexity']) == float('inf'):
                edges_to_remove.append((u, v, 0))
        G.remove_edges_from(edges_to_remove)
        print(f'Removed {len(edges_to_remove)} edges with infinite decision complexity')
        self.graph = G
        self.removed_inf_edges = edges_to_remove

    def save_graph(self, filepath):
        """
        Save the graph to a graphml file.

        Parameters:
        - filepath: str, path to save the graphml file.

        Output:
        - Saves the graph to the specified filepath.
        """
        if self.graph is not None:
            ox.save_graphml(self.graph, filepath)
        else:
            print("Graph not created yet. Please call create_graph() first.")

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

    def add_simplest_path_weights(self):
        """
        Add simplest path weights to the graph using a custom algorithm.

        Output:
        - Modifies the graph by adding simplest path weights.
        """
        G = sp.simplest_path_weight_algorithm(self.graph, self.start_node)
        self.graph = G

    def add_routes(self, target_distance, margin):
        """
        Add routes to the graph based on the target distance and margin.

        Parameters:
        - target_distance: float, target distance for the routes.
        - margin: float, margin for the target distance.

        Output:
        - Adds routes to the graph and updates the routes attribute.
        """
        shortest_routes = rt.get_possible_paths(G=self.graph, origin_node=self.start_node, target_distance=target_distance, margin=margin, model='least_decision_complex', origin_name=self.origin_name)
        complex_routes = rt.get_possible_paths(G=self.graph, origin_node=self.start_node, target_distance=target_distance, margin=margin, model='shortest', origin_name=self.origin_name)

        routes = None
        if shortest_routes != None and complex_routes != None:
            routes = pd.DataFrame(shortest_routes + complex_routes)
            self.routes = routes
            print(f'Nr of shortest routes: {len(shortest_routes)} in {self.origin_name}')
            print(f'Nr of complex routes: {len(complex_routes)} in {self.origin_name}')
        elif shortest_routes == None and complex_routes != None:
            routes = pd.DataFrame(complex_routes)
            self.routes = routes
            print(f'Nr of complex routes: {len(complex_routes)} in {self.origin_name}')
            print(f'No shortest routes found in: {self.origin_name}')
        elif complex_routes == None and shortest_routes != None:
            routes = pd.DataFrame(shortest_routes)
            self.routes = routes
            print(f'Nr of shortest routes: {len(shortest_routes)} in {self.origin_name}')
            print(f'No complex routes found in: {self.origin_name}')
        else:
            print(f'No routes found at all for origin {self.origin_name} with distance {target_distance} and margin {margin}')
            #distances, paths = nx.single_source_dijkstra(self.graph, source=self.start_node, weight='length')
            #count_within_margin = sum(1 for distance in distances.values() if 1800 <= distance <= 2200)
            #print(f'Number of distances within margin: {count_within_margin}')
            #print(f'startnode has outdegree {self.graph.out_degree(self.start_node)}')



    def visualize_filtered_routes_2(self, folderpath, replace=False):
        if len(self.graph.edges) < 10:
            print(f'Graph for origin {self.origin_name} has less than 10 edges')
            return

        # Prepare folders
        city_img_folder = f'{folderpath}/maps/{self.city_name}/img'
        city_html_folder = f'{folderpath}/maps/{self.city_name}/html'
        os.makedirs(city_img_folder, exist_ok=True)
        os.makedirs(city_html_folder, exist_ok=True)

        self.filtered_routes['bbox_XminYminXmaxYmax'] = None
        self.filtered_routes['bbox_order'] = None
        self.filtered_routes['num_edges'] = None

        # Prepare data for processing
        self.filtered_routes['file_path'] = self.filtered_routes.apply(
            lambda row: f'{city_html_folder}/{row["unique_id"]}_{row["condition"]}_{self.order_category}_{self.origin_name}.html',
            axis=1
        )
        self.filtered_routes['imgpath'] = self.filtered_routes.apply(
            lambda row: f'{city_img_folder}/{row["unique_id"]}_{row["condition"]}_{self.order_category}_{self.origin_name}.png',
            axis=1
        )

        # Process routes
        for idx, route in tqdm(self.filtered_routes.iterrows(), total=len(self.filtered_routes), desc="Processing routes"):
            try:
                # Get route_gdf
                if route['model'] == 'shortest':
                    route_gdf = ox.routing.route_to_gdf(self.graph, route['nodes'], weight='length')
                elif route['model'] == 'least_decision_complex':
                    route_gdf = ox.routing.route_to_gdf(self.graph, route['nodes'], weight='normalized_decision_complexity')
                else:
                    continue

                # Get bbox
                bbox = mp.plot_route_gdf(
                    self.graph, route_gdf, route['file_path'],
                    start_node=route['start_node'], end_node=route['end_node'],
                    path_text=f'{self.city_name}', imgpath=route['imgpath'],
                    only_bbox=True
                )

                # Calculate entropy
                order, num_edges = self.calculate_street_network_entropy((bbox["north"], bbox["south"], bbox["east"], bbox["west"]))

                # Update the DataFrame directly
                self.filtered_routes.at[idx, 'bbox_XminYminXmaxYmax'] = str(bbox)
                self.filtered_routes.at[idx, 'bbox_order'] = order
                self.filtered_routes.at[idx, 'num_edges'] = num_edges

            except Exception as e:
                logger.error(f"Error processing route: {e} in {self.origin_name}")
                continue

        # Verify updates
        if 'bbox_order' not in self.filtered_routes.columns or self.filtered_routes['bbox_order'].isna().all():
            print("Warning: No routes were successfully processed.")

    def visualize_filtered_routes(self, folderpath,replace=False):
        """
        Visualize filtered routes and save the visualizations.

        Parameters:
        - folderpath: str, path to save the visualizations.

        Output:
        - Saves visualizations of the filtered routes to the specified folderpath.
        """
        if len(self.graph.edges) < 10:
            print(f'Graph for origin {self.origin_name} has less than 10 edges')
        for index, route in self.filtered_routes.iterrows():
            #print(route)
            unique_id = route['unique_id']
            condition = route['condition']
            route_nodes = route['nodes']

            city_img_folder = f'{folderpath}/maps/{self.city_name}/img'
            city_html_folder = f'{folderpath}/maps/{self.city_name}/html'
            file_path = f'{city_html_folder}/{unique_id}_{route["condition"]}_{self.order_category}_{self.origin_name}.html'
            imgpath = f'{city_img_folder}/{unique_id}_{route["condition"]}_{self.order_category}_{self.origin_name}.png'

            try:
                if route['model'] == 'shortest':
                    route_gdf = ox.routing.route_to_gdf(self.graph, route_nodes, weight='length')
                elif route['model'] == 'least_decision_complex':
                    route_gdf = ox.routing.route_to_gdf(self.graph, route_nodes, weight='normalized_decision_complexity')
            except Exception as e:
                logger.error(f"Error in route to gdf: {e} in {self.origin_name}")
                #print(f'Error in route to gdf: {e} in {self.origin_name}')
                continue

            if not os.path.exists(city_img_folder):
                os.makedirs(city_img_folder)

            if not os.path.exists(city_html_folder):
                os.makedirs(city_html_folder)

            #print(f'unique_id: {unique_id}')
            file_path = f'{city_html_folder}/{unique_id}_{route["condition"]}_{self.order_category}_{self.origin_name}.html'
            imgpath = f'{city_img_folder}/{unique_id}_{route["condition"]}_{self.order_category}_{self.origin_name}.png'

            end_node = route['end_node']

            bbox = mp.plot_route_gdf(self.graph,
                                    route_gdf,
                                    file_path,
                                    start_node=route['start_node'],
                                    end_node=end_node,
                                    path_text=f'{self.city_name}',
                                    imgpath=imgpath,
                                    only_bbox=True)
            try:
                order,num_edges = self.calculate_street_network_entropy((bbox["north"],bbox["south"],bbox["east"],bbox["west"]))
                self.filtered_routes.loc[index, 'bbox_XminYminXmaxYmax'] = f"{bbox}"
                self.filtered_routes.loc[index, 'bbox_order'] = order
                self.filtered_routes.loc[index, 'num_edges'] = num_edges
            except Exception as e:
                logger.error(f'Error in calculate_street_network_entropy: {str(e)} for {self.origin_name}')
                logger.error(f"input bbox: {bbox}, padded_bbox: {bbox}")
                logger.error(f"formatted for testing: ({bbox})")
    
    def visualize_all_routes_from_origin(self, folderpath, only_grouped=False,only_filtered=False):
        """
        Visualize all routes from the origin and save the visualizations.

        Parameters:
        - folderpath: str, path to save the visualizations.

        Output:
        - Saves visualizations of all routes from the origin to the specified folderpath.
        """
        if self.routes is not None:
            route_nodes = self.routes['nodes']
            print('route_nodes:',route_nodes)
            path = f'{folderpath}/allroutes_{self.city_name}_{self.origin_name}.html'
            #start_node = self.start_node
            mp.plot_all_routes(self.graph, self.routes, path, self.start_node)
            

    def save_plot(self, filepath):
        """
        Save a plot of the graph using osmnx with a black background and white nodes and edges.

        Parameters:
        - filepath: str, path to save the plot image.

        Output:
        - Saves the plot to the specified filepath.
        """
        if self.graph is not None:
            fig, ax = ox.plot_graph(self.graph, bgcolor='k', node_color='w', edge_color='w', show=False, close=True)
            fig.savefig(filepath, bbox_inches='tight', pad_inches=0)
        else:
            print("Graph not created yet. Please call create_graph() first.")
    def save_graph_high_res(self, filepath, origin_node, dpi=300, figsize=(20, 20), node_size=15, origin_node_size=100, edge_linewidth=4,weightstring='normalized_decision_complexity'):
        """
        Save the graph visualization in high resolution with proper memory management and black background
        """
        try:
            if self.graph is not None:
                # Clear any existing plots
                plt.close('all')
                
                # Create the main figure with a specific layout
                fig = plt.figure(figsize=figsize, facecolor='white')
                
                # Create a gridspec layout with two columns
                gs = fig.add_gridspec(1, 2, width_ratios=[20, 1])
                
                # Create the map axis with black background
                ax_map = fig.add_subplot(gs[0, 0], facecolor='white')
                
                # Get the normalized_decision_complexity values for all edges
                edge_values = []
                for _, _, data in self.graph.edges(data=True):
                    edge_values.append(data.get(weightstring, 0))
                    
                # Create a colormap for edges
                norm = plt.Normalize(vmin=min(edge_values), vmax=max(edge_values))
                cmap = plt.cm.viridis
                
                # Create the edge colors list
                edge_colors = []
                for _, _, data in self.graph.edges(data=True):
                    value = data.get(weightstring, 0)
                    edge_colors.append(cmap(norm(value)))
                
                # Create node colors and sizes lists
                node_colors = []
                node_sizes = []
                for node in self.graph.nodes():
                    if node == origin_node:
                        node_colors.append('#00FF00')
                        node_sizes.append(origin_node_size)
                    else:
                        node_colors.append('black')
                        node_sizes.append(node_size)
                
                # Plot the graph with explicit black background
                ox.plot_graph(
                    self.graph,
                    bgcolor='white',  # Explicit black background
                    node_color=node_colors,
                    node_size=node_sizes,
                    edge_color=edge_colors,
                    edge_linewidth=edge_linewidth,
                    ax=ax_map,
                    show=False,
                    close=False
                )
                
                # Add legend for origin node
                legend_elements = [Line2D([0], [0], marker='o', color='k', label='Origin',
                                        markerfacecolor='#00FF00', markersize=15)]
                ax_map.legend(handles=legend_elements, loc='upper right', 
                            facecolor='white', edgecolor='black')
                ax_map.get_legend().get_texts()[0].set_color('black')
                
                # Set axis background to black
                ax_map.set_facecolor('white')
                
                # Create the colorbar axis with black background
                ax_cbar = fig.add_subplot(gs[0, 1], facecolor='white')
                
                # Add colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, cax=ax_cbar, label=weightstring)
                cbar.ax.yaxis.label.set_color('black')
                cbar.ax.tick_params(colors='black', labelsize=20)  # Increase the size of the colorbar tick text
                cbar.ax.set_facecolor('white')
                # Ensure the figure background is black
                fig.patch.set_facecolor('white')
                
                # Remove axis spines
                ax_map.set_axis_off()
                
                # Adjust the layout to be tight
                plt.tight_layout()
                
                # Save with high resolution and black background
                fig.savefig(
                    filepath,
                    dpi=dpi,
                    bbox_inches='tight',
                    pad_inches=0,
                    facecolor='white',
                    edgecolor='none'
                )
                
                # Clean up
                plt.close(fig)
                plt.close('all')
                gc.collect()
                
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            plt.close('all')
            gc.collect()
        finally:
            plt.close('all')
            gc.collect()
        
    def calculate_street_network_entropy(self, bbox_tuple):
        """
        Calculate the street network entropy within a bounding box.

        Parameters:
        - bbox_tuple: tuple, bounding box coordinates.

        Output:
        - Returns the orientation order calculated from the entropy.
        """
        G = ox.graph_from_bbox(bbox=bbox_tuple, network_type='drive', simplify=True, retain_all=True, truncate_by_edge=False)
        G = ox.convert.to_undirected(G)
        G = ox.add_edge_bearings(G)
        entropy = ox.bearing.orientation_entropy(G, num_bins=36)
        order = self.calculate_orientation_order(entropy, num_bins=36)
        num_edges = G.number_of_edges()
        #print(f"order: {order} and num_edges: {num_edges} for city {self.city_name} and origin {self.origin_name}")
        return order,num_edges

    def calculate_orientation_order(self, entropy, num_bins=36):
        """
        Calculate the orientation order from the entropy.

        Parameters:
        - entropy: float, the entropy value.
        - num_bins: int, number of bins used in the entropy calculation.

        Output:
        - Returns the orientation order.
        """
        max_nats = math.log(num_bins)
        min_nats = math.log(4)
        orientation_order = 1 - ((entropy - min_nats) / (max_nats - min_nats))**2
        return orientation_order
    
    def group_routes_by_order(self):
        # Group 1: 0.0 to 0.3
        routes_below_0_2 = self.filtered_routes[self.filtered_routes['bbox_order'] < 0.2].copy()
        routes_below_0_2['bbox_order_type'] = 'below_0_2'

        # Group 2: 0.3 to 0.6
        routes_between_0_4_and_0_5 = self.filtered_routes[(self.filtered_routes['bbox_order'] >= 0.4) & 
                                                        (self.filtered_routes['bbox_order'] < 0.5)].copy()
        routes_between_0_4_and_0_5['bbox_order_type'] = 'between_0_4_and_0_5'

        # Group 3: 0.6 and above
        routes_above_0_6 = self.filtered_routes[self.filtered_routes['bbox_order'] >= 0.6].copy()
        routes_above_0_6['bbox_order_type'] = 'above_0_6'
        
        if len(routes_above_0_6) > 0:
            complex_routes = routes_above_0_6[routes_above_0_6['condition'] == 'complex']
            print(f'Number of complex routes above 0.6: {len(complex_routes)} in {self.origin_name}, {self.city_name}')

        self.order_grouped_routes = pd.concat([routes_below_0_2, 
                                            routes_between_0_4_and_0_5, 
                                            routes_above_0_6])
    def visualize_grouped_routes(self, folderpath):
        """
        Visualize grouped routes and save the visualizations.

        Output:
        - Saves visualizations of the grouped routes.
        """
        imgpath_list = []
        htmlpath_list = []
        print(f'visualizing routes in {self.origin_name} in {self.city_name}')

        cutoff_complex_above_0_6 = 10
        cutoff_complex_between_0_4_and_0_5 = 10
        cutoff_complex_below_0_2 = 10
        cutoff_simple_above_0_6 = 10
        cutoff_simple_between_0_4_and_0_5 = 10
        cutoff_simple_below_0_2 = 10
        if self.order_grouped_routes is not None:
            for index, route in self.order_grouped_routes.iterrows():
                if route['condition'] == 'complex' and route['bbox_order_type'] == 'above_0_6' and cutoff_complex_above_0_6 > 0:
                    cutoff_complex_above_0_6 -= 1
                elif route['condition'] == 'complex' and route['bbox_order_type'] == 'between_0_4_and_0_5' and cutoff_complex_between_0_4_and_0_5 > 0:
                    cutoff_complex_between_0_4_and_0_5 -= 1
                elif route['condition'] == 'complex' and route['bbox_order_type'] == 'below_0_2' and cutoff_complex_below_0_2 > 0:
                    cutoff_complex_below_0_2 -= 1
                elif route['condition'] == 'simple' and route['bbox_order_type'] == 'above_0_6' and cutoff_simple_above_0_6 > 0:
                    cutoff_simple_above_0_6 -= 1
                elif route['condition'] == 'simple' and route['bbox_order_type'] == 'between_0_4_and_0_5' and cutoff_simple_between_0_4_and_0_5 > 0:
                    cutoff_simple_between_0_4_and_0_5 -= 1
                elif route['condition'] == 'simple' and route['bbox_order_type'] == 'below_0_2' and cutoff_simple_below_0_2 > 0:
                    cutoff_simple_below_0_2 -= 1
                else:
                    continue

                route_nodes = route['nodes']
                if route['model'] == 'shortest':
                    route_gdf = ox.routing.route_to_gdf(self.graph, route_nodes, weight='length')
                else:
                    route_gdf = ox.routing.route_to_gdf(self.graph, route_nodes, weight='normalized_decision_complexity')

                city_img_folder = f'{folderpath}/final_routes/img'
                city_html_folder = f'{folderpath}/final_routes/html'

                if not os.path.exists(city_img_folder):
                    os.makedirs(city_img_folder)

                if not os.path.exists(city_html_folder):
                    os.makedirs(city_html_folder)

                file_path = f'{city_html_folder}/{route["condition"]}_{route["bbox_order_type"]}_{self.origin_name}_{route["unique_id"]}.html'
                imgpath = f'{city_img_folder}/{route["condition"]}_{route["bbox_order_type"]}_{self.origin_name}_{route["unique_id"]}.png'
                imgpath_list.append(imgpath)
                htmlpath_list.append(file_path)

                if os.path.exists(imgpath):
                    continue

                end_node = route['end_node']
                start_node = route['start_node']

                mp.plot_route_gdf(G=self.graph,
                        route_gdf=route_gdf,
                        file_path=file_path,
                        start_node=start_node,
                        end_node=end_node,
                        path_text=f'{self.city_name}, route model:shortest environment_category {self.order_category} origin_name:{self.origin_name}',
                        imgpath=imgpath)
                
        return imgpath_list, htmlpath_list
    
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

class routing_graph_set:
    def __init__(self, routing_graphs):
        """
        Initialize a routing_graph_set object.

        Parameters:
        - routing_graphs: list, list of routing_graph objects.

        Output:
        - Initializes the routing_graph_set object with the provided parameters.
        """
        self.routing_graphs = routing_graphs
        self.max_distance = None
        self.max_decision_complexity = None
        self.max_instruction_equivalence = None
        self.potential_routes = None
        self.filtered_routes = None
        self.order_grouped_routes = None

    def visualize_all_filtered_routes(self, folderpath):
        """
        Visualize all filtered routes and save the visualizations.

        Parameters:
        - folderpath: str, path to save the visualizations.

        Output:
        - Saves visualizations of the filtered routes to the specified folderpath.
        """
        all_filtered_routes = []
        n_cities = len(self.routing_graphs)
        for routing_graph in self.routing_graphs:
            if routing_graph.filtered_routes is not None:
                routing_graph.visualize_filtered_routes_2(folderpath)
                routing_graph.filtered_routes.to_csv(f'{folderpath}/csv_checking/bbox_routes_{routing_graph.origin_name}.csv')
                all_filtered_routes.append(routing_graph.filtered_routes)
            n_cities -= 1
            print("nr of cities left to visualize:", n_cities)
        all_filtered_routes_df = pd.concat(all_filtered_routes, ignore_index=True)
        all_filtered_routes_df.to_csv(f'{folderpath}/visualized_routes.csv')
        self.filtered_routes = all_filtered_routes_df
    def get_min_max_routes_percentile(self, percentile_range=(25, 75)):
        """
        Get the routes with the minimum and maximum decision complexity using percentiles.

        Parameters:
        - percentile_range: tuple, range of percentiles to consider for simple and complex routes.
                            Default is (25, 75), meaning the bottom 25% are 'simple' and top 25% are 'complex'.

        Output:
        - Updates the filtered_routes attribute with the routes having min and max complexity.
        """
    
        # Calculate percentiles
        lower_percentile, upper_percentile = percentile_range
        lower_threshold = np.percentile(self.potential_routes['sum_decision_complexity'], lower_percentile)
        upper_threshold = np.percentile(self.potential_routes['sum_decision_complexity'], upper_percentile)

        print(f'Lower threshold (percentile {lower_percentile}): {lower_threshold}')
        print(f'Upper threshold (percentile {upper_percentile}): {upper_threshold}')

        # Select simple and complex routes
        simplest_routes = self.potential_routes[self.potential_routes['sum_decision_complexity'] <= lower_threshold]
        complex_routes = self.potential_routes[self.potential_routes['sum_decision_complexity'] >= upper_threshold]

        simplest_routes['condition'] = 'simple'
        complex_routes['condition'] = 'complex'

        filtered_routes = pd.concat([simplest_routes, complex_routes], ignore_index=True).reset_index(drop=True)

        print('Number of filtered routes:', len(filtered_routes))
        
        for routing_graph in self.routing_graphs:
            if routing_graph.routes is not None:
                routing_graph.filtered_routes = filtered_routes[filtered_routes['origin_name'] == routing_graph.origin_name]
            else:
                print(f'No filtered routes in {routing_graph.origin_name}')
        
        self.filtered_routes = filtered_routes

        # Additional statistics
        print('\nDescriptive statistics:')
        print(filtered_routes.groupby('condition')['sum_decision_complexity'].describe())

        # Calculate and print the ratio of mean complexities
        mean_simple = simplest_routes['sum_decision_complexity'].mean()
        mean_complex = complex_routes['sum_decision_complexity'].mean()
        complexity_ratio = mean_complex / mean_simple
        print(f'\nRatio of mean complexities (complex / simple): {complexity_ratio:.2f}')
    def get_min_max_routes(self,zscore_range=(1,1.5)):
        """
        Get the routes with the minimum and maximum decision complexity.

        Parameters:
        - method: str, method to calculate the min and max complexity ('median', 'mean', 'global').

        Output:
        - Updates the filtered_routes attribute with the routes having min and max complexity.
        """

        global_mean = self.potential_routes['sum_decision_complexity'].mean()
        global_std = self.potential_routes['sum_decision_complexity'].std()
        self.potential_routes['zscore'] = (self.potential_routes['sum_decision_complexity'] - global_mean) / global_std
        print(f'Global mean: {global_mean} and global std: {global_std}')
        simple_z_score_floor = -zscore_range[0]
        simple_z_score_roof = -zscore_range[1]
        complex_z_score_floor = zscore_range[0]
        complex_z_score_roof = zscore_range[1]

        simplest_routes = self.potential_routes[(self.potential_routes['zscore'] >= simple_z_score_roof) & (self.potential_routes['zscore'] <= simple_z_score_floor)]
        complex_routes = self.potential_routes[(self.potential_routes['zscore'] >= complex_z_score_floor) & (self.potential_routes['zscore'] <= complex_z_score_roof)]
        simplest_routes['condition'] = 'simple'
        complex_routes['condition'] = 'complex'

        print(f'Number of simplest routes: {len(simplest_routes)}')
        print(f'Number of complex routes: {len(complex_routes)}')


        filtered_routes = pd.concat([simplest_routes, complex_routes], ignore_index=True).reset_index(drop=True)

        print('len of filtered routes', len(filtered_routes))
        for routing_graph in self.routing_graphs:
            if routing_graph.routes is not None:
                routing_graph.filtered_routes = filtered_routes[filtered_routes['origin_name'] == routing_graph.origin_name]
            else:
                print(f'No filtered routes in {routing_graph.origin_name}')
                continue

        self.filtered_routes = filtered_routes




    def routes_to_csv(self, filepath):
        """
        Save the potential routes to a CSV file.

        Parameters:
        - filepath: str, path to save the CSV file.

        Output:
        - Saves the potential routes to the specified filepath.
        """
        self.potential_routes.to_csv(filepath)

    def add_potential_routes(self, target_distance, margin):
        """
        Add potential routes to the routing graphs based on the target distance and margin.

        Parameters:
        - target_distance: float, target distance for the routes.
        - margin: float, margin for the target distance.

        Output:
        - Updates the potential_routes attribute with the added routes.
        """
        all_routes = []
        for routing_graph in self.routing_graphs:
            routing_graph.add_routes(target_distance, margin)
            all_routes.append(routing_graph.routes)
        pd.concat(all_routes, ignore_index=True)
        all_routes_df = pd.concat(all_routes, ignore_index=True)
        self.potential_routes = all_routes_df

    def get_max_values(self):
        """
        Get the maximum values of various attributes from the routing graphs.

        Output:
        - Updates the max_distance, max_decision_complexity, and max_instruction_equivalence attributes.
        """
        graphs_max_values = []
        for routing_graph in self.routing_graphs:
            G = routing_graph.graph
            distance_list = []
            decision_complexity_list = []
            instruction_equivalent_list = []
            deviation_from_prototypical_list = []
            for (u, v, data) in G.edges(data=True):
                if float(G.edges[(u, v, 0)]['weight_decision_complexity']) != float('inf'):
                    decision_complexity_list.append(float(G.edges[(u, v, 0)]['weight_decision_complexity']))
                else:
                    G.edges[(u, v, 0)]['weight_decision_complexity'] = 1
                    decision_complexity_list.append(float(G.edges[(u, v, 0)]['weight_decision_complexity']))
                distance_list.append(float(G.edges[(u, v, 0)]['length']))
                deviation_from_prototypical_list.append(float(G.edges[(u, v, 0)]['weight_deviation_from_prototypical']))
                instruction_equivalent_list.append(float(G.edges[(u, v, 0)]['weight_instruction_equivalent']))
            max_distance = max(distance_list)
            max_instruction_equivalence = max(instruction_equivalent_list)
            max_decision_complexity = max(decision_complexity_list)
            max_deviation_from_prototypical = max(deviation_from_prototypical_list)
            graph_max = {
                'max_distance': max_distance,
                'max_decision_complexity': max_decision_complexity,
                'max_instruction_equivalence': max_instruction_equivalence,
                'max_deviation_from_prototypical': max_deviation_from_prototypical
            }
            graphs_max_values.append(graph_max)
        graphs_max_values_df = pd.DataFrame(graphs_max_values)
        self.max_distance = graphs_max_values_df['max_distance'].max()
        self.max_decision_complexity = graphs_max_values_df['max_decision_complexity'].max()

    def normalize_graphs(self):
        """
        Normalize the graphs based on the maximum values of various attributes.

        Output:
        - Updates the routing graphs with normalized edge attributes.
        """
        normalized_graphs = []
        for routing_graph in self.routing_graphs:
            G = routing_graph.graph
            for (u, v, data) in G.edges(data=True):
                distance = float(G.edges[(u, v, 0)]['length'])
                decision_complexity = float(G.edges[(u, v, 0)]['weight_decision_complexity'])
                instruction_equivalence = float(G.edges[(u, v, 0)]['weight_instruction_equivalent'])
                deviation_from_prototypical = float(G.edges[(u, v, 0)]['weight_deviation_from_prototypical'])
                if decision_complexity == float('inf'):
                    decision_complexity = 1
                normalized_distance = (1 - (distance / self.max_distance))
                normalized_decision_complexity = decision_complexity / self.max_decision_complexity
                G.edges[(u, v, 0)]['normalized_distance'] = float(normalized_distance)
                G.edges[(u, v, 0)]['normalized_decision_complexity'] = float(normalized_decision_complexity)
            routing_graph.graph = G
            normalized_graphs.append(routing_graph)
        
        self.routing_graphs = normalized_graphs
        
    def group_routes_by_order(self):
        all_grouped_routes = []
        
        for routing_graph in self.routing_graphs:
            try:
                if routing_graph.filtered_routes is not None and not routing_graph.filtered_routes.empty:
                    routing_graph.group_routes_by_order()
                    if routing_graph.order_grouped_routes is not None and not routing_graph.order_grouped_routes.empty:
                        all_grouped_routes.append(routing_graph.order_grouped_routes)
                    else:
                        print(f'No grouped routes generated for {routing_graph.origin_name}')
                else:
                    print(f'No filtered routes in {routing_graph.origin_name}')
            except Exception as e:
                print(f'Error processing routes for {routing_graph.origin_name}: {str(e)}')
                continue
        
        if all_grouped_routes:
            self.order_grouped_routes = pd.concat(all_grouped_routes, ignore_index=True)
        else:
            print('No routes were grouped successfully')
            self.order_grouped_routes = pd.DataFrame()

    def visualize_all_grouped_routes(self, folderpath):
        """
        Visualize grouped routes and save the visualizations.

        Output:
        - Saves visualizations of the grouped routes.
        """
        all_filtered_routes = []
        all_imgpaths = []
        all_htmlpaths = []
        for routing_graph in self.routing_graphs:
            imgpath_list,htmlpath_list =routing_graph.visualize_grouped_routes(folderpath)
            all_filtered_routes.append(routing_graph.order_grouped_routes)
            all_imgpaths.extend(imgpath_list)
            all_htmlpaths.extend(htmlpath_list)
        all_filtered_routes_df = pd.concat(all_filtered_routes, ignore_index=True)
        all_filtered_routes_df.to_csv(f'{folderpath}/grouped_routes.csv')
        self.order_grouped_routes = all_filtered_routes_df
        return all_imgpaths, all_htmlpaths

    def save_pickle(self, filepath):
        """
        Save the routing_graph_set object to a pickle file.

        Parameters:
        - filepath: str, path to save the pickle file.

        Output:
        - Saves the routing_graph_set object to the specified filepath.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_pickle(filepath):
        """
        Load a routing_graph_set object from a pickle file.

        Parameters:
        - filepath: str, path to the pickle file.

        Output:
        - Returns the loaded routing_graph_set object.
        """
        with open(filepath, 'rb') as f:
            routing_graph = pickle.load(f)
        return routing_graph
