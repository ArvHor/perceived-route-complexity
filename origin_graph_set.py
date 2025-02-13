import logging
import pickle
import pandas as pd
import multiprocessing
import osmnx as ox

#info_handler = logging.FileHandler('info.log')
#info_handler.setLevel(logging.INFO)
#info_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))


class origin_graph_set:
    def __init__(self, origin_graphs):
        self.origin_graphs = origin_graphs



    def add_max_values(self):
        """
        Get the maximum values of various attributes from the routing graphs.

        Output:
        - Updates the max_distance, max_decision_complexity, and max_instruction_equivalence attributes.
        """
        graphs_max_values = []
        for origin_graph in self.origin_graphs:
            print(f"Calculating max values for {origin_graph.origin_info['city_name']}")
            distance_list = []
            decision_complexity_list = []
            instruction_equivalent_list = []
            deviation_from_prototypical_list = []
            node_degree_list = []
            for (u, v, data) in origin_graph.graph.edges(data=True):
                #print(f"Data: {data}")
                if float(data['decision_complexity']) != float('inf'):
                    decision_complexity_list.append(float(data['decision_complexity']))
                else:
                    logging.error("THIS SHOULD NOT HAPPEN")
                    data['decision_complexity'] = 1
                    decision_complexity_list.append(float(data['decision_complexity']))

                distance_list.append(float(data['length']))          
                deviation_from_prototypical_list.append(float(data.get('deviation_from_prototypical',0)))
                instruction_equivalent_list.append(float(data['instruction_equivalent']))
                node_degree_list.append(float(data['node_degree']))
            
            max_length = max(distance_list)
            max_instruction_equivalence = max(instruction_equivalent_list)
            max_decision_complexity = max(decision_complexity_list)
            max_deviation_from_prototypical = max(deviation_from_prototypical_list)
            max_node_degree = max(node_degree_list)
            graph_max = {
                'max_length': max_length,
                'max_decision_complexity': max_decision_complexity,
                'max_instruction_equivalence': max_instruction_equivalence,
                'max_deviation_from_prototypical': max_deviation_from_prototypical,
                'max_node_degree': max_node_degree
            }
            graphs_max_values.append(graph_max)
        #print(f"Calculating max values for all graphs, dictlist:{graphs_max_values}")
        graphs_max_values_df = pd.DataFrame(graphs_max_values)
        print(f"Calculating max values for all graphs, df:{graphs_max_values_df}")
        self.max_decision_complexity = graphs_max_values_df['max_decision_complexity'].max()

    def add_normalized_weights(self):
        """
        Normalize the weights of the routing graphs.

        Output:
        - Updates the normalized_decision_complexity, normalized_instruction_equivalence, and normalized_distance attributes.
        """
        self.add_max_values()
        origin_graph_list = []
        for origin_graph in self.origin_graphs:
            origin_graph.graph.graph['max_decision_complexity'] = self.max_decision_complexity
            for (u, v, data) in origin_graph.graph.edges(data=True):
                data['normalized_decision_complexity'] = float(data['decision_complexity']) / float(self.max_decision_complexity)
            origin_graph.weights_list.append('normalized_decision_complexity')
            origin_graph_list.append(origin_graph)
        self.origin_graphs = origin_graph_list
    def save_graphs(self):
        """
        Save the routing graphs to graphml files.

        Output:
        - Saves the routing graphs to graphml files.
        """
        for origin_graph in self.origin_graphs:
            origin_graph.save_graph()
    
    def add_all_od_pairs(self, min_radius=3000, max_radius=3500):
            num_processes = multiprocessing.cpu_count()
            print(f"Number of processes: {num_processes}")

            with multiprocessing.Manager() as manager:
                # Create a shared list that can be modified by the worker processes
                shared_origin_graphs = manager.list(self.origin_graphs)

                with multiprocessing.Pool(processes=num_processes) as pool:
                    results = []
                    for i, o_graph in enumerate(shared_origin_graphs):
                        # Pass the index of o_graph in the shared list, and the shared list itself
                        result = pool.apply_async(origin_graph_set._create_od_pairs_wrapper,
                                                args=(o_graph, min_radius, max_radius, i, shared_origin_graphs))
                        results.append(result)

                    # Wait for all processes to complete
                    for result in results:
                        result.get()

                # Update self.origin_graphs with the modified graphs from the shared list
                self.origin_graphs = list(shared_origin_graphs)

    def get_all_od_pair_data(self):
        od_pair_pair_dfs = []
        for origin_graph in self.origin_graphs:
            od_pair_df = origin_graph.get_od_pair_data()
            od_pair_pair_dfs.append(od_pair_df)
        return pd.concat(od_pair_pair_dfs)
    
    def get_all_od_pair_data_parallel(self):
        num_processes = multiprocessing.cpu_count()
        print(f"Number of processes: {num_processes}")

        with multiprocessing.Pool(processes=num_processes) as pool:
            # Use pool.apply_async to call get_od_pair_data on each origin_graph
            results = [pool.apply_async(origin_graph.get_od_pair_data)
                       for origin_graph in self.origin_graphs]

            # Retrieve results and ensure they are dataframes
            od_pair_dfs = []
            for result in results:
                df = result.get()
                if isinstance(df, pd.DataFrame):  # Check if the result is a DataFrame
                    od_pair_dfs.append(df)
                else:
                    print(f"Warning: Result is not a DataFrame: {type(df)}")

        # Concatenate the DataFrames outside the pool context
        if od_pair_dfs:
            return pd.concat(od_pair_dfs)
        else:
            print("Warning: No DataFrames generated.")
            return None # or None, depending on how you want to handle this case
   
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
    def add_length_fix(self):
        for origin_graph in self.origin_graphs:
            ox.distance.add_edge_lengths(origin_graph.graph)
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
    def _create_od_pairs_wrapper(o_graph, min_radius, max_radius, index, shared_list):
        """
        Wrapper function to call create_od_pairs and update the shared list.
        This needs to be a static method to be pickleable.
        """
        o_graph.create_od_pairs(min_radius, max_radius)
        shared_list[index] = o_graph  # Update the o_graph in the shared list
        return "done" # returning something is necessary for apply_async
    
    
    