import pandas as pd

class GraphSet:
    def __init__(self):
        self.graphs = set()

    def add_graph(self, graph):
        """Adds a Graph to the set of graphs."""
        self.graphs.add(graph)

    def get_all_routes_data(self):
        """Retrieves data about all routes from all OD pairs in all graphs."""
        all_routes_data = []
        for graph in self.graphs:
            for od_pair in graph.od_pairs:
                df = od_pair.to_dataframe()
                df['graph_id'] = id(graph)  # Add an identifier for the graph
                all_routes_data.append(df)
        return pd.concat(all_routes_data, ignore_index=True)

    def get_graph_set_info(self):
        """Calculates and returns information about the set of graphs."""
        info = {}
        info['num_graphs'] = len(self.graphs)
        # ... Add more aggregate calculations as needed ...
        return info