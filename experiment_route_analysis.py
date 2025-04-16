import os
import ast
import pandas as pd
import osmnx as ox
from network_route_analysis import weighting_algorithms as wa
from network_route_analysis.od_pair import od_pair
import networkx as nx
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compare_routes():
    route_data = pd.read_csv(os.path.join("experiment_routes", "route_data.csv"))
    graph_files = pd.read_csv(os.path.join("experiment_routes", "graph_city_dicts.csv"))

    comparison_dicts = []
    graph_files['graph_file'] = graph_files['graph_file'].str.replace('\\', '/')

    #api_keys = pd.read_csv("api_keys.csv")
    #google_key = api_keys.loc[api_keys['service'] == 'Gmaps', 'key'].values[0]
    for index, row in route_data.iterrows():
        filepath = graph_files.loc[graph_files["city_name"] == row["city_name"]]["graph_file"].values[0]
        print(filepath)
        graph = ox.load_graphml(filepath)
        graph,_ = wa.add_deviation_from_prototypical_weights(graph)
        graph,_ = wa.add_instruction_equivalent_weights(graph)
        graph,_ = wa.add_node_degree_weights(graph)
        betweenness_centrality = nx.betweenness_centrality(graph, normalized=True)
        nx.set_node_attributes(graph, betweenness_centrality, 'betweenness_centrality')
        #graph = ox.elevation.add_node_elevations_google(graph, api_key=google_key,pause=0.1)

        
        #graph.graph['node_attributes'] = ast.literal_eval(graph.graph['node_attributes']).append('elevation')



        old_complexity = row['sum_decision_complexity']
        route_nodes = ast.literal_eval(row["nodes"])
        
        wstring = "length"
        if row["weight"] == "least_decision_complex":
            route_nodes = route_nodes.reverse()
            wstring = "decision_complexity"

        new_route_od_pair_data = od_pair.from_route(graph,route_nodes,wstring)
        new_route_dict = new_route_od_pair_data.get_comparison_dict_single_path()
        new_route_dict['old_complexity'] = old_complexity
        new_route_dict['id'] = row['id']
        new_route_dict['complexity_difference'] = old_complexity - new_route_od_pair_data.path.complexity
        new_route_dict['route_exp_condition'] = row['condition']
       
        comparison_dicts.append(new_route_dict)

    return pd.DataFrame(comparison_dicts)


if __name__ == "__main__":
    df = compare_routes()
    df.to_json(os.path.join("experiment_routes/experiment_route_data.json"), orient="records",
               default_handler=str, indent=2)
    df.to_csv(os.path.join("experiment_routes/experiment_route_data.csv"))
