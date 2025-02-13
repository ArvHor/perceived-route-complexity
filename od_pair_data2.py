import ast
import sys

import numpy as np
import pandas as pd
from origin_graph_set import origin_graph_set
import osmnx as ox
from route import route
import multiprocessing
import logging
import networkx as nx
logging.basicConfig(
    filename='info.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

path_1 = "graph_sets/local_graphs_randnode1_simplified_drive.pkl"
path_2 = "graph_sets/local_graphs_randnode2_simplified_drive.pkl"
path_3 = "graph_sets/local_graphs_randnode3_simplified_drive.pkl"
paths = [path_1, path_2, path_3]

#og_set_1 = origin_graph_set.load_pickle(path_1)
#og_set_2 = origin_graph_set.load_pickle(path_2)
#og_set_3 = origin_graph_set.load_pickle(path_3)




def get_od_pairs(paths):
    od_complexities = []
    for path in paths:
        og_set = origin_graph_set.load_pickle(path)
        print("Loaded origin graph set")
        for origin_graph in og_set.origin_graphs:
            G = origin_graph.graph
            city_name = origin_graph.city_name
            print(f"now processing City: {city_name}")
            for od_pair in origin_graph.od_pairs:
                shortest_path = od_pair.shortest_path
                simplest_path = od_pair.simplest_path


                shortest_edges = list(nx.utils.pairwise(shortest_path.nodes))
                simplest_edges = list(nx.utils.pairwise(simplest_path.nodes))


                shortest_complexity,complexities_shortest,shortest_turns = route.calculate_route_complexity_post_hoc(G,shortest_edges)


                simplest_complexity,complexities_simplest,simplest_turns = route.get_route_complexity_optimized(G,simplest_edges)

                origin_node = od_pair.origin_node
                destination_node = od_pair.destination_node


                alt_simplest = None
                simplest_complexity_2 = None

                if nx.has_path(G, origin_node, destination_node):
                    
                    alt_simplest_nodes,alt_simplest_edges,weightsum = route.find_simplest_path(G, origin_node, destination_node)

                    simplest_complexity_2,complexities_simplest_2,simplest_turns_2 = route.get_route_complexity_optimized(G,list(nx.utils.pairwise(alt_simplest_nodes)))


                




                if shortest_complexity < simplest_complexity:
                    print("-----------------------beginning------------------------")
                    print(f"Shortest complexity: {shortest_complexity}, simplest complexity: {simplest_complexity_2}")
                    print(f"difference: {shortest_complexity - simplest_complexity}")
                    
                    print("...")
                    if simplest_complexity_2 is not None:
                        if simplest_complexity_2 > shortest_complexity:
                            print(f":-( alt simplest complexity: {simplest_complexity_2} difference: {shortest_complexity - simplest_complexity_2}")
                            print(f"weightsum: {weightsum}")

                        else:
                            print(f":) no worries alt simplest complexity: {simplest_complexity_2} difference: {shortest_complexity - simplest_complexity_2}")
                            print(f"sum of complexities in shortest path: {sum(complexities_shortest)}, sum of complexities in simplest path: {sum(complexities_simplest_2)}")
                            if sum(complexities_shortest) < sum(complexities_simplest_2):
                                print(f"?????? sum of complexities in shortest path is less than sum of complexities in simplest path difference: {sum(complexities_shortest) - sum(complexities_simplest_2)}")
                                print(f"shortest complexities: {complexities_shortest}")
                                print(f"simplest complexities: {complexities_simplest_2}")

                    print("-----------------------end------------------------")
                else:
                    pass

                od_complexities.append({
                    "origin_node": origin_node,
                    "destination_node": destination_node,
                    "shortest_complexity": shortest_complexity,
                    "simplest_complexity": simplest_complexity
                })
    return od_complexities

od_pair_data = get_od_pairs(paths)

od_pair_data_df = pd.DataFrame(od_pair_data)
od_pair_data_df.to_csv("od_pair_data/isbrokenmaybe.csv")