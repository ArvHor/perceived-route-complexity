import os
import osmnx as ox
import route_network_analysis as rna
import numpy as np

ume = (63.825603, 20.262998)
home = (63.862483, 20.336844)

if os.path.exists("bug_testing/ume_graph.graphml"):
    ume_graph = rna.origin_graph.from_graphml("bug_testing/ume_graph.graphml")
else:
    ume_graph = rna.origin_graph(
        origin_point=ume,
        distance_from_point=6000,
        network_type="drive",
        simplify=True,
        remove_parallel=True,
        city_name="Ume",
    )
    ume_graph.add_simplest_paths_from_origin()

    ume_graph.add_weights(
        [
            "deviation_from_prototypical",
            "node_degree",
            "node_degree",
            "instruction_equivalent",
            "betweenness_centrality",
        ]
    )
    ume_graph.save_graph("bug_testing/ume_graph.graphml")
    ox.plot_graph(
        ume_graph.graph,
        node_color="#8b0000",
        node_size=5,
        edge_linewidth=1,
        edge_color="black",
        bgcolor="white",
        save=True,
        filepath="bug_testing/ume.png",
        show=False,
    )

home_node = ox.nearest_nodes(ume_graph.graph, home[1], home[0])
print(home_node)
od_p = rna.od_pair(ume_graph.graph, origin=ume_graph.start_node, destination=home_node)
od_p_data = od_p.get_comparison_dict()
# Find the shortest route and plot it
shortest_route = ox.shortest_path(
    ume_graph.graph, ume_graph.start_node, home_node, weight="length"
)
ox.plot_graph_route(
    od_p.subgraph,
    shortest_route,
    node_size=5,
    bgcolor="white",
    route_color="red",
    edge_color="black",
    node_color="black",
    route_linewidth=4,
    edge_linewidth=0.5,
)

od_p.create_orientation_plot("bug_testing/ume_od_pair.png")
od_p_env_dist = od_p.env_bearing_dist_weighted
od_p_route_dist = od_p.route_direction_bearing_dist
# print(od_p_env_dist)
# print(od_p_route_dist)
# print("\n Normalized: \n")
env = od_p_env_dist / np.sum(od_p_env_dist)
route = od_p_route_dist / np.sum(od_p_env_dist)
