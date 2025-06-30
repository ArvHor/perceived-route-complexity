import os
from matplotlib import pyplot as plt
import osmnx as ox
import pandas as pd
import route_network_analysis as rna
import json
import numpy as np

ume = (63.825603, 20.262998)
home = (63.862483, 20.336844)

if os.path.exists("bug_testing/ume_graph.graphml"):
    ume_graph = rna.origin_graph.from_graphml("bug_testing/ume_graph.graphml")
else:
    ume_graph = rna.origin_graph(origin_point=ume, distance_from_point=6000, network_type="drive", simplify=True,
                             remove_parallel=True, city_name="Ume")
    ume_graph.add_simplest_paths_from_origin()
    ume_graph.add_weights('deviation_from_prototypical')
    ume_graph.add_weights('node_degree')
    ume_graph.add_weights('instruction_equivalent')
    ume_graph.add_weights('betweenness_centrality')
    ume_graph.save_graph("bug_testing/ume_graph.graphml")
    ox.plot_graph(ume_graph.graph, node_color='#8b0000', node_size=5, edge_linewidth=1, edge_color='black',
                  bgcolor='white',
                  save=True, filepath="bug_testing/ume.png", show=False)

home_node = ox.nearest_nodes(ume_graph.graph, home[1], home[0])
print(home_node)
od_p = rna.od_pair(ume_graph.graph, origin=ume_graph.start_node, destination=home_node)
od_p_data = od_p.get_odpair_df()

od_p_data.to_csv("bug_testing/ume_od_pair_data.csv", index=False)
# Find the shortest route and plot it
shortest_route = ox.shortest_path(ume_graph.graph, ume_graph.start_node, home_node, weight='length')
subgraph = od_p.get_subgraph()

route_gdf = ox.routing.route_to_gdf(ume_graph.graph, shortest_route,weight='length')
filepath = "bug_testing/ume_route.html"
polygon = od_p.polygon
#rna.map_plotting.plot_route_gdf(G=subgraph, start_node=od_p.origin_node,
#                                end_node=od_p.destination_node,
#                                route_gdf=route_gdf,
#                                map_tiles="OpenStreetMap.Mapnik",
#                                file_path=filepath,
#                                truncation_polygon=polygon)
filepath = "bug_testing/ume_route_clean.html"
#rna.map_plotting.plot_route_gdf(G=subgraph, start_node=od_p.origin_node,
#                                end_node=od_p.destination_node,
#                                route_gdf=route_gdf,
#                                map_tiles="OpenStreetMap.Mapnik",
#                                file_path=filepath)


fig,_ = ox.plot_graph_route(
    subgraph,
    shortest_route,
    node_size=5,
    bgcolor='white',
    route_color='blue',
    edge_color="black",
    node_color="black",
    route_linewidth=4,
    edge_linewidth=0.5,
    show=False,
    close=False
)
fig.savefig("bug_testing/ume_graph_route.png", bbox_inches='tight')
plt.close(fig)

od_p.create_orientation_plot("bug_testing/ume_od_pair_orientation.png")
od_p.create_alignment_plot("bug_testing/ume_od_pair_alignment_corr.png")
od_p_env_dist = od_p_data['bearings_dist_env_directed_dist_weighted']
od_p_route_dist = od_p_data['bearings_dist_od_directed_dist']
#print(od_p_env_dist)
#print(od_p_route_dist)
#print("\n Normalized: \n")
#env = od_p_env_dist / np.sum(od_p_env_dist)
#route = od_p_route_dist  / np.sum(od_p_env_dist)
#print(env)
#print(route)

