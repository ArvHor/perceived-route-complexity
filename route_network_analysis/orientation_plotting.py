import osmnx as ox

import numpy as np

from . import map_plotting as mp

def plot_all_city_routes_html(od_pair_data,city_name):
    
    od_pair_data = od_pair_data[od_pair_data['city_name'] == city_name]
    unique_origins  = od_pair_data['origin_point'].unique()

    all_city_routes = []
    for unique_origin in unique_origins:
        shortest_path_nodes = od_pair_data[od_pair_data['origin_point'] == unique_origin]['shortest_path_nodes'].values[0]
        graph_path = od_pair_data[od_pair_data['origin_point'] == unique_origin]['graph_path'].values[0]
        graph = ox.load_graphml(graph_path)
        route_gdf = ox.routing.route_to_gdf(graph, shortest_path_nodes, weight='length')
        all_city_routes.append(route_gdf)

    mp.plot_all_routes(all_city_routes,"demonstration/barcelona.html",unique_origins)





