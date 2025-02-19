import math
import osmnx as ox
import map_plotting as mp


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


def plot_origin_orientation(origin_graph,filepath):
    city_name = origin_graph.city_name

    Graph = origin_graph.graph
    Graph = ox.convert.to_undirected(Graph)
    fig, ax = ox.plot.plot_orientation(Graph, num_bins=36)
    fig.savefig(filepath)
    env_entropy = ox.bearing.orientation_entropy(Graph,num_bins=36,weight="length",min_length=10)
    max_nats = math.log(36)
    min_nats = math.log(4)
    orientation_order = 1 - ((env_entropy - min_nats) / (max_nats - min_nats))**2
    print(f"orientation order: {orientation_order} for {city_name}")


