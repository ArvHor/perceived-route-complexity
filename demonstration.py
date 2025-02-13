import ast
import numpy as np
import pandas as pd
from origin_graph_set import origin_graph_set
import osmnx as ox
import map_plotting as mp

#df1 = pd.read_csv('od_pair_data/local_graphs_randnode1_simplified_drive.csv')
#df2 = pd.read_csv('od_pair_data/local_graphs_randnode2_simplified_drive.csv')
#df3 = pd.read_csv('od_pair_data/local_graphs_randnode3_simplified_drive.csv')

#Barcelona,"(41.4308308, 2.1570151)","(41.4436003, 2.1620599)","(41.389087, 2.1278867)",España,Europe


def get_city_routes(og_set,city_name):

    city_routes = []
    origin_point = None
    for origin_graph in og_set.origin_graphs:
        #print("now processing origin graph", origin_graph.city_name)
        if origin_graph.city_name == city_name:
            #origin_point = origin_graph.origin_point
            #print(f"found city, origin point: {origin_point}")
            for od_pair in origin_graph.od_pairs:
                if origin_point is None:
                    origin_point = od_pair.origin_point
                    print(f"origin point: {origin_point}")
                route = od_pair.shortest_path.nodes
                route_gdf = ox.routing.route_to_gdf(od_pair.graph, route,weight='length')
                city_routes.append(route_gdf)

    return city_routes,origin_point

def get_city_routes_from_csv(df):
    city_routes = []
    temp_graph_path = None
    temp_graph = None
    origin_points = []
    for i, row in df.iterrows():
        if row['city_name'] != 'Chicago':
            continue
        else:
            #print("found city")
            origin_point = (row['origin_node_lat'],row['origin_node_lon'])
            if origin_point not in origin_points:

                #print(origin_point)
                origin_points.append(origin_point)
                print(f"origin point: {origin_point}")
            route = ast.literal_eval(row['simplest_path_nodes'])
            graphpath = row['graph_path']

            if graphpath != temp_graph_path:
                #print(f"new city, using stored graph for {row['city_name']}")
                graph = ox.load_graphml(graphpath)
                temp_graph = graph
                temp_graph_path = graphpath
            else:
                graph = temp_graph
            #print(graph)
            route_gdf = ox.routing.route_to_gdf(graph,route,weight='length')
            city_routes.append(route_gdf)

    return city_routes,origin_points

def get_all_routes_from_pickle():

    og_set = origin_graph_set.load_pickle('graph_sets/local_graphs_randnode1_simplified_drive.pkl')
    city_routes1,origin_1 = get_city_routes(og_set,'Barcelona')
    #mp.plot_all_routes(graph,city_routes1,"demonstration/barcelona_1.html",origin)
    
    og_set = origin_graph_set.load_pickle('graph_sets/local_graphs_randnode2_simplified_drive.pkl')
    city_routes2,origin_2 = get_city_routes(og_set,'Barcelona')

    og_set = origin_graph_set.load_pickle('graph_sets/local_graphs_randnode3_simplified_drive.pkl')
    city_routes3,origin_3 = get_city_routes(og_set,'Barcelona')

    all_city_routes = city_routes1 + city_routes2 + city_routes3
    all_origins = [origin_1,origin_2,origin_3]
    mp.plot_all_routes(all_city_routes,"demonstration/barcelona.html",all_origins)

def plot_some_odpairs(city_name):
    og_set = origin_graph_set.load_pickle('graph_sets/local_graphs_randnode1_simplified_drive.pkl')
    city_routes = []
    origin_point = None
    for origin_graph in og_set.origin_graphs:
        #print("now processing origin graph", origin_graph.city_name)
        if origin_graph.city_name == city_name:
            #origin_point = origin_graph.origin_point
            #print(f"found city, origin point: {origin_point}")
            n = 0
            for od_pair in origin_graph.od_pairs:
                if origin_point is None:
                    origin_point = od_pair.origin_point
                    print(f"origin point: {origin_point}")
                if n < 5:
                    path = f"demonstration/orientation_plot_{city_name}_{n}.png"
                    od_pair.create_orientation_plot(path)
                    n += 1

    return city_routes,origin_point

def plot_some_odpair_polygons(city_name):
    og_set = origin_graph_set.load_pickle('graph_sets/local_graphs_randnode1_simplified_drive.pkl')
    city_routes = []
    origin_point = None
    for origin_graph in og_set.origin_graphs:
        #print("now processing origin graph", origin_graph.city_name)
        if origin_graph.city_name == city_name:
            #origin_point = origin_graph.origin_point
            #print(f"found city, origin point: {origin_point}")
            n = 0
            for od_pair in origin_graph.od_pairs:
                if origin_point is None:
                    origin_point = od_pair.origin_point
                    print(f"origin point: {origin_point}")
                if n < 5:
                    path = f"demonstration/orientation_plot_{city_name}_{n}.png"
                    plot_od_pair_polygon(od_pair,path)
                    n += 1

    return city_routes,origin_point

def plot_od_pair_polygon(od_pair,path):
    polygon = od_pair.polygon
    origin_point = od_pair.origin_point
    destination_point = od_pair.destination_point


if __name__ == '__main__':
    
    #df = pd.read_csv('od_pair_data/route_pathinfo.csv')
    plot_some_odpairs('Chicago')
    plot_some_odpairs('Baghdad')
    plot_some_odpairs('Barcelona')
    plot_some_odpairs('Beijing')
    plot_some_odpairs('Berlin')
    plot_some_odpairs('Pnom Penh')
    #city_routes,origin = get_city_routes_from_csv(df)
    #mp.plot_all_routes(city_routes,"demonstration/Chicago_simple.html",origin)