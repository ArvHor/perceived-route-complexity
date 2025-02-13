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

def get_turn_count(turn_list):
    turns = 0
    for turn in turn_list:
        #print(turn)
        if "turn" in turn:
            #print("turn found")
            turns += 1
    return turns

def add_routes(route_df,nr):

   
    route_df["origin_node_lat"] = None
    route_df["origin_node_lon"] = None

    route_df["destination_node_lat"] = None
    route_df["destination_node_lon"] = None

    route_df["simplest_path_nodes"] = None
    route_df["shortest_path_nodes"] = None
    route_df["graph_path"] = None

    temp_g = None
    temp_graph_path = None
    row_n = len(route_df)
    for i, row in route_df.iterrows():
        if i % 500==0:
            print(f"on row {i} out of {row_n}")

        origin_node = row['origin_node']
        destination_node = row['destination_node']
        graph_path = f"local_graphs_randnode{nr}_simplified_drive/{row['city_name']}_{nr}_graph.graphml"

        if temp_graph_path == graph_path:
            
            G = temp_g
        else:
            print(f"new city, using stored graph for {row['city_name']}")
            edge_types = {"decision_complexity": float,"length": float}
            G = ox.load_graphml(graph_path,edge_dtypes=edge_types)
            temp_g = G
            temp_graph_path = graph_path
        
        route_df.at[i,"graph_path"] = graph_path
        route_df.at[i,"origin_node_lat"] = G.nodes[origin_node]['y']
        route_df.at[i,"origin_node_lon"] = G.nodes[origin_node]['x']
        route_df.at[i,"destination_node_lat"] = G.nodes[destination_node]['y']
        route_df.at[i,"destination_node_lon"] = G.nodes[destination_node]['x']
        route_df.at[i,"simplest_path_nodes"] = simplest["nodes"]
        route_df.at[i,"shortest_path_nodes"] = shortest["nodes"]
        shortest = route.get_dijkstra_path(G, origin_node, destination_node, "length")
        simplest = route.get_dijkstra_path(G, origin_node, destination_node, "decision_complexity")

        

    route_df.to_csv(f'od_pair_data/local_graphs_randnode{nr}_simplified_drive_withpaths.csv', index=False)
    return route_df

def add_route_complexity(df, nr):

    print(f"adding route complexity for {nr} with {len(df)} rows")
    df["simplest_path_complexity"] = None
    df["shortest_path_complexity"] = None
    df["shortest_path_turn_count"] = None
    df["simplest_path_turn_count"] = None
    df["simplest_path_nodes"] = None
    df["shortest_path_nodes"] = None
    df["graph_path"] = None
    df["origin_node_lat"] = None
    df["origin_node_lon"] = None
    df["destination_node_lat"] = None
    df["destination_node_lon"] = None


    temp_g = None
    temp_graph_path = None
    row_n = len(df)
    for i, row in df.iterrows():
        if i % 1000==0:
            print(f"on row {i} out of {row_n} head of df")
            
        try:
            origin_node = int(row['origin_node'])
            destination_node = int(row['destination_node'])
            graph_path = f"local_graphs_randnode{nr}_simplified_drive/{row['city_name']}_{nr}_graph.graphml"

            if temp_graph_path == graph_path:
                
                G = temp_g
            else:
                logging.info(f"new city, using stored graph for {row['city_name']}")
                edge_types = {"decision_complexity": float,"length": float}
                G = ox.load_graphml(graph_path,edge_dtypes=edge_types)
                print(f" graph has edges: {len(G.edges)}")
                temp_g = G
                temp_graph_path = graph_path
            try:
                shortest = route.get_dijkstra_path(G, origin_node, destination_node, "length")
                shortest_nodes = shortest["nodes"]
                simplest_nodes = route.find_simplest_path(G, origin_node, destination_node)

                simplest_edges = list(nx.utils.pairwise(simplest_nodes))
                shortest_edges = list(nx.utils.pairwise(shortest_nodes))
                
                simplest_complexity,complexities_simplest,simplest_turns = route.get_route_complexity_optimized(G,simplest_edges)
                shortest_complexity,complexities_shortest,shortest_turns = route.calculate_route_complexity_post_hoc(G,shortest_edges)
                simplest_path_length = route.static_get_edges_sum(G,simplest_edges,"length")
                shortest_path_length = route.static_get_edges_sum(G,shortest_edges,"length")
            except Exception as e:
                logging.error(f"Error calculating route complexity for {nr} on row {i} origin node {origin_node}, destination node {destination_node}",exc_info=True)
                print(f"Error calculating route complexity for {nr} on row {i}")
                print(e)
                continue
        
            shortest_n_turns = get_turn_count(shortest_turns)
            simplest_n_turns = get_turn_count(simplest_turns)
            #simplest_complexity = simplest["path_weights_sum"]

            df.at[i,"shortest_path_complexity"] = shortest_complexity
            df.at[i,"simplest_path_complexity"] = simplest_complexity
            df.at[i,"shortest_path_turn_count"] = shortest_n_turns
            df.at[i,"simplest_path_turn_count"] = simplest_n_turns
            df.at[i,"simplest_path_length"] = simplest_path_length
            df.at[i,"shortest_path_length"] = shortest_path_length
            df.at[i,"graph_path"] = graph_path
            df.at[i,"origin_node_lat"] = G.nodes[origin_node]['y']
            df.at[i,"origin_node_lon"] = G.nodes[origin_node]['x']
            df.at[i,"destination_node_lat"] = G.nodes[destination_node]['y']
            df.at[i,"destination_node_lon"] = G.nodes[destination_node]['x']
            df.at[i,"simplest_path_nodes"] = simplest_nodes
            df.at[i,"shortest_path_nodes"] = shortest["nodes"]
        except Exception as e:
            logging.error(f"Error adding route complexity for {nr} on row {i}",exc_info=True)
            print(f"Error adding route complexity for {nr} on row {i}")
            print(e)
    print(f"done with adding route complexity for {nr}, df length: {df.head()}")
    return df


def normalize_complexity(df):
    
    shortest_max = df['shortest_path_complexity'].max()
    simplest_max = df['simplest_path_complexity'].max()

    max_complexity = max(shortest_max,simplest_max)
    print(f"max complexity: {df['shortest_path_complexity'].max()} sum of columns: {df['shortest_path_complexity'].sum()}, mean: {df['shortest_path_complexity'].mean()}, median: {df['shortest_path_complexity'].median()}")
    # now for the shortest path
    print(f"max complexity: {df['simplest_path_complexity'].max()} sum of columns: {df['simplest_path_complexity'].sum()}, mean: {df['simplest_path_complexity'].mean()}, median: {df['simplest_path_complexity'].median()}")
    
    
    df['simplest_path_complexity'] = df['simplest_path_complexity'] / max_complexity
    df['shortest_path_complexity'] = df['shortest_path_complexity'] / max_complexity
    print(f"max complexity: {df['shortest_path_complexity'].max()} sum of columns: {df['shortest_path_complexity'].sum()}, mean: {df['shortest_path_complexity'].mean()}, median: {df['shortest_path_complexity'].median()}")
    # now for the shortest path
    print(f"max complexity: {df['simplest_path_complexity'].max()} sum of columns: {df['simplest_path_complexity'].sum()}, mean: {df['simplest_path_complexity'].mean()}, median: {df['simplest_path_complexity'].median()}")
    
    return df

def fix_and_combine(od_pair_data):
    #print(df1.columns)

    #
    # od_pair_data['cross_correlation'] = od_pair_data['cross_correlation'].abs()
    median_value = od_pair_data['environment_orientation_order'].median()
    od_pair_data['gridlike_median'] = od_pair_data['environment_orientation_order'].apply(lambda x: 'above_median' if x > median_value else 'below_median')
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
    labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
    od_pair_data['gridlike_group'] = pd.cut(od_pair_data['environment_orientation_order'], bins=bins, labels=labels, include_lowest=True)

    Q1 = od_pair_data[['simplest_path_length', 'shortest_path_length']].quantile(0.25)
    Q3 = od_pair_data[['simplest_path_length', 'shortest_path_length']].quantile(0.75)
    IQR = Q3 - Q1

    def is_outlier(row):
        return (
            (row['shortest_path_length'] < (Q1['shortest_path_length'] - 1.5 * IQR['shortest_path_length'])) or
            (row['shortest_path_length'] > (Q3['shortest_path_length'] + 1.5 * IQR['shortest_path_length']))
        )

    od_pair_data['length_outliers'] = od_pair_data.apply(is_outlier, axis=1)


    #od_pair_data.to_csv('od_pair_data/data.csv', index=False)
    return od_pair_data


def save_all_paths(df1,df2,df3):
    df1_dict = {"df":df1,"nr":1}
    df2_dict = {"df":df2,"nr":2}
    df3_dict = {"df":df3,"nr":3}

    df_dicts = [df1_dict,df2_dict,df3_dict]


    num_processes = multiprocessing.cpu_count()  # Use the number of CPU cores
    print(f"Number of processes: {num_processes}")
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use pool.apply_async for asynchronous processing
        results = []
        for df_dict in df_dicts:
            result = pool.apply_async(add_routes, args=(df_dict["df"], df_dict["nr"]))
            results.append(result)
        # Wait for all processes to complete
        for result in results:
            result.get()
            combined_df = pd.concat([result.get() for result in results], ignore_index=True)
            combined_df.to_csv('od_pair_data/route_pathinfo.csv', index=False)



def add_complexity_normalize_combine(df1,df2,df3):
    df1_dict = {"df":df1,"nr":1}
    df2_dict = {"df":df2,"nr":2}
    df3_dict = {"df":df3,"nr":3}

    df_dicts = [df1_dict,df2_dict,df3_dict]


    num_processes = multiprocessing.cpu_count()  # Use the number of CPU cores
    print(f"Number of processes: {num_processes}")
    results = []
    combined_df = None
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use pool.apply_async for asynchronous processing
        
        for df_dict in df_dicts:
            print(f"adding route complexity for {df_dict['nr']}")
            result = pool.apply_async(add_route_complexity, args=(df_dict["df"], df_dict["nr"]))
            results.append(result)
        
        dfs_with_complexity = [result.get() for result in results]

    combined_df = pd.concat(dfs_with_complexity, ignore_index=True)

    print(f"combined_df length: {len(combined_df)}")
    combined_df = normalize_complexity(combined_df)
    combined_df = fix_and_combine(combined_df)
    combined_df.to_csv('od_pair_data/od_pair_data.csv', index=False)


    #data.to_csv('od_pair_data/od_pair_data.csv', index=False)
df1 = pd.read_csv('od_pair_data/2024_01_30/local_graphs_randnode1_simplified_drive_B.csv')
df2 = pd.read_csv('od_pair_data/2024_01_30/local_graphs_randnode2_simplified_drive_B.csv')
df3 = pd.read_csv('od_pair_data/2024_01_30/local_graphs_randnode3_simplified_drive_B.csv')
print(f"df1 length: {len(df1)}, columns: {df1.columns}")
add_complexity_normalize_combine(df1,df2,df3)