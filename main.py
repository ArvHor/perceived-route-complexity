import ast
import pandas as pd
from origin_graph import origin_graph
from origin_graph_set import origin_graph_set
import logging
import os
import multiprocessing
import osmnx as ox
logging.basicConfig(level=logging.info, format='%(asctime)s - %(levelname)s - %(message)s',filename='app.log', filemode='w')

error_handler = logging.FileHandler('error.log', mode='w')
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Add the error handler to the root logger
logging.getLogger().addHandler(error_handler)

def process_single_graph(row, folder_path, remove_parallel, distance_from_point, network_type):
    """
    Creates and processes a single graph based on a row from the DataFrame.
    """
    lat, lon = ast.literal_eval(row["node1_latlon"])
    file_path = f"{folder_path}/{row['city_name']}_1_graph.graphml"
    imgpath = f"{folder_path}/{row['city_name']}_1_graph.png"
    if folder_path == "local_graphs_randnode2_simplified_drive":
        lat, lon = ast.literal_eval(row["node2_latlon"])
        file_path = f"{folder_path}/{row['city_name']}_2_graph.graphml"
        imgpath = f"{folder_path}/{row['city_name']}_2_graph.png"
    elif folder_path == "local_graphs_randnode3_simplified_drive":
        lat, lon = ast.literal_eval(row["node3_latlon"])
        file_path = f"{folder_path}/{row['city_name']}_3_graph.graphml"
        imgpath = f"{folder_path}/{row['city_name']}_3_graph.png"
    origin_info = {
        "city_name": row["city_name"],
        "region_name": row["region"],
        "country_name": row["country"],
        "network_type": "drive"
    }
    
    
    #print(file_path)
    if os.path.exists(file_path):
        #print(f"Graph for {row['city_name']} already exists. Skipping...")
        if origin_info["city_name"] == "Barcelona":
            print(f"lat lon for {row['city_name']}: {lat}, {lon}")
        return

    try:
        print(f"Creating graph for {row['city_name']} with origin point: {lat}, {lon}")
        o_graph = origin_graph(origin_point=(lat, lon), origin_info=origin_info, network_type=network_type,
                               remove_parallel=remove_parallel, distance_from_point=distance_from_point)
        o_graph.add_weights(weightstring="decision_complexity")
        o_graph.add_weights(weightstring="deviation_from_prototypical")
        o_graph.add_weights(weightstring="instruction_equivalent")
        o_graph.add_weights(weightstring="node_degree")
        ox.plot.plot_graph(o_graph.graph, save=True,show=False, filepath=imgpath)
        o_graph.save_graph(file_path)
    except Exception:
        logging.error(f"Error creating graph for {row['city_name']}", exc_info=True)

def create_graphs_parallel(folder_path: str, remove_parallel: bool = True, distance_from_point: int = 6000,
                          network_type: str = "drive"):
    """
    Creates graphs in parallel using multiprocessing.
    """
    
    location_df = pd.read_csv('parameter_data/boeing_locations_3node_sample.csv')
    num_processes = multiprocessing.cpu_count()  # Use the number of CPU cores
    print(f"Number of processes: {num_processes}")
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use pool.apply_async for asynchronous processing
        results = []
        for index, row in location_df.iterrows():
            result = pool.apply_async(process_single_graph,
                                      args=(row, folder_path, remove_parallel, distance_from_point, network_type))
            results.append(result)

        # Wait for all processes to complete
        for result in results:
            result.get()  # This will also raise any exceptions that occurred in the child process



def create_origin_graph_set(folder_path:str,remove_parallel:bool = True,distance_from_point:int = 6000,network_type:str = "drive"):
    

    location_df = pd.read_csv('parameter_data/boeing_locations_3node_sample.csv')

    origin_graph_list = []
    for index,row in location_df.iterrows():

        origin_info = {
            "city_name": row["city_name"],
            "region_name": row["region"],
            "country_name": row["country"],
            "network_type":"drive"
        }
        if row["city_name"] == "Manila":
            continue
        if folder_path == "local_graphs_randnode1_simplified_drive":
            graph_path = f"{folder_path}/{row['city_name']}_1_graph.graphml"
            lat,lon = ast.literal_eval(row["node1_latlon"])

        elif folder_path == "local_graphs_randnode2_simplified_drive":
            graph_path = f"{folder_path}/{row['city_name']}_2_graph.graphml"
            lat,lon = ast.literal_eval(row["node2_latlon"])

        elif folder_path == "local_graphs_randnode3_simplified_drive":
            graph_path = f"{folder_path}/{row['city_name']}_3_graph.graphml"
            lat,lon = ast.literal_eval(row["node3_latlon"])

        
        try:
            o_graph = origin_graph(origin_point=(lat,lon),origin_info = origin_info,network_type=network_type,
                                   remove_parallel=True,distance_from_point=6000,
                                   load_graphml=graph_path)
            o_graph.add_weights(weightstring="node_degree")
            
            o_graph.ensure_data_types()
            o_graph.remove_infinite_edges()
            # Print all the attributes of a random edge
            
            #print(f"Attributes of a random edge in {row['city_name']} graph: {random_edge}")
            
            origin_graph_list.append(o_graph)
            print(f'Graph {row["city_name"]}, removed parallel edged: {len(o_graph.removed_parallel_edges)}, removed infinite edges: {len(o_graph.removed_inf_edges)} edges')
        except:
            logging.error(f"Error creating graph for {row['city_name']}",exc_info=True)
        
    og_set = origin_graph_set(origin_graph_list)
    og_set.add_normalized_weights()
    og_set.save_graphs()
    og_set.save_pickle(f'graph_sets/{folder_path}.pkl')
    return og_set


def add_od_pairs(folder_path,min_radius:int,max_radius:int,og_set=None):
    
    if og_set is None:
        og_set = origin_graph_set.load_pickle(f'graph_sets/{folder_path}.pkl')
    og_set.add_length_fix()
    og_set.add_all_od_pairs(min_radius=min_radius,max_radius=max_radius)
    od_pair_data = og_set.get_all_od_pair_data_parallel()
    od_pair_data.to_csv(f'od_pair_data/{folder_path}_B.csv')
    og_set.save_pickle(f'graph_sets/{folder_path}.pkl')


def update_od_pairs(folder_path,og_set=None):
    if og_set is None:
        og_set = origin_graph_set.load_pickle(f'graph_sets/{folder_path}.pkl')
    od_pair_data = og_set.get_all_od_pair_data_parallel()
    od_pair_data.to_csv(f'od_pair_data/{folder_path}.csv')
    og_set.save_pickle(f'graph_sets/{folder_path}.pkl')

def driving_graphs():
    #folderpath = "local_graphs_randnode1_simplified_drive"
    #create_graphs_parallel(folder_path=folderpath,remove_parallel=True,distance_from_point=6000,network_type="drive")
    #og_set = create_origin_graph_set(folder_path=folderpath,network_type="drive")
    #add_od_pairs(folder_path=folderpath,min_radius=3800,max_radius=4200)
    #update_od_pairs(folder_path=folderpath)
    
    #folderpath = "local_graphs_randnode2_simplified_drive"
    #create_graphs_parallel(folder_path=folderpath,remove_parallel=True,distance_from_point=6000,network_type="drive")
    #og_set = create_origin_graph_set(folder_path=folderpath,network_type="drive")
    #add_od_pairs(folder_path=folderpath,min_radius=3800,max_radius=4200)
    #update_od_pairs(folder_path=folderpath)

    folderpath = "local_graphs_randnode3_simplified_drive"
    create_graphs_parallel(folder_path=folderpath,remove_parallel=True,distance_from_point=6000,network_type="drive")
    og_set = create_origin_graph_set(folder_path=folderpath,network_type="drive")
    add_od_pairs(folder_path=folderpath,min_radius=3800,max_radius=4200)

def walking_graphs():
    create_graphs_parallel(folder_path="local_graphs_simplified_walk",remove_parallel=True,distance_from_point=6000,network_type="walk")
    create_origin_graph_set(folder_path="local_graphs_simplified_walk",network_type="walk")



if __name__ == '__main__':
    driving_graphs()