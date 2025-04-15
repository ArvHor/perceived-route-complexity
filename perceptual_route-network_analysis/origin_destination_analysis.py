import osmnx as ox
import numpy as np

def get_route_direction_bearing_dist(fwd,bwd, perp_fwd=None, perp_bwd=None):


    if perp_fwd and perp_bwd:
        bearings = [fwd, bwd, perp_fwd, perp_bwd]
    else:
        bearings = [fwd, bwd]

    num_bins = 36
    num_split_bins = num_bins * 2
    split_bin_edges = np.arange(num_split_bins + 1) * 360 / num_split_bins

    split_bin_counts, split_bin_edges = np.histogram(bearings, bins=split_bin_edges)
    split_bin_counts = np.roll(split_bin_counts, 1)
    bin_counts = split_bin_counts[::2] + split_bin_counts[1::2]
    bin_centers = split_bin_edges[range(0, num_split_bins - 1, 2)]

    return bin_counts


def get_od_cardinal_direction(G, origin, destination):
    origin_point = (G.nodes[origin]['x'],G.nodes[origin]['y'])
    destination_point = (G.nodes[destination]['x'],G.nodes[destination]['y'])
    bearing = ox.bearing.calculate_bearing(origin_point[0], origin_point[1], destination_point[0], destination_point[1])

    # Define cardinal direction ranges
    if 337.5 <= bearing < 360 or 0 <= bearing < 22.5:
        cardinal_direction = "N"
    elif 22.5 <= bearing < 67.5:
        cardinal_direction = "NE"
    elif 67.5 <= bearing < 112.5:
        cardinal_direction = "E"
    elif 112.5 <= bearing < 157.5:
        cardinal_direction = "SE"
    elif 157.5 <= bearing < 202.5:
        cardinal_direction = "S"
    elif 202.5 <= bearing < 247.5:
        cardinal_direction = "SW"
    elif 247.5 <= bearing < 292.5:
            cardinal_direction = "W"
    elif 292.5 <= bearing < 337.5:
        cardinal_direction = "NW"

    return cardinal_direction

def get_od_pair_subgraph(G,map_bbox=None,polygon=None):
    if map_bbox:
        subgraph = ox.truncate.truncate_graph_bbox(G=G, bbox=map_bbox, truncate_by_edge=True)
    elif polygon:
        subgraph = ox.truncate.truncate_graph_polygon(G=G, polygon=polygon, truncate_by_edge=True)
    return subgraph