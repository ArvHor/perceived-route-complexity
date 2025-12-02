import osmnx as ox
import numpy as np
import networkx as nx
from typing import Union

# Local modules


def get_od_pair_bearing_dist(fwd, bwd=None, perp_fwd=None, perp_bwd=None):

    if fwd and bwd and perp_fwd and perp_bwd:
        bearings = [fwd, bwd, perp_fwd, perp_bwd]
    elif fwd and bwd:
        bearings = [fwd, bwd]
    elif fwd:
        bearings = [fwd]

    num_bins = 36
    num_split_bins = num_bins * 2
    split_bin_edges = np.arange(num_split_bins + 1) * 360 / num_split_bins
    split_bin_counts, split_bin_edges = np.histogram(bearings, bins=split_bin_edges)
    split_bin_counts = np.roll(split_bin_counts, 1)
    bin_counts = split_bin_counts[::2] + split_bin_counts[1::2]
    bin_centers = split_bin_edges[range(0, num_split_bins - 1, 2)]

    return bin_counts

def get_od_cardinal_direction(G, origin, destination):
    lat1 = G.nodes[origin]["y"]
    lon1 = G.nodes[origin]["x"]
    lat2 = G.nodes[origin]["y"]
    lon2 = G.nodes[origin]["x"]

    bearing = ox.bearing.calculate_bearing(lat1, lon1, lat2, lon2)

    # Define cardinal direction ranges
    if 337.5 <= bearing < 360 or 0 <= bearing < 22.5:
        return "N"
    elif 22.5 <= bearing < 67.5:
        return "NE"
    elif 67.5 <= bearing < 112.5:
        return "E"
    elif 112.5 <= bearing < 157.5:
        return "SE"
    elif 157.5 <= bearing < 202.5:
        return "S"
    elif 202.5 <= bearing < 247.5:
        return "SW"
    elif 247.5 <= bearing < 292.5:
        return "W"
    elif 292.5 <= bearing < 337.5:
        return "NW"


def get_od_pair_subgraph(
    G: nx.Graph, bbox=None, polygon=None
) -> Union[nx.Graph, nx.DiGraph]:
    if bbox:
        return ox.truncate.truncate_graph_bbox(G=G, bbox=bbox, truncate_by_edge=True)
    elif polygon:
        return ox.truncate.truncate_graph_polygon(
            G=G, polygon=polygon, truncate_by_edge=True
        )
    else:
        raise ValueError(
            "Either bbox or polygon must be provided for subgraph extraction."
        )
