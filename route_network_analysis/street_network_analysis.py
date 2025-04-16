import itertools
import math
import numpy as np
import numpy.typing as npt
import networkx as nx
from typing import overload
from osmnx import projection
from warnings import warn
import scipy
# Local modules
from . import geo_util

def add_deviation_from_prototypical_weights(G):
    max_weight = 0
    for u in G.nodes():
        if G.out_degree(u) >= 2:
            successor_bearings = get_bearings_to_successors(G,u)
            deviation_weight = calculate_deviation_from_prototypical(successor_bearings)
            if deviation_weight > max_weight:
                max_weight = deviation_weight
            for u,v,k in G.out_edges(u,keys=True):
                G.edges[u,v,k]["deviation_from_prototypical"] = deviation_weight
        else:
            for u,v,k in G.out_edges(u,keys=True):
                G.edges[u,v,k]["deviation_from_prototypical"] = 0

    return G, max_weight

def add_instruction_equivalent_weights(G):
    max_weight = 0
    for u in G.nodes():
        if G.out_degree(u) >= 2:
            successor_bearings = get_bearings_to_successors(G,u)
            instruction_weight = calculate_instruction_equivalent(successor_bearings)
            if instruction_weight > max_weight:
                max_weight = instruction_weight
            for u,v,k in G.out_edges(u,keys=True):
                G.edges[u,v,k]["instruction_equivalent"] = instruction_weight
        else:
            for u,v,k in G.out_edges(u,keys=True):
                G.edges[u,v,k]["instruction_equivalent"] = 0
    return G,max_weight

def add_node_degree_weights(G):
    max_degree = 0
    for u in G.nodes():
        n_degree = G.out_degree(u)
        if n_degree > max_degree:
            max_degree = n_degree

        for u, v, k in G.out_edges(u, keys=True):
            G.edges[u, v, k]["node_degree"] = n_degree

    return G, max_degree


def get_bearings_to_successors(G, node):
    successors = list(G.successors(node))
    bearing_list = []
    for successor in successors:
        fwd_azimuth = geo_util.get_azimuth(G, node, successor)
        bearing_list.append(fwd_azimuth)

    return bearing_list


def calculate_instruction_equivalent(bearing_list):
    """How many turns at a decision point can be described with the same linguistic label?"""
    bearing_difference_list = []
    if len(bearing_list) > 1:
        for bearing_a, bearing_b in itertools.combinations(bearing_list, 2):
            bearing_difference = bearing_b - bearing_a
            bearing_difference = bearing_difference % 360

            if bearing_difference > 180:
                bearing_difference -= 360
            elif bearing_difference < -180:
                bearing_difference += 360

            bearing_difference_list.append(bearing_difference)

        if bearing_difference_list:
            zero_to_ninety = len([bearing for bearing in bearing_difference_list if 0 < bearing < 90])
            ninety_to_oneeighty = len([bearing for bearing in bearing_difference_list if 90 < bearing < 180])
            minus_ninety_to_zero = len([bearing for bearing in bearing_difference_list if -90 < bearing < 0])
            minus_oneeighty_to_minus_ninety = len(
                [bearing for bearing in bearing_difference_list if -180 < bearing < -90])
            max_count = max(zero_to_ninety, ninety_to_oneeighty, minus_ninety_to_zero, minus_oneeighty_to_minus_ninety,
                            1)
            return max_count
        else:
            return 1
    else:
        return 1


def calculate_deviation_from_prototypical(bearing_list):
    deviation_list = []
    if len(bearing_list) > 1:
        for bearing_a, bearing_b in itertools.combinations(bearing_list, 2):
            bearing_difference = bearing_a - bearing_b
            bearing_difference = bearing_difference % 360
            if bearing_difference > 180:
                bearing_difference -= 360
            elif bearing_difference < -180:
                bearing_difference += 360
            deviation = calculate_deviation(bearing_difference)
            deviation_list.append(deviation)
        # print(deviation_list)
        avg_deviations = sum(deviation_list) / len(deviation_list)
    else:
        avg_deviations = 0
    return avg_deviations


def calculate_deviation(bearing):
    if bearing < 0:
        if bearing >= -90:
            a = -90 - bearing
            deviation = abs(min(a, bearing))
            return deviation
        elif bearing >= -180:
            a = -180 - bearing
            b = -90 - bearing
            deviation = abs(min(a, b))
            return deviation
    elif bearing > 0:
        if bearing <= 90:
            a = 90 - bearing
            deviation = abs(min(a, bearing))
            return deviation
        elif bearing <= 180:
            a = 180 - bearing
            b = 90 - bearing
            deviation = abs(min(a, b))
            return deviation
    return 0



"""


The functions below are derived from OMSNX osmnx.bearing._bearing_distribution.
https://osmnx.readthedocs.io/en/latest/user-reference.html#osmnx.bearing.orientation_entropy
Used under the MIT license.
"""
def get_orientation_order(entropy,num_bins=36):
    max_nats = math.log(num_bins)
    min_nats = math.log(4)
    orientation_order = 1 - ((entropy - min_nats) / (max_nats - min_nats)) ** 2
    return orientation_order


def orientation_entropy(
    G: nx.MultiGraph | nx.MultiDiGraph,
    *,
    num_bins: int = 36,
    min_length: float = 0,
    weight: str | None = None,
) -> float:
    """
    Calculate graph's orientation entropy.

    Orientation entropy is the Shannon entropy of the graphs' edges' bearings
    across evenly spaced bins. Ignores self-loop edges as their bearings are
    undefined. If `G` is a MultiGraph, all edge bearings will be bidirectional
    (ie, two reciprocal bearings per undirected edge). If `G` is a
    MultiDiGraph, all edge bearings will be directional (ie, one bearing per
    directed edge).

    For more info see: Boeing, G. 2019. "Urban Spatial Order: Street Network
    Orientation, Configuration, and Entropy." Applied Network Science, 4 (1),
    67. https://doi.org/10.1007/s41109-019-0189-1

    Parameters
    ----------
    G
        Unprojected graph with `bearing` attributes on each edge.
    num_bins
        Number of bins. For example, if `num_bins=36` is provided, then each
        bin will represent 10 degrees around the compass.
    min_length
        Ignore edges with "length" attributes less than `min_length`. Useful
        to ignore the noise of many very short edges.
    weight
        If None, apply equal weight for each bearing. Otherwise, weight edges'
        bearings by this (non-null) edge attribute. For example, if "length"
        is provided, each edge's bearing observation will be weighted by its
        "length" attribute value.

    Returns
    -------
    entropy
        The orientation entropy of `G`.
    """
    # check if we were able to import scipy
    if scipy is None:  # pragma: no cover
        msg = "scipy must be installed as an optional dependency to calculate entropy."
        raise ImportError(msg)
    bin_counts, _ = bearings_distribution(G, num_bins, min_length, weight)
    entropy: float = scipy.stats.entropy(bin_counts)
    return entropy



def extract_edge_bearings(
    G: nx.MultiGraph | nx.MultiDiGraph,
    min_length: float,
    weight: str | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Extract graph's edge bearings.

    Ignores self-loop edges as their bearings are undefined. If `G` is a
    MultiGraph, all edge bearings will be bidirectional (ie, two reciprocal
    bearings per undirected edge). If `G` is a MultiDiGraph, all edge bearings
    will be directional (ie, one bearing per directed edge). For example, if
    an undirected edge has a bearing of 90 degrees then we will record
    bearings of both 90 degrees and 270 degrees for this edge.

    Parameters
    ----------
    G
        Unprojected graph with `bearing` attributes on each edge.
    min_length
        Ignore edges with `length` attributes less than `min_length`. Useful
        to ignore the noise of many very short edges.
    weight
        If None, apply equal weight for each bearing. Otherwise, weight edges'
        bearings by this (non-null) edge attribute. For example, if "length"
        is provided, each edge's bearing observation will be weighted by its
        "length" attribute value.

    Returns
    -------
    bearings, weights
        The edge bearings of `G` and their corresponding weights.
    """
    if projection.is_projected(G.graph["crs"]):  # pragma: no cover
        msg = "Graph must be unprojected to analyze edge bearings."
        raise ValueError(msg)
    bearings = []
    weights = []
    for u, v, data in G.edges(data=True):
        # ignore self-loops and any edges below min_length
        if u != v and data["length"] >= min_length:
            bearings.append(data["bearing"])
            weights.append(data[weight] if weight is not None else 1.0)

    # drop any nulls
    bearings_array = np.array(bearings)
    weights_array = np.array(weights)
    keep_idx = ~np.isnan(bearings_array)
    bearings_array = bearings_array[keep_idx]
    weights_array = weights_array[keep_idx]
    if nx.is_directed(G):
        msg = (
            "`G` is a MultiDiGraph, so edge bearings will be directional (one per "
            "edge). If you want bidirectional edge bearings (two reciprocal bearings "
            "per edge), pass a MultiGraph instead. Use `convert.to_undirected`."
        )
        warn(msg, category=UserWarning, stacklevel=2)
        return bearings_array, weights_array
    # for undirected graphs, add reverse bearings
    bearings_array = np.concatenate([bearings_array, (bearings_array - 180) % 360])
    weights_array = np.concatenate([weights_array, weights_array])
    return bearings_array, weights_array


def bearings_distribution(
    G: nx.MultiGraph | nx.MultiDiGraph,
    num_bins: int,
    min_length: float,
    weight: str | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute distribution of bearings across evenly spaced bins.

    Prevents bin-edge effects around common values like 0 degrees and 90
    degrees by initially creating twice as many bins as desired, then merging
    them in pairs. For example, if `num_bins=36` is provided, then each bin
    will represent 10 degrees around the compass, with the first bin
    representing 355 degrees to 5 degrees.

    Parameters
    ----------
    G
        Unprojected graph with `bearing` attributes on each edge.
    num_bins
        Number of bins for the bearing histogram.
    min_length
        Ignore edges with `length` attributes less than `min_length`. Useful
        to ignore the noise of many very short edges.
    weight
        If None, apply equal weight for each bearing. Otherwise, weight edges'
        bearings by this (non-null) edge attribute. For example, if "length"
        is provided, each edge's bearing observation will be weighted by its
        "length" attribute value.

    Returns
    -------
    bin_counts, bin_centers
        Counts of bearings per bin and the bins' centers in degrees. Both
        arrays are of length `num_bins`.
    """
    # Split bins in half to prevent bin-edge effects around common values.
    # Bins will be merged in pairs after the histogram is computed. The last
    # bin edge is the same as the first (i.e., 0 degrees = 360 degrees).
    num_split_bins = num_bins * 2
    split_bin_edges = np.arange(num_split_bins + 1) * 360 / num_split_bins

    bearings, weights = extract_edge_bearings(G, min_length, weight)
    split_bin_counts, split_bin_edges = np.histogram(
        bearings,
        bins=split_bin_edges,
        weights=weights,
    )

    # Move last bin to front, so eg 0.01 degrees and 359.99 degrees will be
    # binned together. Then combine counts from pairs of split bins.
    split_bin_counts = np.roll(split_bin_counts, 1)
    bin_counts = split_bin_counts[::2] + split_bin_counts[1::2]

    # Every other edge of the split bins is the center of a merged bin.
    bin_centers = split_bin_edges[range(0, num_split_bins - 1, 2)]
    return bin_counts, bin_centers
