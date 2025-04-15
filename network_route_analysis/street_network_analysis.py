
import heapq
import itertools
import math

import osmnx as ox
import pyproj
import logging

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
        fwd_azimuth = get_azimuth(G, node, successor)
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

def get_orientation_order(entropy,num_bins=36):
    max_nats = math.log(num_bins)
    min_nats = math.log(4)
    orientation_order = 1 - ((entropy - min_nats) / (max_nats - min_nats)) ** 2
    return orientation_order