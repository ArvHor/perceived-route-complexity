import math
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import wasserstein_distance
from scipy.signal import find_peaks, correlation_lags, correlate

def COT_sample_distance(route_bearings, env_bearings, typeOfData="Angles"):

    if typeOfData == "UnitInt":
            route_bearings = route_bearings % 1
            env_bearings = env_bearings % 1
        elif typeOfData == "Radian":
            route_bearings = (route_bearings % (2 * math.pi)) / (2 * math.pi)
            env_bearings = (env_bearings % (2 * math.pi)) / (2 * math.pi)
        elif typeOfData == "Angles":
            route_bearings = (route_bearings % 360) / 360
            env_bearings = (env_bearings % 360) / 360
        else:
            raise ValueError("Type of Data has to be specified as \"UnitInt\", \"Radian\" or \"Angles\".")
    # Combine and order samples
    combined_sample = np.concatenate([route_bearings, env_bearings])
    order_of_samples = np.argsort(combined_sample)
    combined_sample_sorted = np.concatenate([combined_sample[order_of_samples], [1]])

    k = len(combined_sample_sorted) - 1

    # Calculate diffCDFs
    diffCDFs_part1 = np.repeat(1/len(route_bearings), len(route_bearings))
    diffCDFs_part2 = np.repeat(-1/len(env_bearings), len(env_bearings))
    diffCDFs_parts = np.concatenate([diffCDFs_part1, diffCDFs_part2])
    diffCDFs_ordered = diffCDFs_parts[order_of_samples]
    diffCDFs = np.cumsum(diffCDFs_ordered)

    # Order diffCDFs
    order_diffCDFs = np.argsort(diffCDFs)
    sorted_diffCDFs = diffCDFs[order_diffCDFs]

    # Calculate weighting
    combined_sample_with_1 = combined_sample_sorted
    weighting = combined_sample_with_1[1:(k+1)] - combined_sample_with_1[:k]

    # Find the median level
    cumsum_weighting = np.cumsum(weighting[order_diffCDFs])
    if len(np.where(cumsum_weighting >= 0.5)[0]) == 0:
        levMed_index = 0
    else:
        levMed_index = np.where(cumsum_weighting >= 0.5)[0][0]
    levMed = sorted_diffCDFs[levMed_index]

    # Final calculation
    result = np.sum(np.abs(diffCDFs - levMed) * weighting)

    return result



def get_crosscorrelation_alignment(route_dist, env_dist):
    route_dist = wrap_dist(route_dist)
    env_dist = wrap_dist(env_dist)
    max_index = np.argmax(route_dist)
    route_dist = center_around_index(route_dist, max_index)
    env_dist = center_around_index(env_dist, max_index)
    route_dist = route_dist / np.sum(route_dist)
    env_dist = env_dist / np.sum(env_dist)
    cross_correlation = np.correlate(env_dist, route_dist, mode="full")

    lag = np.argmax(cross_correlation) - (len(route_dist) - 1)
    max_correlation = cross_correlation[lag + (len(route_dist) - 1)]

    return lag, max_correlation


def get_EMD_alignment(route_dist, env_dist):
    route_dist = route_dist / np.sum(route_dist)
    env_dist = env_dist / np.sum(env_dist)
    EMD_alignment = wasserstein_distance(env_dist, route_dist)
    return EMD_alignment


def circular_cross_correlation(route, env):
    """Calculates the circular cross-correlation using FFT."""
    route = np.asarray(route)
    env = np.asarray(env)

    fft_route = np.fft.fft(route)
    fft_env = np.fft.fft(env)

    fft_env_conj = np.conj(fft_env)

    fft_product = fft_route * fft_env_conj

    result = np.fft.ifft(fft_product)

    result = np.abs(result)

    n = len(result)

    result = np.roll(result, -(n // 2))

    return result

def circular_distance(a, b, n):
    """Compute the minimum circular distance between indices a and b in array of length n."""
    return min(abs(a - b), n - abs(a - b))

def wrap_dist(dist):
    half = len(dist) // 2
    a = dist[half:]
    b = dist[:half]
    folded = np.empty(half, dtype=int)

    for i in range(0, half):
        folded[i] = a[i] + b[i]

    return folded

def center_around_index(dist, max_index):
    """Roll the distribution so that the specified index is at the center."""
    center = len(dist) // 2
    shift = center - max_index
    dist = np.roll(dist, shift)
    return dist


def wrap_and_center_dists(od_dist, env_dist):
    """Wraps and centers the distributions around the maximum index."""
    od_dist = wrap_dist(od_dist)
    env_dist = wrap_dist(env_dist)
    max_index = np.argmax(od_dist)
    od_dist = center_around_index(od_dist, max_index)
    env_dist = center_around_index(env_dist, max_index)
    

    return od_dist, env_dist


def get_peaks_alignment(od_dist, env_dist):
    # Normalize
    od_dist, env_dist = wrap_and_center_dists(od_dist, env_dist)
    od_dist = od_dist / np.sum(od_dist)
    env_dist = env_dist / np.sum(env_dist)

    # Find peaks
    route_peaks, _ = find_peaks(od_dist)
    env_peaks, properties = find_peaks(env_dist, prominence=0.01)

    if len(route_peaks) == 0 or len(env_peaks) == 0:
        peak_alignment = {
            "route_main_peak": None,
            "closest_env_peak": None,
            "strongest_env_peak": None,
            "closest_env_peak_value": None,
            "strongest_env_peak_value": None,
            "distance_to_closest": None,
            "distance_to_strongest": None,
        }
        return peak_alignment

    # Take the main peak in route_dist (highest)
    route_main_peak = route_peaks[np.argmax(od_dist[route_peaks])]

    # Find the closest peak in env_dist
    closest_env_peak = env_peaks[np.argmin(np.abs(env_peaks - route_main_peak))]

    # Find the strongest peak in env_dist
    strongest_env_peak = env_peaks[np.argmax(env_dist[env_peaks])]

    # Calculate distances
    distance_to_closest = circular_distance(
        route_main_peak, closest_env_peak, len(env_dist)
    )
    distance_to_strongest = circular_distance(
        route_main_peak, strongest_env_peak, len(env_dist)
    )
    peak_alignment = {
        "route_main_peak": route_main_peak,
        "closest_env_peak": closest_env_peak,
        "strongest_env_peak": strongest_env_peak,
        "closest_env_peak_value": env_dist[closest_env_peak],
        "strongest_env_peak_value": env_dist[strongest_env_peak],
        "distance_to_closest": distance_to_closest,
        "distance_to_strongest": distance_to_strongest,
    }
    return peak_alignment

def to_orientation_distribution(dist):
    """Transforms a distribution to a circular distribution."""
    half = len(dist) // 2
    a = dist[half:]
    b = dist[:half]
    orientation_dist = np.empty(half, dtype=int)
    for i in range(0, half):
        orientation_dist[i] = a[i] + b[i]
    return orientation_dist


def get_weighted_crosscorr(crosscorr, rotation_steps, weighting="relative", proximity_weight=1):

    max_orientation_frequency = np.max(crosscorr)
    max_steps = np.max(np.abs(rotation_steps))
    print("max steps:", max_steps)
    print("max orientation frequency:", max_orientation_frequency)
    weighted_orientation_dist = np.zeros_like(crosscorr, dtype=float)
    print("Rotation steps:", rotation_steps)
    for i, frequency in enumerate(crosscorr):
        steps = np.abs(rotation_steps[i])
        rotation_distance = proximity_weight * (steps / max_steps)
        if weighting == "relative":
            penalty = max_orientation_frequency - rotation_distance
            weighted_frequency = frequency - penalty
            weighted_orientation_dist[i] = weighted_frequency
        elif weighting == "absolute":
            weighted_frequency = frequency * (1 - rotation_distance)
            weighted_orientation_dist[i] = weighted_frequency
    return weighted_orientation_dist


def crosscorrelate_alignment(
    route_dist, env_dist, proximity_weight=1, method="direct", mode="same",weighting="relative"
):

    crosscorr = correlate(env_dist, route_dist, mode="same", method="direct")
    rotation_steps = correlation_lags(len(env_dist), len(route_dist), mode="same" )
    zero_rotation_index = len(env_dist) // 2
    zero_rotation_frequency = crosscorr[zero_rotation_index]
    rotation_steps_to_strongest = rotation_steps[np.argmax(crosscorr)]
    strongest_rotation_frequency = np.max(crosscorr)

    strongest_correlation = {
        "cross_correlation": crosscorr,
        "rotation_steps": rotation_steps,
        "zero_rotation_index": zero_rotation_index,
        "zero_rotation_frequency": zero_rotation_frequency,
        "steps_to_strongest": rotation_steps_to_strongest,
        "strongest_frequency": strongest_rotation_frequency,
    }
    weighted_corr = get_weighted_crosscorr(crosscorr,rotation_steps, weighting, proximity_weight)
    rotation_steps_to_closest_strongest = rotation_steps[np.argmax(weighted_corr)]
    closest_strongest_frequency = np.max(weighted_corr)
    closest_strongest_correlation = {
        "closest_strongest_cross_correlation": weighted_corr,
        "steps_to_closest_strongest": rotation_steps_to_closest_strongest,
        "closest_strongest_frequency": closest_strongest_frequency,
    }

    return strongest_correlation, closest_strongest_correlation
