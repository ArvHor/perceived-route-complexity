import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import wasserstein_distance
from scipy.signal import find_peaks, correlation_lags, correlate


def get_crosscorrelation_alignment(route_dist, env_dist):
    route_dist = fold_dist(route_dist)
    env_dist = fold_dist(env_dist)
    max_index = np.argmax(route_dist)
    route_dist = roll_to_max(route_dist, max_index)
    env_dist = roll_to_max(env_dist, max_index)
    route_dist = route_dist / np.sum(route_dist)
    env_dist = env_dist / np.sum(env_dist)
    cross_correlation = np.correlate(env_dist, route_dist, mode="full")

    lag = np.argmax(cross_correlation) - (len(route_dist) - 1)
    max_correlation = cross_correlation[lag + (len(route_dist) - 1)]

    return lag, max_correlation


def get_cosine_similarity_alignment(route_dist, env_dist):
    route_dist = route_dist / np.sum(route_dist)
    env_dist = env_dist / np.sum(env_dist)
    cosine_similarity = np.dot(env_dist, route_dist) / (
        np.linalg.norm(env_dist) * np.linalg.norm(route_dist)
    )
    return cosine_similarity


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


def fold_dist(dist):
    half = len(dist) // 2
    a = dist[half:]
    b = dist[:half]
    folded = np.empty(half, dtype=int)

    for i in range(0, half):
        folded[i] = a[i] + b[i]

    return folded


def circular_distance(a, b, n):
    """Compute the minimum circular distance between indices a and b in array of length n."""
    return min(abs(a - b), n - abs(a - b))


def roll_to_max(dist, max_index):
    """Roll the distribution so that the specified index is at the center."""
    center = len(dist) // 2
    shift = center - max_index
    dist = np.roll(dist, shift)
    return dist


def find_peaks_alignment(route_dist, env_dist):
    # Normalize

    route_dist = fold_dist(route_dist)
    env_dist = fold_dist(env_dist)

    route_dist = route_dist / np.sum(route_dist)
    env_dist = env_dist / np.sum(env_dist)

    max_index = np.argmax(route_dist)
    route_dist = roll_to_max(route_dist, max_index)
    env_dist = roll_to_max(env_dist, max_index)
    # Find peaks
    route_peaks, _ = find_peaks(route_dist)
    env_peaks, properties = find_peaks(env_dist, prominence=0.01)

    if len(route_peaks) == 0 or len(env_peaks) == 0:
        peak_alignment = {
            "route_main_peak": "None",
            "closest_env_peak": "None",
            "strongest_env_peak": "None",
            "closest_env_peak_value": "None",
            "strongest_env_peak_value": "None",
            "distance_to_closest": "None",
            "distance_to_strongest": "None",
        }
        return peak_alignment

    # Take the main peak in route_dist (highest)
    route_main_peak = route_peaks[np.argmax(route_dist[route_peaks])]

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


def find_optimal_correlation(
    route_dist, env_dist, proximity_weight=1, method="direct", mode="same"
):
    route_dist = fold_dist(route_dist)
    env_dist = fold_dist(env_dist)

    max_index = np.argmax(route_dist)
    route_dist = roll_to_max(route_dist, max_index)
    env_dist = roll_to_max(env_dist, max_index)

    route_dist = route_dist / np.sum(route_dist)
    env_dist = env_dist / np.sum(env_dist)

    corr = correlate(env_dist,route_dist, mode=mode, method=method)

    corr_lags = correlation_lags(len(env_dist), len(env_dist), mode=mode)
    #print("Correlation lags:", corr_lags)
    max_lag = np.max(np.abs(corr_lags))

    weighted_corr = corr.copy()

    # Apply the proximity penalty to the circular cross-correlation

    max_correlation = np.max(corr)

    for i, strength in enumerate(corr):
        lag = np.abs(corr_lags[i])
        penalty = (max_correlation * (lag / max_lag)) * proximity_weight
        weighted_correlation = strength - penalty
        weighted_corr[i] = weighted_correlation

    # Calculate the circular lag of the strongest correlation
    strongest_lag = corr_lags[np.argmax(corr)]
    strongest_correlation = corr[np.argmax(corr)]

    closest_strongest_lag = corr_lags[np.argmax(weighted_corr)]
    closest_strongest_correlation = weighted_corr[np.argmax(weighted_corr)]

    cos_dist = cosine(route_dist, env_dist)
    euc_dist = euclidean(route_dist, env_dist)

    shifted_env_dist = np.roll(env_dist, closest_strongest_lag)
    shifted_cos_dist = cosine(route_dist, shifted_env_dist)
    shifted_euc_dist = euclidean(route_dist, shifted_env_dist)

    zero_lag_index = np.where(corr_lags == 0)[0][0]
    zero_lag_strength = corr[zero_lag_index]

    strongest_correlation = {
        "zero_lag": zero_lag_strength,
        "lag": strongest_lag,
        "env_index": len(env_dist) // 2 + strongest_lag,
        "strength": strongest_correlation,
        "cross_correlation": corr,
    }

    closest_strongest_correlation = {
        "lag": closest_strongest_lag,
        "strength": closest_strongest_correlation,
        "env_index": len(env_dist) // 2 + closest_strongest_lag,
        "cross_correlation": weighted_corr,
        "cosine_distance": cos_dist,
        "euclidean_distance": euc_dist,
        "shifted_cosine_distance": shifted_cos_dist,
        "shifted_euclidean_distance": shifted_euc_dist,
    }

    return strongest_correlation, closest_strongest_correlation
