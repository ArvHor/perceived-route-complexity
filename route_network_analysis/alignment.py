import logging
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cosine, euclidean
from scipy.signal import correlate, correlation_lags
import numpy as np

# Local modules
from .performance_tracker import PerformanceTracker, track_performance


def get_len(array_type):
    return len(array_type)


alignment_metrics = {
    "n_count": get_len,
}

alignment_tracker = PerformanceTracker(output_file="alignment_performance.json")


def get_crosscorrelation_alignment(route_dist, env_dist):
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


# @track_performance(alignment_tracker, metrics_funcs=alignment_metrics)
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


def fold_dist(dist: np.array):
    half = len(dist) // 2
    a = dist[half:]
    b = dist[:half]
    folded = np.empty(half, dtype=int)

    for i in range(0, half):
        folded[i] = a[i] + b[i]
    print(f"len of folded {len(folded)}")
    return folded


# @track_performance(alignment_tracker, metrics_funcs=alignment_metrics)
def find_optimal_correlation(route_dist, env_dist, proximity_weight=1):
    """Calculate optimal correlation and distances between distributions.

    Returns:
        tuple: (strongest_correlation, closest_strongest_correlation, cosine_distance, euclidean_distance)
    """
    route_dist = fold_dist(route_dist)
    env_dist = fold_dist(env_dist)
    route_dist = route_dist / np.sum(route_dist)
    env_dist = env_dist / np.sum(env_dist)
    route_dist_len = len(route_dist)
    env_dist_len = len(env_dist)
    corr = correlate(route_dist, env_dist, mode="full", method="direct")
    corr_lags = np.arange(0, route_dist_len)
    max_lag = len(corr_lags)
    weighted_corr = corr.copy()
    logging.info(
        f"Max absolute lag: {max_lag}, min lag in corr: {min(corr_lags)}, max lag in corr: {max(corr_lags)}"
    )
    logging.info(f"{corr}")

    # Apply the proximity penalty to the circular cross-correlation

    max_correlation = np.max(corr)

    logging.info(f"Max: {corr}")

    for i in corr_lags:
        strength = corr[i]
        penalty = (max_correlation * (i / max_lag)) * proximity_weight
        weighted_correlation = strength - penalty
        weighted_corr[i] = weighted_correlation

    # Calculate the circular lag of the strongest correlation
    strongest_lag = np.argmax(corr)
    strongest_correlation = corr[strongest_lag]
    # Adjust the lag to be within the range of -max_lag to max_lag
    if strongest_lag >= max_lag:
        strongest_lag -= len(corr)

    # Calculate the circular lag of the closest strongest correlation
    closest_strongest_lag = corr_lags[np.argmax(weighted_corr)]
    closest_strongest_correlation = weighted_corr[closest_strongest_lag]

    cos_dist = cosine(route_dist, env_dist)
    euc_dist = euclidean(route_dist, env_dist)

    # Shift env_dist by the optimal lag
    shifted_env_dist = np.roll(env_dist, closest_strongest_lag)
    shifted_cos_dist = cosine(route_dist, shifted_env_dist)
    shifted_euc_dist = euclidean(route_dist, shifted_env_dist)

    strongest_correlation = {
        "lag": strongest_lag,
        "strength": corr[np.argmax(corr)],
        "cross_correlation": corr,
    }

    closest_strongest_correlation = {
        "lag": closest_strongest_lag,
        "strength": np.argmax(weighted_corr),
        "cross_correlation": weighted_corr,
        "cosine_distance": cos_dist,
        "euclidean_distance": euc_dist,
        "shifted_cosine_distance": shifted_cos_dist,
        "shifted_euclidean_distance": shifted_euc_dist,
    }

    return strongest_correlation, closest_strongest_correlation
