from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cosine, euclidean
import numpy as np

# Local modules
from .performance_tracker import PerformanceTracker, track_performance

def get_len(array_type):
    return len(array_type)

alignment_metrics = {
    'n_count': get_len,
}

alignment_tracker = PerformanceTracker(output_file='alignment_performance.json')

def get_crosscorrelation_alignment(route_dist, env_dist):
    route_dist = route_dist / np.sum(route_dist)
    env_dist = env_dist / np.sum(env_dist)
    cross_correlation = np.correlate(env_dist, route_dist, mode='full')

    lag = np.argmax(cross_correlation) - (len(route_dist) - 1)
    max_correlation = cross_correlation[lag + (len(route_dist) - 1)]

    return lag, max_correlation

def get_cosine_similarity_alignment(route_dist, env_dist):
    route_dist = route_dist / np.sum(route_dist)
    env_dist = env_dist / np.sum(env_dist)
    cosine_similarity = np.dot(env_dist, route_dist) / (np.linalg.norm(env_dist) * np.linalg.norm(route_dist))
    return cosine_similarity


def get_EMD_alignment(route_dist, env_dist):
    route_dist = route_dist / np.sum(route_dist)
    env_dist = env_dist / np.sum(env_dist)
    EMD_alignment = wasserstein_distance(env_dist, route_dist)
    return EMD_alignment

#@track_performance(alignment_tracker, metrics_funcs=alignment_metrics)
def circular_cross_correlation(route,env):
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

    result = np.roll(result,-(n // 2))

    return result

#@track_performance(alignment_tracker, metrics_funcs=alignment_metrics)
def find_optimal_correlation(route_dist,env_dist,proximity_weight=1):
    """Calculate optimal correlation and distances between distributions.
    
    Returns:
        tuple: (strongest_correlation, closest_strongest_correlation, cosine_distance, euclidean_distance)
    """
    import logging
    if route_dist is None or len(route_dist) == 0 or env_dist is None or len(env_dist) == 0:

        logging.error(f"route_dist or env_dist are None or empty. route_dist: {route_dist}, env_dist: {env_dist}")
        return None, None, None, None

    route_dist = route_dist / np.sum(route_dist)
    env_dist = env_dist / np.sum(env_dist)
    max_len = max(len(route_dist), len(env_dist))

    circ_cross_corr = circular_cross_correlation(route_dist, env_dist)


    #logging.error(f"Circular cross-correlation: {circ_cross_corr}")

    lag_range = np.arange(len(circ_cross_corr))
    weighted_circ_cross_corr = circ_cross_corr.copy()
    max_abs_lag = max_len // 2
    logging.info(f"Max absolute lag: {max_abs_lag}, min lag in circ_cross_corr: {min(lag_range)}, max lag in circ_cross_corr: {max(lag_range)}")
    for i in lag_range:
        lag = i if i < max_abs_lag else i - max_len
        strength = circ_cross_corr[i]
        penalty = proximity_weight * (abs(lag) / max_abs_lag)
        weighted_correlation = strength - penalty
        weighted_circ_cross_corr[i] = weighted_correlation


    # Calculate the circular lag of the strongest correlation
    strongest_lag = np.argmax(circ_cross_corr)
    strongest_correlation = circ_cross_corr[strongest_lag]
    # Adjust the lag to be within the range of -max_abs_lag to max_abs_lag
    if strongest_lag >= max_abs_lag:
        strongest_lag -= len(circ_cross_corr)

    # Calculate the circular lag of the closest strongest correlation
    closest_strongest_lag = np.argmax(weighted_circ_cross_corr)
    closest_strongest_correlation = weighted_circ_cross_corr[closest_strongest_lag]
    # Adjust the lag to be within the range of -max_abs_lag to max_abs_lag
    if closest_strongest_lag >= max_abs_lag:
        closest_strongest_lag -= len(weighted_circ_cross_corr)


    cos_dist = cosine(route_dist, env_dist)
    euc_dist = euclidean(route_dist, env_dist)

    # Shift env_dist by the optimal lag
    shifted_env_dist = np.roll(env_dist, closest_strongest_lag)
    shifted_cos_dist = cosine(route_dist, shifted_env_dist)
    shifted_euc_dist = euclidean(route_dist, shifted_env_dist)

    strongest_correlation = {
        "lag": strongest_lag,
        "strength": circ_cross_corr[np.argmax(circ_cross_corr)],
    }

    closest_strongest_correlation = {
        "lag": closest_strongest_lag,
        "strength": weighted_circ_cross_corr[np.argmax(weighted_circ_cross_corr)],
        "cosine_distance": cos_dist,
        "euclidean_distance": euc_dist,
        "shifted_cosine_distance": shifted_cos_dist,
        "shifted_euclidean_distance": shifted_euc_dist,
    }



    return strongest_correlation, closest_strongest_correlation,
