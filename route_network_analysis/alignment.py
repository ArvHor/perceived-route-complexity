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

@track_performance(alignment_tracker, metrics_funcs=alignment_metrics)
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

@track_performance(alignment_tracker, metrics_funcs=alignment_metrics)
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

    
    weighted_circ_cross_corr = np.zeros(max_len)
    max_abs_lag = max_len // 2

    for i in range(max_abs_lag):
        lag = i if i < max_abs_lag else i - max_len
        strength = circ_cross_corr[i]
        penalty = proximity_weight * (abs(lag) / max_abs_lag)
        weighted_correlation = strength - penalty
        weighted_circ_cross_corr[i] = weighted_correlation

    cos_dist = cosine(route_dist, env_dist)
    euc_dist = euclidean(route_dist, env_dist)

    # Shift env_dist by the optimal lag
    shifted_env_dist = np.roll(env_dist, int(np.argmax(weighted_circ_cross_corr)))
    shifted_cos_dist = cosine(route_dist, shifted_env_dist)
    shifted_euc_dist = euclidean(route_dist, shifted_env_dist)

    strongest_correlation = {
        "lag": np.argmax(circ_cross_corr),
        "strength": circ_cross_corr[np.argmax(circ_cross_corr)],
    }

    closest_strongest_correlation = {
        "lag": np.argmax(weighted_circ_cross_corr),
        "strength": weighted_circ_cross_corr[np.argmax(weighted_circ_cross_corr)],
        "cosine_distance": cos_dist,
        "euclidean_distance": euc_dist,
        "shifted_cosine_distance": shifted_cos_dist,
        "shifted_euclidean_distance": shifted_euc_dist,
    }



    return strongest_correlation, closest_strongest_correlation,
