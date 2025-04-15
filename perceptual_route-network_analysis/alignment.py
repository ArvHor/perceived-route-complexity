import numpy as np
from scipy.stats import wasserstein_distance
import logging
import numpy as np



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


def find_optimal_correlation(route_dist,env_dist,proximity_weight):
    import logging
    if route_dist is None or len(route_dist) == 0 or env_dist is None or len(env_dist) == 0:

        logging.error(f"route_dist or env_dist are None or empty. route_dist: {route_dist}, env_dist: {env_dist}")
        return None, None, None, None

    route_dist = route_dist / np.sum(route_dist)
    env_dist = env_dist / np.sum(env_dist)
    max_len = max(len(route_dist), len(env_dist))

    circ_cross_corr = circular_cross_correlation(route_dist, env_dist)

    strongest_correlation = {
        "lag": np.argmax(circ_cross_corr),
        "strength": circ_cross_corr[np.argmax(circ_cross_corr)],
    }
    #logging.error(f"Circular cross-correlation: {circ_cross_corr}")

    n_circ_cross_corr = np.abs(circ_cross_corr) / np.max(np.abs(circ_cross_corr))
    weighted_n_circ_cross_corr = n_circ_cross_corr
    max_abs_lag = max_len // 2

    for i in range(max_abs_lag):
        lag = i if i < max_abs_lag else i - max_len
        strength = n_circ_cross_corr[i]
        penalty = proximity_weight * (abs(lag) / max_abs_lag)
        weighted_correlation = strength - penalty
        weighted_n_circ_cross_corr[i] = weighted_correlation


    closest_strongest_correlation = {
        "lag": np.argmax(weighted_n_circ_cross_corr),
        "strength": weighted_n_circ_cross_corr[np.argmax(weighted_n_circ_cross_corr)],
    }

    return strongest_correlation, closest_strongest_correlation,
