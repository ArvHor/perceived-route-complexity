import numpy as np
from scipy.stats import wasserstein_distance

import numpy as np
def get_circular_crosscorrelation_alignment(route_dist,env_dist):
    """
    Calculates alignment using circular cross-correlation and finds peak distances.
    """
    route_dist = route_dist / np.sum(route_dist)
    env_dist = env_dist / np.sum(env_dist)
    max_correlation, lag = find_circular_max_correlation_and_lag(env_dist, route_dist)

    # Normalize the correlation
    normalized_max_correlation = max_correlation / np.max(env_dist)

    # Align distributions
    aligned_route_dist = circular_shift(route_dist, lag)
    
    # Calculate the cosine similarity after aligning the distributions
    cosine_sim = np.dot(env_dist, aligned_route_dist) / (np.linalg.norm(env_dist) * np.linalg.norm(aligned_route_dist))


    return normalized_max_correlation, lag, cosine_sim

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


def circular_shift(arr, shift):
    """Circularly shifts a 1D array."""
    return np.roll(arr, shift)

    
def circular_cross_correlation(a, b):
    """Calculates the circular cross-correlation using FFT."""
    a = np.asarray(a)
    b = np.asarray(b)
    result = np.fft.ifft(np.fft.fft(a) * np.fft.fft(np.conj(b))[::-1]).real
    return np.roll(result, -(len(b) // 2))

def find_circular_max_correlation_and_lag(a, b):
    """Finds the maximum correlation and lag in circular cross-correlation."""
    max_len = max(len(a), len(b))

    circ_cross_corr = circular_cross_correlation(a, b)
    lag = np.argmax(circ_cross_corr)
    if lag >= max_len // 2:
        lag -= max_len
    max_correlation = circ_cross_corr[lag] if lag >= 0 else circ_cross_corr[lag + max_len]
    return max_correlation, lag

def find_strongest_and_closest_correlation(route_dist,env_dist):
    route_dist = route_dist / np.sum(route_dist)
    env_dist = env_dist / np.sum(env_dist)
    max_len = max(len(route_dist), len(env_dist))

    circ_cross_corr = circular_cross_correlation(route_dist, env_dist)

    normalized_circ_cross_corr = circ_cross_corr / np.max(circ_cross_corr)

    lag_range = np.arange(len(normalized_circ_cross_corr))

    scores = []
    max_lag = max_len//2
    for i, lag_index in enumerate(lag_range):
        lag = lag_index if lag_index < max_len // 2 else lag_index - max_len  # Adjust lag
        correlation_strength = normalized_circ_cross_corr[i]
        score = correlation_strength - (abs(lag) / max_lag)
        scores.append(score)

    # 1. Find the strongest correlation (and its lag)
    strongest_correlation_index = np.argmax(normalized_circ_cross_corr)
    max_correlation = normalized_circ_cross_corr[strongest_correlation_index]
    best_lag = lag_range[strongest_correlation_index]
    if best_lag >= max_len // 2:
        best_lag -= max_len

    # 2. Find the lag with the best combined score
    best_score_index = np.argmax(scores)
    best_score = scores[best_score_index]
    best_score_lag = lag_range[best_score_index]
    if best_score_lag >= max_len // 2:
        best_score_lag -= max_len

    return max_correlation, best_lag, best_score, best_score_lag