import numpy as np
from scipy.signal import correlate
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import wasserstein_distance


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
    print(f"len of folded {len(folded)}")
    return folded


def find_optimal_correlation(route_dist, env_dist, proximity_weight=1):
    """ """
    route_dist = fold_dist(route_dist)
    env_dist = fold_dist(env_dist)
    route_dist = route_dist / np.sum(route_dist)
    env_dist = env_dist / np.sum(env_dist)
    route_dist_len = len(route_dist)
    corr = correlate(route_dist, env_dist, mode="same", method="direct")
    corr_lags = np.arange(0, route_dist_len)

    max_lag = len(corr_lags)
    weighted_corr = corr.copy()

    # Apply the proximity penalty to the circular cross-correlation

    max_correlation = np.max(corr)

    for i in corr_lags:
        strength = corr[i]
        penalty = (max_correlation * (i / max_lag)) * proximity_weight
        weighted_correlation = strength - penalty
        weighted_corr[i] = weighted_correlation

    # Calculate the circular lag of the strongest correlation
    strongest_lag = np.argmax(corr)
    strongest_correlation = corr[strongest_lag]
    """
    print("len of corr", len(corr))
    print("len of weighted corr", len(weighted_corr))
    print("strongest_lag", strongest_lag)
    print("strongest_correlation", strongest_correlation)
    """
    # Adjust the lag to be within the range of -max_lag to max_lag
    if strongest_lag >= max_lag:
        strongest_lag -= len(corr)

    # Calculate the circular lag of the closest strongest correlation
    closest_strongest_lag = np.argmax(weighted_corr)
    closest_strongest_correlation = weighted_corr[closest_strongest_lag]
    # print("weighted correlation:", weighted_corr)
    #   print("lag of closest strongest corr:", closest_strongest_lag)
    #   print("closest strongest correlation:", closest_strongest_correlation)

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
