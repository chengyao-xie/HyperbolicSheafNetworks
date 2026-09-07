import numpy as np

ACX_EMBEDDING_DIM = 3
ACX_DELAY = 1
ACX_SCALES = 33
ACX_THEILER_WINDOW = 2
ACX_EPSILON_MIN = 0.001
ACX_EPSILON_MAX = 1.0
ACX_SYMMETRIZATION = 'mean'
ACX_FEATURE_MODE = 'dual_symmetric_multiscale_reliability'
ACX_SCALE_BANDS = 3


def normalize_signal_matrix(data_matrix):
    matrix = np.asarray(data_matrix, dtype=float)
    if matrix.shape[0] <= matrix.shape[1]:
        return matrix
    return matrix.T


def takens_embedding(signal):
    signal = np.asarray(signal, dtype=float).reshape(-1)
    length = signal.shape[0] - (ACX_EMBEDDING_DIM - 1) * ACX_DELAY
    if length <= 1:
        return signal.reshape(-1, 1)
    columns = [
        signal[i * ACX_DELAY:i * ACX_DELAY + length]
        for i in range(ACX_EMBEDDING_DIM)
    ]
    return np.stack(columns, axis=1)


def pairwise_distances(points):
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def epsilon_percents():
    return np.exp(np.linspace(
        np.log(ACX_EPSILON_MIN), np.log(ACX_EPSILON_MAX), ACX_SCALES))


def shifted_neighbor_mask(reference_dist, epsilon, eliminate_span=0):
    time_count = reference_dist.shape[0]
    mask = reference_dist[1:time_count] <= epsilon
    for t in range(time_count - 1):
        left = max(0, t + 1 - eliminate_span)
        right = min(time_count, t + 2 + eliminate_span)
        mask[t, left:right] = False

    shifted_mask = np.zeros((time_count - 1, time_count), dtype=bool)
    for t in range(time_count - 1):
        shifted = np.where(mask[t])[0] - 1
        shifted = shifted[(shifted >= 0) & (shifted < time_count)]
        shifted_mask[t, shifted] = True
    return shifted_mask


def batched_global_delta(distance_stack, reference_dist, eps_percents, eliminate_span=0):
    time_count = reference_dist.shape[0]
    max_dist = float(np.max(reference_dist))
    if not np.isfinite(max_dist) or max_dist <= 1e-12 or time_count <= 2:
        return None, None, None

    epsilons = eps_percents * max_dist
    effect_distances = distance_stack[:, :time_count - 1, :time_count]
    per_scale = np.full((distance_stack.shape[0], len(epsilons), time_count - 1), np.nan, dtype=float)
    scale_support = np.zeros(len(epsilons), dtype=float)
    fallback = None

    for eps_idx in range(len(epsilons) - 1, -1, -1):
        mask = shifted_neighbor_mask(reference_dist[:time_count, :time_count], epsilons[eps_idx], eliminate_span)
        counts = mask.sum(axis=1)
        valid = counts > 0
        scale_support[eps_idx] = np.mean(valid)
        values = np.full((distance_stack.shape[0], time_count - 1), np.nan, dtype=float)
        if np.any(valid):
            sums = np.einsum('ntk,tk->nt', effect_distances, mask, optimize=True)
            values[:, valid] = sums[:, valid] / counts[valid]
        if fallback is not None:
            missing = ~valid
            values[:, missing] = fallback[:, missing]
        per_scale[:, eps_idx, :] = values
        fallback = values

    finite_count = np.sum(np.isfinite(per_scale), axis=2)
    finite_sum = np.nansum(per_scale, axis=2)
    global_delta = np.full(finite_sum.shape, np.nan, dtype=float)
    np.divide(finite_sum, finite_count, out=global_delta, where=finite_count > 0)
    return epsilons, global_delta, scale_support


def _longest_contiguous_indices(mask):
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return indices
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, splits)
    return max(groups, key=len)


def _huber_line_fit(x, y, max_iters=8):
    """Small dependency-free robust linear fit using Huber IRLS."""
    design = np.column_stack([x, np.ones_like(x)])
    weights = np.ones_like(y)
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(max_iters):
        residual = y - design @ coefficients
        mad = np.median(np.abs(residual - np.median(residual)))
        scale = max(1.4826 * mad, 1e-8)
        cutoff = 1.345 * scale
        abs_residual = np.abs(residual)
        weights = np.ones_like(abs_residual)
        outside = abs_residual > cutoff
        weights[outside] = cutoff / np.maximum(abs_residual[outside], 1e-12)
        weighted_design = design * np.sqrt(weights[:, None])
        weighted_y = y * np.sqrt(weights)
        updated = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
        if np.max(np.abs(updated - coefficients)) < 1e-8:
            coefficients = updated
            break
        coefficients = updated
    return coefficients, weights


def robust_slope_from_global_delta(epsilons, global_delta, scale_support=None,
                                   min_points=5, min_support=0.05):
    """Estimate a robust ACX slope on a supported scale interval.

    Returns the robust slope and a reliability score in ``[0, 1]``.  The
    reliability combines robust R², logarithmic scale coverage and neighbour
    support, so noisy or poorly supported edges are down-weighted later.
    """
    finite = np.isfinite(global_delta) & np.isfinite(epsilons) & (epsilons > 0)
    if scale_support is not None:
        finite &= np.asarray(scale_support) >= min_support
    selected = _longest_contiguous_indices(finite)
    if selected.size < min_points:
        return 0.0, 0.0

    x = np.log(epsilons[selected])
    y = np.asarray(global_delta)[selected]
    if np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return 0.0, 0.0
    x_centered = x - np.mean(x)
    coefficients, weights = _huber_line_fit(x_centered, y)
    slope = float(coefficients[0])
    prediction = np.column_stack([x_centered, np.ones_like(x_centered)]) @ coefficients
    weighted_mean = np.average(y, weights=weights)
    ss_res = np.sum(weights * (y - prediction) ** 2)
    ss_tot = np.sum(weights * (y - weighted_mean) ** 2)
    r_squared = np.clip(1.0 - ss_res / max(ss_tot, 1e-12), 0.0, 1.0)
    full_span = np.ptp(np.log(epsilons[np.isfinite(epsilons) & (epsilons > 0)]))
    span_coverage = np.clip(np.ptp(x) / max(full_span, 1e-12), 0.0, 1.0)
    support_quality = 1.0
    if scale_support is not None:
        support_quality = np.clip(np.mean(np.asarray(scale_support)[selected]), 0.0, 1.0)
    reliability = r_squared * np.sqrt(span_coverage * support_quality)
    if not np.isfinite(slope) or not np.isfinite(reliability):
        return 0.0, 0.0
    return slope, float(np.clip(reliability, 0.0, 1.0))


def multiscale_slopes_from_global_delta(epsilons, global_delta,
                                        scale_support=None):
    """Fit independent robust slopes on contiguous small/mid/large scales."""
    finite = np.isfinite(global_delta) & np.isfinite(epsilons) & (epsilons > 0)
    if scale_support is not None:
        finite &= np.asarray(scale_support) >= 0.05
    selected = _longest_contiguous_indices(finite)
    slopes = np.zeros(ACX_SCALE_BANDS, dtype=float)
    reliability = np.zeros(ACX_SCALE_BANDS, dtype=float)
    if selected.size < ACX_SCALE_BANDS * 3:
        return slopes, reliability

    for band_idx, band_indices in enumerate(
            np.array_split(selected, ACX_SCALE_BANDS)):
        band_support = None
        if scale_support is not None:
            band_support = np.asarray(scale_support)[band_indices]
        slopes[band_idx], reliability[band_idx] = robust_slope_from_global_delta(
            np.asarray(epsilons)[band_indices],
            np.asarray(global_delta)[band_indices],
            scale_support=band_support,
            min_points=3,
        )
    return slopes, reliability


def continuity_scaling_multiscale_matrices(data_matrix):
    """Return aggregate and per-scale directed ACX scores/reliability."""
    signals = normalize_signal_matrix(data_matrix)
    embeddings = [
        takens_embedding(signal)
        for signal in signals
    ]
    min_len = min(emb.shape[0] for emb in embeddings)
    embeddings = [emb[:min_len] for emb in embeddings]
    distances = [pairwise_distances(emb) for emb in embeddings]
    distance_stack = np.stack(distances, axis=0)
    eps_percents = epsilon_percents()

    n_rois = len(distances)
    multiscale_scores = np.zeros((n_rois, n_rois, ACX_SCALE_BANDS), dtype=float)
    multiscale_reliability = np.zeros_like(multiscale_scores)
    for reference_idx in range(n_rois):
        epsilons, global_deltas, scale_support = batched_global_delta(
            distance_stack,
            distances[reference_idx],
            eps_percents,
            eliminate_span=ACX_THEILER_WINDOW,
        )
        if epsilons is None:
            continue
        for effect_idx in range(n_rois):
            if effect_idx == reference_idx:
                continue
            slopes, reliability = multiscale_slopes_from_global_delta(
                epsilons,
                global_deltas[effect_idx],
                scale_support=scale_support,
            )
            multiscale_scores[effect_idx, reference_idx] = slopes
            multiscale_reliability[effect_idx, reference_idx] = reliability

    multiscale_scores = np.nan_to_num(
        multiscale_scores, nan=0.0, posinf=0.0, neginf=0.0)
    multiscale_reliability = np.nan_to_num(
        multiscale_reliability, nan=0.0, posinf=0.0, neginf=0.0)
    weighted_sum = np.sum(multiscale_scores * multiscale_reliability, axis=-1)
    reliability_sum = np.sum(multiscale_reliability, axis=-1)
    aggregate_scores = np.zeros((n_rois, n_rois), dtype=float)
    np.divide(
        weighted_sum,
        reliability_sum,
        out=aggregate_scores,
        where=reliability_sum > 1e-12,
    )
    aggregate_reliability = np.mean(multiscale_reliability, axis=-1)
    np.fill_diagonal(aggregate_scores, 0.0)
    np.fill_diagonal(aggregate_reliability, 0.0)
    return (
        aggregate_scores,
        aggregate_reliability,
        multiscale_scores,
        multiscale_reliability,
    )
