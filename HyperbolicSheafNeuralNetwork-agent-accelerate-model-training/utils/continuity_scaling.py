import numpy as np


# 整理 ROI 时间序列方向为 ROI x time。
def normalize_signal_matrix(data_matrix):
    matrix = np.asarray(data_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Brain data matrix must be 2-dimensional.")
    if matrix.shape[0] <= matrix.shape[1]:
        return matrix
    return matrix.T


# 构造 Takens 延迟嵌入。
def takens_embedding(signal, embedding_dim=3, delay=1):
    signal = np.asarray(signal, dtype=float).reshape(-1)
    length = signal.shape[0] - (embedding_dim - 1) * delay
    if length <= 1:
        return signal.reshape(-1, 1)
    columns = [signal[i * delay:i * delay + length] for i in range(embedding_dim)]
    return np.stack(columns, axis=1)


# 计算相空间点之间的欧氏距离矩阵。
def pairwise_distances(points):
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


# 生成 ACX 使用的指数尺度比例。
def epsilon_percents(scale_count=33, epsilon_min=0.001, epsilon_max=1.0):
    if scale_count < 3:
        raise ValueError("scale_count must be at least 3.")
    return np.exp(np.linspace(np.log(epsilon_min), np.log(epsilon_max), scale_count))


# 计算一个方向上的 ACX 全局 delta 曲线。
def acx_global_delta(effect_dist, reference_dist, eps_percents, eliminate_span=0):
    time_count = min(effect_dist.shape[0], reference_dist.shape[0])
    max_dist = float(np.max(reference_dist))
    if not np.isfinite(max_dist) or max_dist <= 1e-12 or time_count <= 2:
        return None, None

    epsilons = eps_percents * max_dist
    delta = np.full((len(epsilons), time_count - 1), np.nan, dtype=float)
    all_indices = np.arange(time_count)

    for t in range(time_count - 1):
        for eps_idx in range(len(epsilons) - 1, -1, -1):
            neighbors = reference_dist[t + 1] <= epsilons[eps_idx]
            left = max(0, t + 1 - eliminate_span)
            right = min(time_count, t + 2 + eliminate_span)
            neighbors[left:right] = False

            shifted = all_indices[neighbors] - 1
            shifted = shifted[(shifted >= 0) & (shifted < time_count)]
            if shifted.size:
                delta[eps_idx, t] = np.mean(effect_dist[t, shifted])
            elif eps_idx + 1 < len(epsilons):
                delta[eps_idx, t] = delta[eps_idx + 1, t]

    global_delta = np.nanmean(delta, axis=1)
    return epsilons, global_delta


# 从 ACX 全局 delta 曲线估计连续缩放斜率。
def acx_scaling_slope(effect_dist, reference_dist, eps_percents, eliminate_span=0):
    epsilons, global_delta = acx_global_delta(
        effect_dist,
        reference_dist,
        eps_percents,
        eliminate_span=eliminate_span,
    )
    if epsilons is None:
        return 0.0

    finite = np.isfinite(global_delta) & np.isfinite(epsilons) & (epsilons > 0)
    if finite.sum() < 3:
        return 0.0

    log_eps = np.log(epsilons[finite])
    values = global_delta[finite]
    local_slopes = np.diff(values) / np.maximum(np.diff(log_eps), 1e-12)
    if local_slopes.size < 2:
        return 0.0

    keep_count = max(1, int(round(local_slopes.size / 2)))
    keep = np.argsort(local_slopes)[-keep_count:]
    selected = np.unique(np.concatenate([keep, keep + 1]))
    if selected.size < 2:
        return 0.0

    design = np.column_stack([log_eps[selected], np.ones(selected.size)])
    slope, _ = np.linalg.lstsq(design, values[selected], rcond=None)[0]
    return float(slope) if np.isfinite(slope) else 0.0


# 构造参考流形在某个尺度下的移位邻域掩码。
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


# 针对同一个参考 ROI，批量计算所有目标 ROI 的 GlobalDelta。
def batched_global_delta(distance_stack, reference_dist, eps_percents, eliminate_span=0):
    time_count = reference_dist.shape[0]
    max_dist = float(np.max(reference_dist))
    if not np.isfinite(max_dist) or max_dist <= 1e-12 or time_count <= 2:
        return None, None

    epsilons = eps_percents * max_dist
    effect_distances = distance_stack[:, :time_count - 1, :time_count]
    per_scale = np.full((distance_stack.shape[0], len(epsilons), time_count - 1), np.nan, dtype=float)
    fallback = None

    for eps_idx in range(len(epsilons) - 1, -1, -1):
        mask = shifted_neighbor_mask(reference_dist[:time_count, :time_count], epsilons[eps_idx], eliminate_span)
        counts = mask.sum(axis=1)
        valid = counts > 0
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
    return epsilons, global_delta


# 从一条 GlobalDelta 曲线估计 ACX 斜率。
def slope_from_global_delta(epsilons, global_delta):
    finite = np.isfinite(global_delta) & np.isfinite(epsilons) & (epsilons > 0)
    if finite.sum() < 3:
        return 0.0

    log_eps = np.log(epsilons[finite])
    values = global_delta[finite]
    local_slopes = np.diff(values) / np.maximum(np.diff(log_eps), 1e-12)
    if local_slopes.size < 2:
        return 0.0

    keep_count = max(1, int(round(local_slopes.size / 2)))
    keep = np.argsort(local_slopes)[-keep_count:]
    selected = np.unique(np.concatenate([keep, keep + 1]))
    if selected.size < 2:
        return 0.0

    design = np.column_stack([log_eps[selected], np.ones(selected.size)])
    slope, _ = np.linalg.lstsq(design, values[selected], rcond=None)[0]
    return float(slope) if np.isfinite(slope) else 0.0


# 用 ACX-style 连续缩放构造 ROI 间连接矩阵。
def continuity_scaling_matrix(data_matrix, embedding_dim=3, delay=1, scale_count=33,
                              theiler_window=2, epsilon_min=0.001,
                              epsilon_max=1.0):
    signals = normalize_signal_matrix(data_matrix)

    embeddings = [
        takens_embedding(signal, embedding_dim=embedding_dim, delay=delay)
        for signal in signals
    ]
    min_len = min(emb.shape[0] for emb in embeddings)
    embeddings = [emb[:min_len] for emb in embeddings]
    distances = [pairwise_distances(emb) for emb in embeddings]
    distance_stack = np.stack(distances, axis=0)
    eps_percents = epsilon_percents(
        scale_count=scale_count,
        epsilon_min=epsilon_min,
        epsilon_max=epsilon_max,
    )

    n_rois = len(distances)
    directed_scores = np.zeros((n_rois, n_rois), dtype=float)
    for reference_idx in range(n_rois):
        epsilons, global_deltas = batched_global_delta(
            distance_stack,
            distances[reference_idx],
            eps_percents,
            eliminate_span=theiler_window,
        )
        if epsilons is None:
            continue
        for effect_idx in range(n_rois):
            if effect_idx == reference_idx:
                continue
            directed_scores[effect_idx, reference_idx] = slope_from_global_delta(
                epsilons,
                global_deltas[effect_idx],
            )

    scores = 0.5 * (np.abs(directed_scores) + np.abs(directed_scores.T))
    np.fill_diagonal(scores, 0.0)

    max_score = np.max(scores)
    if max_score > 0:
        scores = scores / max_score
    return scores
