import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.io

from utils.continuity_scaling import (
    ACX_DELAY,
    ACX_EMBEDDING_DIM,
    ACX_EPSILON_MAX,
    ACX_EPSILON_MIN,
    ACX_SCALE_BANDS,
    ACX_SCALES,
    ACX_THEILER_WINDOW,
    continuity_scaling_multiscale_matrices,
)


def load_mat_file(file_path):
    mat = scipy.io.loadmat(file_path)
    return next((mat[key] for key in mat if not key.startswith('__')), None)


def save_causality_cache(file_path, matrix, reliability, scores, multiscale_reliability):
    scipy.io.savemat(file_path, {
        'data': matrix,
        'reliability': reliability,
        'multiscale_scores': scores,
        'multiscale_reliability': multiscale_reliability,
    })


def load_causality_features(file_path, shape):
    mat = scipy.io.loadmat(file_path)
    reliability = mat.get('reliability')
    if reliability is None or reliability.shape != shape:
        reliability = np.ones(shape, dtype=float)
        np.fill_diagonal(reliability, 0.0)
    expected_shape = shape + (ACX_SCALE_BANDS,)
    scores = mat.get('multiscale_scores')
    multiscale_reliability = mat.get('multiscale_reliability')
    if scores is None or scores.shape != expected_shape:
        scores = np.zeros(expected_shape, dtype=float)
    if multiscale_reliability is None or multiscale_reliability.shape != expected_shape:
        multiscale_reliability = np.zeros(expected_shape, dtype=float)
    return (
        np.nan_to_num(reliability, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0),
        np.clip(np.nan_to_num(
            multiscale_reliability, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0),
    )


def find_matrix_by_keyword(subj_dir, keyword):
    keyword = keyword.lower()
    return next((
        os.path.join(subj_dir, filename)
        for filename in os.listdir(subj_dir)
        if keyword in filename.lower()
    ), None)


def format_float_tag(value):
    return f"{value:g}".replace('.', 'p').replace('-', 'm')


def causality_cache_path(subj_dir):
    subject = os.path.basename(subj_dir.rstrip(os.sep))
    tag = (
        f'v4multiscale{ACX_SCALE_BANDS}_d{ACX_EMBEDDING_DIM}_delay{ACX_DELAY}'
        f'_scales{ACX_SCALES}_tw{ACX_THEILER_WINDOW}'
        f'_eps{format_float_tag(ACX_EPSILON_MIN)}-{format_float_tag(ACX_EPSILON_MAX)}'
    )
    return os.path.join(subj_dir, f'{subject}_AAL116_causality_{tag}.mat')


def build_causality_cache(subj_dir):
    cache_path = causality_cache_path(subj_dir)
    if os.path.exists(cache_path):
        return
    matrix = load_mat_file(find_matrix_by_keyword(subj_dir, 'features_timeseries'))
    if matrix.shape[0] == matrix.shape[1]:
        reliability = np.ones_like(matrix, dtype=float)
        np.fill_diagonal(reliability, 0.0)
        scores = np.repeat(matrix[..., None], ACX_SCALE_BANDS, axis=-1)
        multiscale_reliability = np.repeat(reliability[..., None], ACX_SCALE_BANDS, axis=-1)
    else:
        matrix, reliability, scores, multiscale_reliability = (
            continuity_scaling_multiscale_matrices(matrix))
    save_causality_cache(
        cache_path, matrix, reliability, scores, multiscale_reliability)


def build_causality_caches(root_dir, subjects, workers):
    missing = [
        os.path.join(root_dir, subject)
        for subject in subjects
        if not os.path.exists(causality_cache_path(os.path.join(root_dir, subject)))
    ]
    if workers > 1 and missing:
        with ProcessPoolExecutor(max_workers=min(workers, len(missing))) as executor:
            list(executor.map(build_causality_cache, missing))
