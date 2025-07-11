"""Change-point detection for spectral time series."""

import numpy as np
from typing import Dict, List


def cusum_detector(series, threshold=3.0) -> Dict:
    """CUSUM detector for persistent level shifts."""
    s = np.asarray(series, dtype=float)
    n = len(s)
    mu = np.mean(s)
    sigma = np.std(s)
    if sigma < 1e-10:
        return {"change_points": [], "cusum_pos": np.zeros(n), "cusum_neg": np.zeros(n)}

    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)
    change_points = []

    for i in range(1, n):
        z = (s[i] - mu) / sigma
        cusum_pos[i] = max(0, cusum_pos[i-1] + z)
        cusum_neg[i] = max(0, cusum_neg[i-1] - z)
        if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
            change_points.append(i)
            cusum_pos[i] = 0
            cusum_neg[i] = 0

    return {"change_points": change_points, "cusum_pos": cusum_pos, "cusum_neg": cusum_neg,
            "threshold": threshold}


def lambda2_rate_of_change(series, window=20) -> Dict:
    """Rolling z-scored derivative of lambda_2."""
    s = np.asarray(series, dtype=float)
    n = len(s)
    deriv = np.zeros(n)
    deriv[1:] = np.diff(s)

    z_scores = np.zeros(n)
    for i in range(window, n):
        w = deriv[i-window:i]
        mu, sig = np.mean(w), np.std(w)
        z_scores[i] = (deriv[i] - mu) / sig if sig > 1e-10 else 0

    alerts = [int(i) for i in range(n) if abs(z_scores[i]) > 2.0]
    return {"derivatives": deriv, "z_scores": z_scores, "alerts": alerts}


def binary_segmentation(series, min_segment=20, max_breaks=10, penalty='bic') -> Dict:
    """Structural break detection via binary segmentation."""
    s = np.asarray(series, dtype=float)
    n = len(s)

    def segment_cost(seg):
        if len(seg) < 2:
            return 0
        return np.sum((seg - np.mean(seg)) ** 2)

    def find_best_split(arr):
        best_gain, best_idx = 0, -1
        total_cost = segment_cost(arr)
        for i in range(min_segment, len(arr) - min_segment):
            cost = segment_cost(arr[:i]) + segment_cost(arr[i:])
            gain = total_cost - cost
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        return best_idx, best_gain

    # BIC penalty
    if penalty == 'bic':
        pen = np.log(n) * np.var(s) if np.var(s) > 0 else 1.0
    else:
        pen = float(penalty) if isinstance(penalty, (int, float)) else 1.0

    breakpoints = []
    segments = [(0, n)]

    for _ in range(max_breaks):
        best_global_gain = 0
        best_seg_idx = -1
        best_split_pos = -1

        for seg_idx, (start, end) in enumerate(segments):
            seg = s[start:end]
            if len(seg) < 2 * min_segment:
                continue
            split_pos, gain = find_best_split(seg)
            if gain > best_global_gain:
                best_global_gain = gain
                best_seg_idx = seg_idx
                best_split_pos = start + split_pos

        if best_global_gain < pen or best_seg_idx < 0:
            break

        breakpoints.append(best_split_pos)
        start, end = segments[best_seg_idx]
        segments[best_seg_idx] = (start, best_split_pos)
        segments.insert(best_seg_idx + 1, (best_split_pos, end))

    breakpoints.sort()
    segment_means = [float(np.mean(s[start:end])) for start, end in segments]

    return {"breakpoints": breakpoints, "n_segments": len(segments),
            "segment_means": segment_means, "segments": segments}
