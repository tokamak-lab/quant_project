from typing import Iterable, Tuple

import numpy as np


def estimate_tick_size(
    best_bid: np.ndarray,
    best_ask: np.ndarray,
    mid_price: np.ndarray | None = None,
    round_decimals: int = 6,
) -> Tuple[float, str]:
    bid_diffs = np.abs(np.diff(best_bid))
    ask_diffs = np.abs(np.diff(best_ask))
    diffs = np.concatenate([bid_diffs, ask_diffs], axis=0)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]

    if diffs.size > 0:
        diffs_round = np.round(diffs, round_decimals)
        values, counts = np.unique(diffs_round, return_counts=True)
        tick_size = float(values[np.argmax(counts)])
        return tick_size, "best_bid_ask_mode"

    if mid_price is not None:
        mid_diffs = np.abs(np.diff(mid_price))
        mid_diffs = mid_diffs[np.isfinite(mid_diffs) & (mid_diffs > 0)]
        if mid_diffs.size > 0:
            return float(np.median(mid_diffs)), "mid_price_median"

    return 0.0, "fallback_zero"


def build_directional_labels(
    mid_price: np.ndarray,
    horizons: Iterable[int],
    tick_size: float,
    neutral_band_ticks: float = 0.5,
    eps: float = 1e-6,
) -> np.ndarray:
    horizons = list(horizons)
    n = len(mid_price)
    labels = []

    for h in horizons:
        future = np.roll(mid_price, -h)
        ret = (future - mid_price) / (mid_price + eps)
        thr_rel = (neutral_band_ticks * tick_size) / (mid_price + eps)
        up = ret > thr_rel
        down = ret < -thr_rel
        y = np.full(n, 1, dtype=np.int64)
        y[up] = 2
        y[down] = 0
        y[n - h :] = 1
        labels.append(y)

    return np.vstack(labels)
