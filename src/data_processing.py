from typing import Dict, Tuple

import os

import numpy as np
from sklearn.preprocessing import StandardScaler


def load_fi2010_file(path: str) -> Tuple[np.ndarray, np.ndarray | None]:
    data = np.loadtxt(path)
    if data.ndim != 2:
        raise ValueError("FI-2010 file should be a 2D matrix")
    if data.shape[0] < 40:
        raise ValueError("FI-2010 file should have at least 40 feature rows")

    x = data[:40, :].T
    y = data[40:45, :].T if data.shape[0] >= 45 else None
    return x, y


def parse_fi2010_levels(
    x: np.ndarray,
    levels_per_side: int = 10,
    order: str = "level",
) -> Dict[str, np.ndarray]:
    n, d = x.shape
    expected = levels_per_side * 4
    if d != expected:
        raise ValueError(f"Expected {expected} features, got {d}")

    if order == "level":
        reshaped = x.reshape(n, levels_per_side, 4)
        ask_px = reshaped[:, :, 0]
        ask_sz = reshaped[:, :, 1]
        bid_px = reshaped[:, :, 2]
        bid_sz = reshaped[:, :, 3]
    elif order == "block":
        ask_px = x[:, 0:levels_per_side]
        ask_sz = x[:, levels_per_side : 2 * levels_per_side]
        bid_px = x[:, 2 * levels_per_side : 3 * levels_per_side]
        bid_sz = x[:, 3 * levels_per_side : 4 * levels_per_side]
    else:
        raise ValueError("order must be 'level' or 'block'")

    return {
        "ask_px": ask_px,
        "ask_sz": ask_sz,
        "bid_px": bid_px,
        "bid_sz": bid_sz,
    }


def compute_microstructure_features(
    ask_px: np.ndarray,
    bid_px: np.ndarray,
    ask_sz: np.ndarray,
    bid_sz: np.ndarray,
) -> Dict[str, np.ndarray]:
    eps = 1e-6
    spread = ask_px[:, 0] - bid_px[:, 0]
    mid_price = (ask_px[:, 0] + bid_px[:, 0]) / 2.0

    total_bid = bid_sz.sum(axis=1)
    total_ask = ask_sz.sum(axis=1)
    imbalance = (total_bid - total_ask) / (total_bid + total_ask + eps)

    microprice = (
        ask_px[:, 0] * bid_sz[:, 0] + bid_px[:, 0] * ask_sz[:, 0]
    ) / (bid_sz[:, 0] + ask_sz[:, 0] + eps)

    weights = np.arange(ask_sz.shape[1], dtype=np.float32)
    slope_bid = (bid_sz * weights).sum(axis=1) / (total_bid + eps)
    slope_ask = (ask_sz * weights).sum(axis=1) / (total_ask + eps)
    depth_ratio = slope_bid - slope_ask

    near = min(5, ask_sz.shape[1])
    vol_pressure = (bid_sz[:, :near].sum(axis=1) + eps) / (
        ask_sz[:, :near].sum(axis=1) + eps
    )

    delta_mid = np.diff(mid_price, prepend=mid_price[0])
    delta_vol = np.diff(total_bid + total_ask, prepend=(total_bid + total_ask)[0])

    return {
        "spread": spread,
        "mid_price": mid_price,
        "imbalance": imbalance,
        "microprice": microprice,
        "depth_ratio": depth_ratio,
        "vol_pressure": vol_pressure,
        "total_volume": total_bid + total_ask,
        "delta_mid_price": delta_mid,
        "delta_total_volume": delta_vol,
    }


def build_lit_tensor(
    x: np.ndarray,
    levels_per_side: int = 10,
    order: str = "level",
    train_end: int | None = None,
    scalers: Dict[str, StandardScaler] | None = None,
    use_extra: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, StandardScaler], Dict[str, np.ndarray]]:
    parsed = parse_fi2010_levels(x, levels_per_side=levels_per_side, order=order)
    ask_px, ask_sz = parsed["ask_px"], parsed["ask_sz"]
    bid_px, bid_sz = parsed["bid_px"], parsed["bid_sz"]

    features = compute_microstructure_features(ask_px, bid_px, ask_sz, bid_sz)
    mid_price = features["mid_price"]

    price_raw = np.concatenate([ask_px, bid_px], axis=1).astype(np.float32)
    size_raw = np.concatenate([ask_sz, bid_sz], axis=1).astype(np.float32)

    eps = 1e-6
    price_rel = (price_raw - mid_price[:, None]) / (mid_price[:, None] + eps)
    price_rel = np.where(np.isfinite(price_rel), price_rel, 0.0)

    size_log = np.log1p(np.maximum(size_raw, 0.0))
    size_log = np.where(np.isfinite(size_log), size_log, 0.0)

    if scalers is None:
        if train_end is None:
            raise ValueError("train_end must be provided when scalers is None")
        price_scaler = StandardScaler().fit(price_rel[:train_end])
        size_scaler = StandardScaler().fit(size_log[:train_end])
        extra_scaler = StandardScaler().fit(
            np.column_stack([features[k] for k in features])[:train_end]
        )
        scalers = {
            "price": price_scaler,
            "size": size_scaler,
            "extra": extra_scaler,
        }

    price_scaled = scalers["price"].transform(price_rel)
    size_scaled = scalers["size"].transform(size_log)

    # Optional normalization diagnostics
    if bool(os.environ.get("LIT_PRINT_NORMALIZATION", "")):
        def _summ(name: str, arr: np.ndarray) -> None:
            flat = arr.reshape(-1)
            pct = np.percentile(flat, [1, 5, 50, 95, 99]).round(4).tolist()
            print(f"{name}: mean={flat.mean():.4f} std={flat.std():.4f} p1/5/50/95/99={pct}")

        print("--- Price normalization ---")
        _summ("price_rel (raw)", price_rel)
        _summ("price_scaled", price_scaled)
        print("--- Size normalization ---")
        _summ("size_log (raw)", size_log)
        _summ("size_scaled", size_scaled)

    price_grid = price_scaled[:, :, None]
    size_grid = size_scaled[:, :, None]

    side_channel = np.concatenate(
        [np.ones((len(x), levels_per_side)), np.zeros((len(x), levels_per_side))],
        axis=1,
    )
    side_channel = side_channel[:, :, None]

    extra_grid = np.zeros((len(x), price_grid.shape[1], 0), dtype=np.float32)
    if use_extra:
        extra_vals = np.column_stack([features[k] for k in features])
        extra_scaled = scalers["extra"].transform(extra_vals).astype(np.float32)
        extra_grid = np.repeat(extra_scaled[:, None, :], repeats=price_grid.shape[1], axis=1)

    lit_tensor = np.concatenate([price_grid, size_grid, side_channel, extra_grid], axis=2)
    return lit_tensor.astype(np.float32), mid_price, scalers, features


def temporal_split(n: int, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, slice]:
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, n),
    }
