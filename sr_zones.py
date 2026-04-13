"""
sr_zones.py — Strong Support/Resistance + Demand/Supply Zone Detection

FIXES from v6.0:
  [FIX] find_swing_points() now uses tolerance comparison instead of exact ==
        Floating-point OHLCV data can have near-equal values that miss with ==
        e.g. max([3001.1, 3001.1000000001]) != 3001.1 due to float precision
  [FIX] zone_pips treated as percentage of price when > 0 and < 1, allowing
        the caller to pass 0.003 (0.3%) instead of a fixed pip value — this
        makes zones scale correctly whether Gold is at $1800 or $3200

Logic:
  - Scan historical OHLCV for swing highs/lows
  - Cluster nearby levels into zones
  - Score each zone by number of touches + recency + volume reaction
  - Identify demand zones (strong buying = supply absorbed) and supply zones
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class Zone:
    price: float
    high: float
    low: float
    zone_type: str        # "SUPPORT" / "RESISTANCE" / "DEMAND" / "SUPPLY"
    touches: int
    strength: int         # 0–100
    last_touch_idx: int
    is_fresh: bool

    def __repr__(self):
        return f"{self.zone_type}({self.price:.2f}, touches={self.touches}, strength={self.strength})"


def find_swing_points(df: pd.DataFrame, window: int = 5) -> tuple:
    """
    Find swing highs and swing lows using rolling window.

    [FIX v6.1] Changed exact == comparison to tolerance-based comparison.
    Float OHLCV data from MT5/exchanges can have tiny precision differences
    (e.g. 3001.1 vs 3001.1000000001) that cause real swing points to be missed.
    Tolerance = 0.0001% of price — imperceptibly small but fixes float issues.
    """
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(df)

    swing_highs = []
    swing_lows  = []

    for i in range(window, n - window):
        window_highs = highs[i - window:i + window + 1]
        window_lows  = lows[i - window:i + window + 1]

        max_high = max(window_highs)
        min_low  = min(window_lows)

        # [FIX] Use tolerance instead of exact ==
        # 0.001% tolerance handles float precision without masking real diffs
        high_tol = max_high * 0.00001
        low_tol  = min_low  * 0.00001

        if abs(highs[i] - max_high) <= high_tol:
            swing_highs.append((i, highs[i]))
        if abs(lows[i] - min_low) <= low_tol:
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


def cluster_levels(points: list, tolerance_pct: float = 0.002) -> List[list]:
    """
    Cluster nearby price levels into zones.
    tolerance_pct: 0.2% — levels within this range are merged.
    """
    if not points:
        return []

    sorted_points = sorted(points, key=lambda x: x[1])
    clusters      = []
    current_cluster = [sorted_points[0]]

    for i in range(1, len(sorted_points)):
        idx, price       = sorted_points[i]
        cluster_center   = np.mean([p[1] for p in current_cluster])

        if abs(price - cluster_center) / cluster_center <= tolerance_pct:
            current_cluster.append((idx, price))
        else:
            clusters.append(current_cluster)
            current_cluster = [(idx, price)]

    clusters.append(current_cluster)
    return clusters


def build_zones(df: pd.DataFrame, lookback: int = 200, min_touches: int = 2,
                zone_pips: float = 5.0) -> List[Zone]:
    """
    Build S/R and Demand/Supply zones from OHLCV data.

    Args:
        df: OHLCV DataFrame (columns: open, high, low, close, volume)
        lookback: number of bars to analyze
        min_touches: minimum touches to qualify as strong zone
        zone_pips: zone half-width. If < 1.0, treated as percentage of price
                   (e.g. 0.003 = 0.3%). If >= 1.0, treated as absolute pip value.
                   [FIX] Percentage mode makes zones scale with Gold price level.
    """
    df = df.tail(lookback).copy().reset_index(drop=True)
    close         = df["close"].values
    current_price = close[-1]

    # [FIX] Convert percentage zone_pips to absolute price units
    if 0 < zone_pips < 1.0:
        zone_pips_abs = current_price * zone_pips
    else:
        zone_pips_abs = zone_pips

    swing_highs, swing_lows = find_swing_points(df, window=5)

    support_clusters    = cluster_levels(swing_lows,  tolerance_pct=0.003)
    resistance_clusters = cluster_levels(swing_highs, tolerance_pct=0.003)

    zones: List[Zone] = []

    def score_zone(cluster, zone_type, df):
        touches = len(cluster)
        if touches < min_touches:
            return None

        prices   = [p[1] for p in cluster]
        indices  = [p[0] for p in cluster]
        center   = np.mean(prices)
        last_idx = max(indices)

        recency_score = min(40, int(40 * (last_idx / len(df))))

        vol_score = 0
        if "volume" in df.columns and df["volume"].sum() > 0:
            zone_mask = (
                (df["close"] >= center - zone_pips_abs * 2) &
                (df["close"] <= center + zone_pips_abs * 2)
            )
            zone_vol = df.loc[zone_mask, "volume"].mean()
            avg_vol  = df["volume"].mean()
            if avg_vol > 0:
                vol_score = min(20, int(20 * (zone_vol / avg_vol)))

        touch_score = min(40, touches * 10)
        total       = touch_score + recency_score + vol_score

        recent_closes = df["close"].values[-20:]
        if zone_type == "SUPPORT":
            broken = any(c < center - zone_pips_abs * 3 for c in recent_closes)
        else:
            broken = any(c > center + zone_pips_abs * 3 for c in recent_closes)
        is_fresh = not broken

        return Zone(
            price=center,
            high=center + zone_pips_abs,
            low=center  - zone_pips_abs,
            zone_type=zone_type,
            touches=touches,
            strength=total,
            last_touch_idx=last_idx,
            is_fresh=is_fresh,
        )

    for cluster in support_clusters:
        z = score_zone(cluster, "SUPPORT", df)
        if z and z.price < current_price:
            zones.append(z)

    for cluster in resistance_clusters:
        z = score_zone(cluster, "RESISTANCE", df)
        if z and z.price > current_price:
            zones.append(z)

    for z in zones:
        if z.zone_type == "SUPPORT"    and z.strength >= 60:
            z.zone_type = "DEMAND"
        elif z.zone_type == "RESISTANCE" and z.strength >= 60:
            z.zone_type = "SUPPLY"

    zones.sort(key=lambda z: z.strength, reverse=True)
    return zones


def get_nearest_zones(zones: List[Zone], current_price: float,
                      max_distance_pct: float = 0.005):
    """
    Return nearest support/demand below and resistance/supply above current price.
    """
    nearest_support    = None
    nearest_resistance = None

    supports    = [z for z in zones if z.zone_type in ("SUPPORT", "DEMAND")     and z.price < current_price]
    resistances = [z for z in zones if z.zone_type in ("RESISTANCE", "SUPPLY")  and z.price > current_price]

    if supports:
        nearest_support    = max(supports,    key=lambda z: z.price)
    if resistances:
        nearest_resistance = min(resistances, key=lambda z: z.price)

    at_support    = (nearest_support    and
                     abs(current_price - nearest_support.price)    / current_price <= max_distance_pct)
    at_resistance = (nearest_resistance and
                     abs(current_price - nearest_resistance.price) / current_price <= max_distance_pct)

    return nearest_support, nearest_resistance, at_support, at_resistance
