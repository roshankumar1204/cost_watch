"""Step 1: weekly cost per tonne-km per route, plus the two comparison baselines.

Everything here is plain pandas, with no randomness, so the numbers are identical on every run.
"""
import pandas as pd

import config


def load_shipments(path=config.SHIPMENTS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["shipment_date"])
    required = {"origin", "destination", "route_type", "quantity_tonnes",
                "distance_km", "freight_cost_inr", "shipment_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"shipment file is missing columns: {sorted(missing)}")
    bad = (df["quantity_tonnes"] <= 0) | (df["distance_km"] <= 0) | (df["freight_cost_inr"] < 0)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows have non-positive quantity/distance or negative cost")
    df["route"] = df["origin"].str.strip() + "-" + df["destination"].str.strip()
    # Weeks run Monday-Sunday; week_of is that Monday.
    df["week_of"] = df["shipment_date"] - pd.to_timedelta(df["shipment_date"].dt.weekday, unit="D")
    return df


def weekly_costs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (route, route_type, week).

    cost_per_tonne_km = total freight cost / total (tonnes x km) for the week,
    i.e. a weighted ratio, as the brief defines it, not the mean of per-shipment ratios.
    """
    df = df.assign(tonne_km=df["quantity_tonnes"] * df["distance_km"])
    weekly = (
        df.groupby(["route", "route_type", "week_of"], as_index=False)
          .agg(total_cost=("freight_cost_inr", "sum"),
               total_tonne_km=("tonne_km", "sum"),
               shipments=("freight_cost_inr", "size"))
    )
    weekly["cost_per_tonne_km"] = weekly["total_cost"] / weekly["total_tonne_km"]
    return weekly.sort_values(["route", "week_of"]).reset_index(drop=True)


def add_baselines(weekly: pd.DataFrame) -> pd.DataFrame:
    w = weekly.copy()

    # vs own history: mean of up to 8 PRIOR weeks (shift(1) = no look-ahead).
    # Weeks with fewer than 8 prior weeks use what exists; week 1 has none -> NaN.
    by_route = w.groupby("route")["cost_per_tonne_km"]
    w["own_baseline"] = by_route.transform(
        lambda s: s.shift(1).rolling(config.HISTORY_WEEKS, min_periods=1).mean()
    )
    w["prior_weeks_used"] = by_route.cumcount().clip(upper=config.HISTORY_WEEKS)

    # vs similar routes: same week, same route_type, excluding the route itself.
    grp = w.groupby(["route_type", "week_of"])["cost_per_tonne_km"]
    peer_sum = grp.transform("sum") - w["cost_per_tonne_km"]
    peer_count = grp.transform("size") - 1
    w["peer_baseline"] = peer_sum / peer_count.where(peer_count > 0)
    w["peer_count"] = peer_count

    w["vs_own_pct"] = w["cost_per_tonne_km"] / w["own_baseline"] - 1
    w["vs_peer_pct"] = w["cost_per_tonne_km"] / w["peer_baseline"] - 1
    return w


def build_weekly_table(path=config.SHIPMENTS_CSV) -> pd.DataFrame:
    return add_baselines(weekly_costs(load_shipments(path)))
