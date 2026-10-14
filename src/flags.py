"""Step 2: decide which route-weeks look like an unusual cost rise."""
import pandas as pd

import config


def apply_flag_rule(w: pd.DataFrame) -> pd.DataFrame:
    w = w.copy()
    rising = w["vs_own_pct"] > 0
    own_hit = w["vs_own_pct"] >= config.OWN_HISTORY_THRESHOLD
    peer_hit = w["vs_peer_pct"] >= config.PEER_THRESHOLD
    w["own_trigger"] = own_hit.fillna(False)
    w["peer_trigger"] = peer_hit.fillna(False)
    w["is_anomaly"] = rising.fillna(False) & (w["own_trigger"] | w["peer_trigger"])
    return w
