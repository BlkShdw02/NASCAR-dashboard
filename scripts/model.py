"""Scoring model shared by ingest.py (live predictions) and evaluate.py (backtest + tuning).
Lower score = better expected finish. Four factors, each an average finish position:
  track       - driver's avg at this track since 2022 (shrunk toward the field avg when few starts)
  similar_wx  - driver's avg at this track in similar weather (temp +/-10F, wind +/-8 mph; needs 20+ results)
  mile_half   - driver's avg this season at tracks of the same type (e.g. 1.5-mile)
  form        - driver's avg over their last 5 races"""
import os, re, json
import numpy as np, pandas as pd

FACTORS = ["track", "similar_wx", "mile_half", "form"]
DEFAULT_W = (0.45, 0.20, 0.20, 0.15)
SHRINK_K = 3

def clean_name(n): return re.sub(r"\s*\(.*?\)|[#*]", "", str(n)).strip()
def nkey(n):
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", clean_name(n).lower()); return re.sub(r"[^a-z]", "", n)

def load_finishes(path):
    f = pd.read_csv(path); f["date"] = pd.to_datetime(f["date"])
    f = f[(f.series == "Cup") & (f.date >= "2022-01-01")].copy(); f["key"] = f.driver.map(nkey); return f

def display_names(f): return f.sort_values("date").groupby("key").driver.last().map(clean_name).to_dict()

def active_keys(f, entry_path=None, n_recent=5, min_hits=2):
    """Active = raced in >=2 of the 5 most recent races in the data. config/entry_list.txt (one name per line) overrides."""
    if entry_path and os.path.exists(entry_path):
        names = [l.strip() for l in open(entry_path) if l.strip() and not l.startswith("#")]
        if names: return {nkey(n) for n in names}, {nkey(n): n for n in names}
    last = f[["date", "track"]].drop_duplicates().sort_values("date").tail(n_recent)
    hits = f.merge(last, on=["date", "track"]).drop_duplicates(["key", "date", "track"]).groupby("key").size()
    return set(hits[hits >= min_hits].index), {}

def _agg(d):
    if not len(d): return pd.DataFrame({"mean": pd.Series(dtype=float), "count": pd.Series(dtype=float)})
    return d.groupby("key").finish.agg(["mean", "count"]).astype(float)

def factor_table(f, track, track_type, asof, temp, wind, keys):
    """Factor values per driver using only races before `asof` (no look-ahead)."""
    h = f[f.date < asof]; here = h[h.track == track]
    mile = h[(h.track_type == track_type) & (h.date.dt.year == asof.year)]
    fm = here.finish.mean() if len(here) else 20.0
    T = _agg(here); T = (T["mean"] * T["count"] + fm * SHRINK_K) / (T["count"] + SHRINK_K)
    sim = here.iloc[0:0]
    if temp is not None and not pd.isna(temp) and {"temp_f", "wind_mph"} <= set(here.columns):
        w = 0 if wind is None or pd.isna(wind) else wind
        s2 = here[(abs(here.temp_f - temp) <= 10) & (abs(here.wind_mph - w) <= 8)]
        if len(s2) >= 20: sim = s2
    tab = pd.DataFrame({"track": T, "similar_wx": _agg(sim)["mean"], "mile_half": _agg(mile)["mean"],
                        "form": _agg(h.sort_values("date").groupby("key").tail(5))["mean"]})
    return tab.reindex(list(keys)).astype(float)

def combine(tab, W, fill=20.0):
    v = tab[FACTORS].to_numpy(float); m = ~np.isnan(v); w = np.asarray(W, float)
    num = (np.where(m, v, 0) * w).sum(1); den = (m * w).sum(1)
    s = pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan), index=tab.index)
    return s.fillna(s.mean() if s.notna().any() else fill)

def race_metrics(score, actual):
    pr = score.rank(method="first"); a = actual.reindex(score.index).dropna(); pr = pr.reindex(a.index)
    if len(a) < 5: return None
    return float((pr - a).abs().mean()), float(np.corrcoef(pr.rank(), a.rank())[0, 1])

def random_mae(n): return (n * n - 1) / (3 * n)

def build_backtest(f, since="2023-01-01"):
    out = []
    for (date, track), g in f[f.date >= since].groupby(["date", "track"]):
        g = g.drop_duplicates("key").set_index("key")
        temp = g.temp_f.mean() if "temp_f" in g else None; wind = g.wind_mph.mean() if "wind_mph" in g else None
        tab = factor_table(f, track, g.track_type.iloc[0], date, temp, wind, g.index)
        if tab["track"].notna().mean() < 0.3: continue      # not enough earlier history at this track
        out.append({"date": date, "track": track, "tab": tab, "actual": g.finish.astype(float)})
    return out

def score_weights(bt, W):
    r = [race_metrics(combine(b["tab"], W), b["actual"]) for b in bt]; r = [x for x in r if x]
    return (float(np.mean([x[0] for x in r])), float(np.mean([x[1] for x in r]))) if r else (None, None)

def simplex(step=0.05):
    n = round(1 / step)
    for a in range(n + 1):
        for b in range(n + 1 - a):
            for c in range(n + 1 - a - b): yield (a * step, b * step, c * step, (n - a - b - c) * step)

def tune(bt, step=0.05):
    best, bm = DEFAULT_W, score_weights(bt, DEFAULT_W)[0]
    for W in simplex(step):
        m = score_weights(bt, W)[0]
        if m is not None and m < bm - 1e-9: best, bm = W, m
    return best, bm

def load_weights(root):
    p = os.path.join(root, "config", "weights.json")
    if os.path.exists(p):
        try:
            j = json.load(open(p)); return tuple(j["weights"]), j.get("source", "tuned")
        except Exception: pass
    return DEFAULT_W, "default"
