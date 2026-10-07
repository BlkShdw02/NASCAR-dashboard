"""Weather ingest + finish prediction -> data/latest.json. Run on a schedule."""
import json, os, re, sys, math, datetime as dt, zoneinfo
import numpy as np, requests, pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
UA = {"User-Agent": "nascar-dashboard (malik.i.bryant@gmail.com)"}  # NWS requires a UA
# Open-Meteo model ids. VERIFY against open-meteo.com/en/docs if one returns empty.
OM = {"GFS": "gfs_global", "ECMWF": "ecmwf_ifs025", "NBM": "ncep_nbm_conus",
      "NAM": "ncep_nam_conus", "HRRR": "ncep_hrrr_conus"}
WEIGHT = {"HRRR": 2, "NAM": 1.5, "NBM": 1.3, "NDFD": 1, "ECMWF": 1.2, "GFS": 1}
# race-day checkpoints, hours relative to green flag
STEPS = [("fan_zone", "1. Fan Zone", -3.0), ("pre_race", "2. Pre Race", -1.0), ("green_flag", "3. Green Flag", 0.0),
         ("halfway", "4. Halfway", 1.5), ("checkered", "5. Checkered Flag", 3.5)]
C16 = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
COMPASS = {c: i * 22.5 for i, c in enumerate(C16)}
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")

def active(model, lead):
    return lead <= 48 if model == "HRRR" else lead <= 60 if model == "NAM" else True

def _col(df, var):
    return var if var in df else next((c for c in df.columns if c.startswith(var)), None)

def om_hourly(r):
    out = {}
    for name, mid in OM.items():
        try:
            j = requests.get("https://api.open-meteo.com/v1/forecast", timeout=30, params={
                "latitude": r["lat"], "longitude": r["lon"], "models": mid,
                "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,precipitation_probability",
                "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                "timezone": "UTC", "forecast_days": 10}).json()["hourly"]
            raw = pd.DataFrame(j); raw.index = pd.to_datetime(raw.pop("time"), utc=True)
            df = pd.DataFrame({k: raw[_col(raw, v)] if _col(raw, v) else np.nan for k, v in
                               {"temp": "temperature_2m", "wind": "wind_speed_10m", "dir": "wind_direction_10m",
                                "precip": "precipitation_probability"}.items()})
            if df.notna().any().any(): out[name] = df
        except Exception as e:
            print(f"[warn] {name}: {e}", file=sys.stderr)
    return out

def ndfd_hourly(r):  # NWS point forecast is built from NDFD grids
    try:
        p = requests.get(f"https://api.weather.gov/points/{r['lat']},{r['lon']}", headers=UA, timeout=30).json()
        h = requests.get(p["properties"]["forecastHourly"], headers=UA, timeout=30).json()["properties"]["periods"]
        rows = []
        for x in h:
            ws = re.findall(r"\d+", str(x.get("windSpeed", "")))
            rows.append({"t": pd.to_datetime(x["startTime"], utc=True), "temp": x["temperature"],
                         "wind": float(ws[-1]) if ws else None, "dir": COMPASS.get(x.get("windDirection")),
                         "precip": (x.get("probabilityOfPrecipitation") or {}).get("value")})
        return {"NDFD": pd.DataFrame(rows).set_index("t").sort_index().astype(float)}
    except Exception as e:
        print(f"[warn] NDFD: {e}", file=sys.stderr); return {}

def _sec(idx): return np.asarray((idx - EPOCH).total_seconds())

def at(df, t):
    """Value of each metric at time t (linear interpolation; wind direction via unit vectors)."""
    ts = (t - EPOCH).total_seconds(); res = {}
    for k in ("temp", "wind", "precip", "dir"):
        s = df[k].dropna() if k in df else pd.Series(dtype=float)
        if len(s) < 2: res[k] = None; continue
        x = _sec(s.index)
        if not (x[0] <= ts <= x[-1]): res[k] = None; continue
        if k == "dir":
            rad = np.radians(s.values)
            res[k] = float(np.degrees(math.atan2(np.interp(ts, x, np.sin(rad)), np.interp(ts, x, np.cos(rad)))) % 360)
        else: res[k] = float(np.interp(ts, x, s.values))
    return res

def blend_vals(vals, lead):
    res = {}
    for k in ("temp", "wind", "precip"):
        num = den = 0
        for m, v in vals.items():
            if active(m, lead) and v.get(k) is not None: num += WEIGHT[m] * v[k]; den += WEIGHT[m]
        res[k] = num / den if den else None
    sn = cs = 0
    for m, v in vals.items():
        if active(m, lead) and v.get("dir") is not None:
            sn += WEIGHT[m] * math.sin(math.radians(v["dir"])); cs += WEIGHT[m] * math.cos(math.radians(v["dir"]))
    res["dir"] = math.degrees(math.atan2(sn, cs)) % 360 if (sn or cs) else None
    return res

def rnd(v): return {k: (None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, 0 if k == "dir" else 1)) for k, x in v.items()}

def race_weather(r, start, lead):
    hourly = {m: d for m, d in {**om_hourly(r), **ndfd_hourly(r)}.items() if active(m, lead)}
    steps = []
    for key, label, off in STEPS:
        t = pd.Timestamp(start + dt.timedelta(hours=off)); mv = {m: at(d, t) for m, d in hourly.items()}
        steps.append({"key": key, "label": label, "offset_h": off, "time_utc": t.isoformat(),
                      "blend": rnd(blend_vals(mv, lead)), "models": {m: rnd(v) for m, v in mv.items()}})
    t0 = pd.Timestamp(start).floor("h") - pd.Timedelta(hours=6); grid = [t0 + pd.Timedelta(hours=i) for i in range(13)]
    gv = {m: [at(d, t) for t in grid] for m, d in hourly.items()}
    ser = {m: {k: [rnd(v)[k] for v in vs] for k in ("temp", "wind", "dir", "precip")} for m, vs in gv.items()}
    bl = [rnd(blend_vals({m: gv[m][i] for m in gv}, lead)) for i in range(len(grid))]
    series = {"times": [t.isoformat() for t in grid], "models": ser, "blend": {k: [b[k] for b in bl] for k in ("temp", "wind", "dir", "precip")}}
    race = [s["blend"] for s in steps[2:]]          # green flag -> checkered
    mean = lambda k: round(float(np.mean([b[k] for b in race if b[k] is not None])), 1) if any(b[k] is not None for b in race) else None
    sn = sum(math.sin(math.radians(b["dir"])) for b in race if b["dir"] is not None); cs = sum(math.cos(math.radians(b["dir"])) for b in race if b["dir"] is not None)
    fc = {"temp": mean("temp"), "wind": mean("wind"), "precip": mean("precip"), "dir": round(math.degrees(math.atan2(sn, cs)) % 360) if (sn or cs) else None}
    return list(hourly), steps, series, fc

def clean_name(n): return re.sub(r"\s*\(.*?\)|[#*]", "", str(n)).strip()
def nkey(n):
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", clean_name(n).lower()); return re.sub(r"[^a-z]", "", n)

def predict(r, fc, W=(0.45, 0.20, 0.20, 0.15)):
    # finishes.csv: date,track,series,track_type,driver,finish[,temp_f,wind_mph]
    f = pd.read_csv(os.path.join(ROOT, "data", "finishes.csv")); f["date"] = pd.to_datetime(f["date"])
    f = f[(f.series == r["series"]) & (f.date >= "2022-01-01")].copy(); f["key"] = f.driver.map(nkey)
    disp = f.sort_values("date").groupby("key").driver.last().map(clean_name).to_dict()
    # ACTIVE drivers: raced in at least 2 of the 5 most recent races in the data; config/entry_list.txt overrides
    last_races = f[["date", "track"]].drop_duplicates().sort_values("date").tail(5)
    hits = f.merge(last_races, on=["date", "track"]).drop_duplicates(["key", "date", "track"]).groupby("key").size()
    act = set(hits[hits >= 2].index)
    entry = os.path.join(ROOT, "config", "entry_list.txt")
    if os.path.exists(entry):
        names = [l.strip() for l in open(entry) if l.strip() and not l.startswith("#")]
        if names:
            act = {nkey(n) for n in names}
            for n in names: disp.setdefault(nkey(n), n)
    here = f[f.track == r["track"]]; mile = f[(f.track_type == r["track_type"]) & (f.date.dt.year == dt.date.today().year)]
    sim = here.iloc[0:0]
    if fc["temp"] is not None and {"temp_f", "wind_mph"} <= set(here.columns):
        s2 = here[(abs(here.temp_f - fc["temp"]) <= 10) & (abs(here.wind_mph - (fc["wind"] or 0)) <= 8)]
        if len(s2) >= 20: sim = s2
    g = lambda d: d.groupby("key").finish.agg(["mean", "count"]) if len(d) else None
    T, S, M, R = g(here), g(sim), g(mile), g(f.sort_values("date").groupby("key").tail(5))
    fm = here.finish.mean() if len(here) else 20; rows = []
    for k in sorted(act):
        v = [None]
        if T is not None and k in T.index:
            n = T.loc[k, "count"]; v[0] = (T.loc[k, "mean"] * n + fm * 3) / (n + 3)  # shrink small samples
        v += [X.loc[k, "mean"] if X is not None and k in X.index else None for X in (S, M, R)]
        w = [(x, wt) for x, wt in zip(v, W) if x is not None]
        score = sum(x * wt for x, wt in w) / sum(wt for _, wt in w) if w else fm
        rows.append({"driver": disp.get(k, k), "score": round(float(score), 2), "track": v[0], "similar_wx": v[1], "mile_half": v[2], "form": v[3]})
    rows.sort(key=lambda x: x["score"])
    for i, x in enumerate(rows, 1): x["pred"] = i
    return rows

def main():
    cfg = json.load(open(os.path.join(ROOT, "config", "races.json")))
    now = dt.datetime.now(dt.timezone.utc); out = []
    fin = os.path.exists(os.path.join(ROOT, "data", "finishes.csv"))
    for race in cfg["races"]:
        r = {**cfg["tracks"][race["track"]], **race}   # merge track info (lat/lon/tz/track_type) into race
        start = dt.datetime.fromisoformat(r["start_et"]).replace(tzinfo=zoneinfo.ZoneInfo("America/New_York")).astimezone(dt.timezone.utc)
        if start < now - dt.timedelta(hours=8): continue  # skip finished races
        lead = (start - now).total_seconds() / 3600
        if lead > 24 * 10: continue  # beyond forecast horizon
        used, steps, series, fc = race_weather(r, start, lead)
        out.append({"id": r["id"], "track": r["track"], "tz": r["tz"], "track_type": r["track_type"], "lat": r["lat"], "lon": r["lon"],
                    "frontstretch_deg": r.get("frontstretch_deg"), "start_utc": start.isoformat(), "lead_hours": round(lead, 1),
                    "models_used": used, "steps": steps, "series": series, "forecast": fc,
                    "predictions": predict(r, fc) if fin else []})
    json.dump({"generated_utc": now.isoformat(), "races": out}, open(os.path.join(ROOT, "data", "latest.json"), "w"), indent=1, default=str)

if __name__ == "__main__": main()
