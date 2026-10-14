"""Weather ingest + finish prediction -> data/latest.json. Run on a schedule."""
import json, os, re, sys, math, datetime as dt, zoneinfo
import numpy as np, requests, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); import model

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

def predict(r, fc, start):
    f = model.load_finishes(os.path.join(ROOT, "data", "finishes.csv")); W, src = model.load_weights(ROOT)
    keys, extra = model.active_keys(f, os.path.join(ROOT, "config", "entry_list.txt")); disp = model.display_names(f)
    tab = model.factor_table(f, r["track"], r["track_type"], pd.Timestamp(start.date()), fc["temp"], fc["wind"], sorted(keys))
    score = model.combine(tab, W); nn = lambda v: None if pd.isna(v) else round(float(v), 2)
    rows = sorted(({"driver": disp.get(k, extra.get(k, k)), "score": round(float(score[k]), 2), "track": nn(tab.loc[k, "track"]),
                    "similar_wx": nn(tab.loc[k, "similar_wx"]), "mile_half": nn(tab.loc[k, "mile_half"]), "form": nn(tab.loc[k, "form"])}
                   for k in tab.index), key=lambda x: x["score"])
    for i, x in enumerate(rows, 1): x["pred"] = i
    return rows, list(W), src

def main():
    cfg = json.load(open(os.path.join(ROOT, "config", "races.json")))
    orient = json.load(open(os.path.join(ROOT, "config", "orientation.json"))) if os.path.exists(os.path.join(ROOT, "config", "orientation.json")) else {}
    now = dt.datetime.now(dt.timezone.utc); out = []; log = []
    fin = os.path.exists(os.path.join(ROOT, "data", "finishes.csv"))
    for race in cfg["races"]:
        r = {**cfg["tracks"][race["track"]], **race}   # merge track info (lat/lon/tz/track_type) into race
        start = dt.datetime.fromisoformat(r["start_et"]).replace(tzinfo=zoneinfo.ZoneInfo("America/New_York")).astimezone(dt.timezone.utc)
        if start < now - dt.timedelta(hours=8): continue  # skip finished races
        lead = (start - now).total_seconds() / 3600
        if lead > 24 * 10: continue  # beyond forecast horizon
        used, steps, series, fc = race_weather(r, start, lead)
        preds, W, src = predict(r, fc, start) if fin else ([], list(model.DEFAULT_W), "default")
        if lead > 0:   # keep the last PRE-race snapshot for grading
            log += [{"race_id": r["id"], "track": r["track"], "race_date": start.astimezone(zoneinfo.ZoneInfo("America/New_York")).date().isoformat(),
                     "driver": x["driver"], "pred": x["pred"], "score": x["score"], "generated_utc": now.isoformat()} for x in preds]
        out.append({"id": r["id"], "track": r["track"], "tz": r["tz"], "track_type": r["track_type"], "lat": r["lat"], "lon": r["lon"],
                    "frontstretch_deg": r.get("frontstretch_deg"), "osm_heading_deg": (orient.get(r["track"]) or {}).get("heading_deg"),
                    "start_utc": start.isoformat(), "lead_hours": round(lead, 1), "models_used": used, "steps": steps, "series": series,
                    "forecast": fc, "model": {"weights": W, "source": src}, "predictions": preds})
    json.dump({"generated_utc": now.isoformat(), "races": out}, open(os.path.join(ROOT, "data", "latest.json"), "w"), indent=1, default=str)
    if log:
        lp = os.path.join(ROOT, "data", "predictions_log.csv"); new = pd.DataFrame(log)
        old = pd.read_csv(lp) if os.path.exists(lp) else pd.DataFrame()
        if len(old): old = old[~old.race_id.isin(new.race_id.unique())]
        pd.concat([old, new], ignore_index=True).to_csv(lp, index=False)

if __name__ == "__main__": main()
