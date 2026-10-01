"""Weather ingest + finish prediction -> data/latest.json. Run on a schedule."""
import json, os, sys, datetime as dt, zoneinfo
import requests, pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
UA = {"User-Agent": "nascar-dashboard (malik.i.bryant@gmail.com)"}  # NWS requires a UA
# Open-Meteo model ids. VERIFY against open-meteo.com/en/docs if one returns empty.
OM = {"GFS": "gfs_global", "ECMWF": "ecmwf_ifs025", "NBM": "ncep_nbm_conus",
      "NAM": "ncep_nam_conus", "HRRR": "ncep_hrrr_conus"}
WEIGHT = {"HRRR": 2, "NAM": 1.5, "NBM": 1.3, "NDFD": 1, "ECMWF": 1.2, "GFS": 1}

def active(model, lead):
    return lead <= 48 if model == "HRRR" else lead <= 60 if model == "NAM" else True

def mean_in(df, col, a, b):
    s = df[(df.index >= a) & (df.index <= b)][col].dropna()
    return None if s.empty else float(s.mean())

def open_meteo(r, a, b):
    out = {}
    for name, mid in OM.items():
        try:
            j = requests.get("https://api.open-meteo.com/v1/forecast", timeout=30, params={
                "latitude": r["lat"], "longitude": r["lon"], "models": mid,
                "hourly": "temperature_2m,wind_speed_10m,precipitation_probability",
                "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                "timezone": "UTC", "forecast_days": 10}).json()["hourly"]
            df = pd.DataFrame(j); df.index = pd.to_datetime(df.pop("time"), utc=True)
            out[name] = {"temp": mean_in(df, "temperature_2m", a, b),
                         "wind": mean_in(df, "wind_speed_10m", a, b),
                         "precip": mean_in(df, "precipitation_probability", a, b)}
        except Exception as e:
            print(f"[warn] {name}: {e}", file=sys.stderr)
    return out

def ndfd(r, a, b):  # NWS point forecast is built from NDFD grids
    try:
        p = requests.get(f"https://api.weather.gov/points/{r['lat']},{r['lon']}", headers=UA, timeout=30).json()
        h = requests.get(p["properties"]["forecastHourly"], headers=UA, timeout=30).json()["properties"]["periods"]
        df = pd.DataFrame([{"t": pd.to_datetime(x["startTime"], utc=True), "temp": x["temperature"],
            "wind": float(str(x["windSpeed"]).split()[-2].split("-")[-1]),
            "precip": (x.get("probabilityOfPrecipitation") or {}).get("value")} for x in h]).set_index("t")
        return {"NDFD": {k: mean_in(df, k, a, b) for k in ("temp", "wind", "precip")}}
    except Exception as e:
        print(f"[warn] NDFD: {e}", file=sys.stderr); return {}

def blend(models, lead):
    res = {}
    for k in ("temp", "wind", "precip"):
        num = den = 0
        for m, v in models.items():
            if active(m, lead) and v.get(k) is not None:
                num += WEIGHT[m] * v[k]; den += WEIGHT[m]
        res[k] = round(num / den, 1) if den else None
    return res

def predict(r, fc, W=(0.45, 0.20, 0.20, 0.15)):
    # finishes.csv: date,track,series,track_type,driver,finish[,temp_f,wind_mph]
    f = pd.read_csv(os.path.join(ROOT, "data", "finishes.csv")); f["date"] = pd.to_datetime(f["date"])
    f = f[(f.series == r["series"]) & (f.date >= "2022-01-01")]
    here = f[f.track == r["track"]]
    mile = f[(f.track_type == r["track_type"]) & (f.date.dt.year == dt.date.today().year)]
    sim = here.iloc[0:0]
    if fc["temp"] is not None and {"temp_f", "wind_mph"} <= set(here.columns):
        s2 = here[(abs(here.temp_f - fc["temp"]) <= 10) & (abs(here.wind_mph - (fc["wind"] or 0)) <= 8)]
        if len(s2) >= 20: sim = s2
    recent = f.sort_values("date").groupby("driver").tail(5)
    g = lambda d: d.groupby("driver").finish.agg(["mean", "count"]) if len(d) else None
    T, S, M, R = g(here), g(sim), g(mile), g(recent)
    fm = here.finish.mean() if len(here) else 20
    rows = []
    for d in sorted(f.driver.unique()):
        v = [None]
        if T is not None and d in T.index:
            n = T.loc[d, "count"]; v[0] = (T.loc[d, "mean"] * n + fm * 3) / (n + 3)  # shrink small samples
        v += [X.loc[d, "mean"] if X is not None and d in X.index else None for X in (S, M, R)]
        w = [(x, wt) for x, wt in zip(v, W) if x is not None]
        score = sum(x * wt for x, wt in w) / sum(wt for _, wt in w) if w else fm
        rows.append({"driver": d, "score": round(score, 2), "track": v[0], "similar_wx": v[1], "mile_half": v[2], "form": v[3]})
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
        a, b = start, start + dt.timedelta(hours=4)
        models = {**open_meteo(r, a, b), **ndfd(r, a, b)}
        fc = blend(models, lead)
        out.append({"id": r["id"], "track": r["track"], "start_utc": start.isoformat(), "lead_hours": round(lead, 1),
                    "models_used": [m for m in models if active(m, lead)], "models_raw": models,
                    "forecast": fc, "predictions": predict(r, fc) if fin else []})
    json.dump({"generated_utc": now.isoformat(), "races": out}, open(os.path.join(ROOT, "data", "latest.json"), "w"), indent=1, default=str)

if __name__ == "__main__": main()
