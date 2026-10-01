"""Racing Reference scraper + Open-Meteo archive weather -> data/finishes.csv (weekly, Mondays).
RR has no public API and its terms limit scraping: tiny volume, 5s delay, skip already-seen URLs.
UNTESTED against live HTML - run locally once and adjust race_links()/parse_race() if needed."""
import os, re, time, sys, json, datetime as dt, requests, pandas as pd
from bs4 import BeautifulSoup
ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "finishes.csv"); CACHE = os.path.join(ROOT, "data", "weather_cache.json")
H = {"User-Agent": "nascar-dashboard research (malik.i.bryant@gmail.com)"}
MONTHS = {m: i for i, m in enumerate(["january","february","march","april","may","june","july","august","september","october","november","december"], 1)}

def get(url):
    time.sleep(5); r = requests.get(url, headers=H, timeout=30); r.raise_for_status(); return r.text

def race_links(track_url):
    soup = BeautifulSoup(get(track_url), "lxml"); links = set()
    for a in soup.find_all("a", href=re.compile(r"/race-results/(\d{4})_")):
        if int(re.search(r"/race-results/(\d{4})_", a["href"]).group(1)) >= 2022:
            links.add(a["href"] if a["href"].startswith("http") else "https://www.racing-reference.info" + a["href"])
    return sorted(links)

def find_date(text, url):
    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(20\d\d)\b", text)
    if m: return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}", False
    m = re.search(r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2}),?\s+(20\d\d)", text, re.I)
    if m: return f"{m.group(3)}-{MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}", False
    return re.search(r"/(\d{4})_", url).group(1) + "-06-01", True   # approximate fallback

def parse_race(url):
    html = get(url); tbls = pd.read_html(html)
    t = next(x for x in tbls if "Driver" in map(str, x.columns) and any("Fin" in str(c) for c in x.columns))
    fin = next(c for c in t.columns if "Fin" in str(c))
    t = t[pd.to_numeric(t[fin], errors="coerce").notna()]
    date, approx = find_date(BeautifulSoup(html, "lxml").get_text(" "), url)
    return pd.DataFrame({"driver": t["Driver"], "finish": pd.to_numeric(t[fin])}), date, approx

def race_weather(tr, date, cache):
    """Mean temp (F) / wind (mph) 1-6 PM track-local from Open-Meteo's historical archive."""
    key = f"{tr['lat']},{tr['lon']}|{date}"
    if key in cache: return cache[key]
    j = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=30, params={
        "latitude": tr["lat"], "longitude": tr["lon"], "start_date": date, "end_date": date,
        "hourly": "temperature_2m,wind_speed_10m", "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph", "timezone": tr["tz"]}).json().get("hourly")
    if not j: return None            # archive lags ~2 days; retried next run
    df = pd.DataFrame(j); df["h"] = pd.to_datetime(df["time"]).dt.hour; df = df[(df.h >= 13) & (df.h <= 18)]
    cache[key] = {"temp_f": round(df.temperature_2m.mean(), 1), "wind_mph": round(df.wind_speed_10m.mean(), 1)}
    return cache[key]

if __name__ == "__main__":
    cfg = json.load(open(os.path.join(ROOT, "config", "races.json")))
    have = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
    done = set(have["url"]) if "url" in have else set(); new = []
    for name, tr in cfg["tracks"].items():
        try: links = race_links(tr["rr_url"])
        except Exception as e: print("[skip track]", name, e, file=sys.stderr); continue
        for u in links:
            if u in done: continue
            try:
                d, date, approx = parse_race(u)
                d["url"], d["track"], d["series"], d["track_type"] = u, name, "Cup", tr["track_type"]
                d["date"], d["date_approx"] = date, approx; new.append(d)
            except Exception as e: print("[skip race]", u, e, file=sys.stderr)
    df = pd.concat([have] + new, ignore_index=True)
    if df.empty: sys.exit("no data scraped - check rr_url values in config/races.json")
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    for c in ("temp_f", "wind_mph"):
        if c not in df: df[c] = None
    for (trk, date), idx in df[df.temp_f.isna()].groupby(["track", "date"]).groups.items():
        try:
            w = race_weather(cfg["tracks"][trk], date, cache)
            if w: df.loc[idx, ["temp_f", "wind_mph"]] = w["temp_f"], w["wind_mph"]
        except Exception as e: print("[wx skip]", trk, date, e, file=sys.stderr)
    df.to_csv(OUT, index=False); json.dump(cache, open(CACHE, "w"))
