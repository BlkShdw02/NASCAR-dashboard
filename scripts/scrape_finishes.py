"""Cup finishes from Jayski -> data/finishes.csv, plus Open-Meteo archive weather for each race.
Flow: season page (2022..now) lists races with dates + 'Results' links -> fetch only races at tracks in
config/races.json -> parse the finishing-order table. Already-fetched results URLs are skipped, so weekly
runs only fetch new races. Jayski's terms ask that content not be duplicated/redistributed: keep the volume
small, keep the request delay, and don't republish their tables."""
import os, re, sys, json, time, datetime as dt, requests, pandas as pd
from bs4 import BeautifulSoup
ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "data", "finishes.csv"); CACHE = os.path.join(ROOT, "data", "weather_cache.json")
H = {"User-Agent": "nascar-dashboard personal project (malik.i.bryant@gmail.com)"}
BASE = "https://www.jayski.com"
DELAY = 3

def get(url):
    time.sleep(DELAY); r = requests.get(url, headers=H, timeout=30); r.raise_for_status(); return r.text

def season_urls(y):  # Jayski is inconsistent about the path, so try each
    return [f"{BASE}/race-results/{y}-nascar-cup-series-race-results/",
            f"{BASE}/nascar-cup-series/{y}-nascar-cup-series-race-results/",
            f"{BASE}/nascar-cup-series/{y}-nascar-cup-series-results/"]

def season_races(y, tracks):
    """-> [(date 'YYYY-MM-DD', config_track, results_url)] for points races at wanted tracks."""
    norm = {k.lower(): k for k in tracks}
    for u in season_urls(y):
        try: html = get(u)
        except requests.HTTPError as e:
            if e.response.status_code == 403: sys.exit("403 Forbidden from Jayski - blocked on this network")
            print(f"[debug] {y}: {u} -> HTTP {e.response.status_code}"); continue
        out, rows = [], BeautifulSoup(html, "lxml").select("table tr")
        for tr in rows:
            td = tr.find_all("td")
            if len(td) < 6 or not td[0].get_text(strip=True).isdigit(): continue   # skips Q / * non-points events
            trk = norm.get(td[2].get_text(strip=True).lower()); m = re.match(r"(\d{1,2})/(\d{1,2})", td[1].get_text(strip=True))
            link = next((a["href"] for a in td[5].find_all("a") if a.get_text(strip=True).lower() == "results"), None)
            if trk and m and link:
                out.append((f"{y}-{int(m.group(1)):02d}-{int(m.group(2)):02d}", trk, link if link.startswith("http") else BASE + link))
        if out: return out
        print(f"[debug] {y}: {u} -> page loaded, {len(rows)} table rows, 0 usable races")   # keep trying other URL patterns
        for tr in rows[:4]:
            print("   row:", [c.get_text(" ", strip=True)[:25] for c in tr.find_all(["td", "th"])][:7],
                  "links:", [a.get_text(strip=True) for a in tr.find_all("a")][:4])
    print(f"[warn] no season page worked for {y}", file=sys.stderr); return []

def parse_results(html):
    for t in BeautifulSoup(html, "lxml").find_all("table"):
        head = [h.get_text(strip=True).lower() for h in t.find_all("th")]
        if "fin" in head and "driver" in head:
            ix = {h: i for i, h in enumerate(head)}; rows = []
            for tr in t.find_all("tr"):
                td = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
                if len(td) >= len(head) and td[ix["fin"]].isdigit():
                    rows.append({"driver": td[ix["driver"]], "finish": int(td[ix["fin"]]),
                                 "start": int(td[ix["start"]]) if "start" in ix and td[ix["start"]].isdigit() else None,
                                 "status": td[ix["status"]] if "status" in ix else None})
            return rows
    return []

def race_weather(tr, date, cache):
    """Mean temp (F)/wind (mph), 1-6 PM track-local, from Open-Meteo's historical archive."""
    key = f"{tr['lat']},{tr['lon']}|{date}"
    if key in cache: return cache[key]
    j = None
    for attempt in range(4):                       # Open-Meteo can be slow/throttle shared GitHub IPs
        time.sleep(1 + attempt * 3)
        try:
            j = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=60, params={
                "latitude": tr["lat"], "longitude": tr["lon"], "start_date": date, "end_date": date,
                "hourly": "temperature_2m,wind_speed_10m", "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph", "timezone": tr["tz"]}).json().get("hourly"); break
        except requests.RequestException as e:
            if attempt == 3: raise
    if not j: return None    # archive lags ~2 days; filled on a later run
    h = pd.DataFrame(j); hr = pd.to_datetime(h["time"]).dt.hour; h = h[(hr >= 13) & (hr <= 18)]
    if h.temperature_2m.isna().all(): return None
    cache[key] = {"temp_f": round(h.temperature_2m.mean(), 1), "wind_mph": round(h.wind_speed_10m.mean(), 1)}
    return cache[key]

def add_weather(df, cfg):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    for c in ("temp_f", "wind_mph"):
        if c not in df: df[c] = None
    for (trk, date), idx in df[df.temp_f.isna()].groupby(["track", "date"]).groups.items():
        try:
            w = race_weather(cfg["tracks"][trk], date, cache)
            if w: df.loc[idx, "temp_f"], df.loc[idx, "wind_mph"] = w["temp_f"], w["wind_mph"]
        except Exception as e: print("[wx skip]", trk, date, e, file=sys.stderr)
    json.dump(cache, open(CACHE, "w")); return df

if __name__ == "__main__":
    cfg = json.load(open(os.path.join(ROOT, "config", "races.json"))); tracks = cfg["tracks"]
    have = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
    if "url" not in have: have = pd.DataFrame()      # older/other schema -> rebuild
    done = set(have["url"]) if len(have) else set(); new = []
    for y in range(2022, dt.date.today().year + 1):
        races = season_races(y, tracks); print(y, "races at configured tracks:", len(races))
        for date, trk, url in races:
            if url in done: continue
            if dt.date.fromisoformat(date) >= dt.date.today(): continue   # not run yet
            try: rows = parse_results(get(url))
            except Exception as e: print("[skip race]", url, e, file=sys.stderr); continue
            if not rows: print("[no table]", url, file=sys.stderr); continue
            d = pd.DataFrame(rows); d["date"], d["track"], d["series"] = date, trk, "Cup"
            d["track_type"], d["url"] = tracks[trk]["track_type"], url; new.append(d); print("  +", date, trk, len(d), "drivers")
    df = pd.concat([have] + new, ignore_index=True) if new or len(have) else pd.DataFrame()
    if df.empty: sys.exit("nothing scraped - check the log above")
    df = add_weather(df, cfg); df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} rows, {df.groupby(['track','date']).ngroups} races, weather on {df.temp_f.notna().sum()} rows")
