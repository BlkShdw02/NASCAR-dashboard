"""One-time/occasional: derive each track's straightaway heading from OpenStreetMap (free Overpass API).
Finds the closed raceway loop whose perimeter matches the track's length_mi, locates its longest straight section,
and reports the compass heading cars travel (counter-clockwise) on that straight. Which of the two long straights is
the FRONT stretch can't be read from OSM, so the dashboard has a Flip button. Writes config/orientation.json."""
import os, sys, json, math, time, requests
ROOT = os.path.join(os.path.dirname(__file__), "..")
URL = "https://overpass-api.de/api/interpreter"; H = {"User-Agent": "nascar-dashboard (malik.i.bryant@gmail.com)"}

def overpass(lat, lon, radius=2500):
    q = (f'[out:json][timeout:60];(way(around:{radius},{lat},{lon})["highway"="raceway"];'
         f'way(around:{radius},{lat},{lon})["leisure"="track"]["sport"~"motor|racing|auto"];);out geom;')
    r = requests.post(URL, data={"data": q}, headers=H, timeout=90); r.raise_for_status(); return r.json().get("elements", [])

def to_xy(pts, lat0, lon0):
    return [((lo - lon0) * math.cos(math.radians(lat0)) * 111320, (la - lat0) * 110540) for la, lo in pts]

def perimeter(xy): return sum(math.dist(xy[i], xy[i + 1]) for i in range(len(xy) - 1))
def brg(a, b): return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360
def adiff(a, b): return abs((a - b + 180) % 360 - 180)

def longest_straight(xy, tol=5):
    n = len(xy) - 1; segs = [(brg(xy[i], xy[i + 1]), math.dist(xy[i], xy[i + 1]), i) for i in range(n)]; best = (0, None, None)
    for i in range(n):
        ref, tot, mid = segs[i][0], 0, []
        for k in range(n):                                  # walk around the loop (wraps past the start point)
            b, d, idx = segs[(i + k) % n]
            if adiff(b, ref) > tol: break
            tot += d; mid.append(idx)
        if tot > best[0]: best = (tot, ref, (mid[0], mid[-1]))
    return best

def heading(xy):
    tot, ref, (i0, i1) = longest_straight(xy); n = len(xy) - 1
    a, b = xy[i0 % n], xy[(i1 + 1) % n if (i1 + 1) % n else n]; m = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    cx = sum(p[0] for p in xy[:-1]) / n; cy = sum(p[1] for p in xy[:-1]) / n
    d = (math.sin(math.radians(ref)), math.cos(math.radians(ref)))                # east, north components
    cross = d[0] * (cy - m[1]) - d[1] * (cx - m[0])                                # >0: centre is on the left
    return (ref if cross > 0 else (ref + 180) % 360), tot                          # counter-clockwise => centre on the left

def run(root=ROOT, fetch=overpass):
    cfg = json.load(open(os.path.join(root, "config", "races.json")))["tracks"]; out = {}
    for name, t in cfg.items():
        try: els = fetch(t["lat"], t["lon"])
        except Exception as e: print("[skip]", name, e, file=sys.stderr); continue
        want = t.get("length_mi", 1.5) * 1609.34; best = None
        for el in els:
            pts = [(g["lat"], g["lon"]) for g in el.get("geometry", [])]
            if len(pts) < 8 or pts[0] != pts[-1]: continue                          # closed loops only
            xy = to_xy(pts, t["lat"], t["lon"]); per = perimeter(xy); err = abs(per - want) / want
            if err <= 0.25 and (best is None or err < best[0]): best = (err, el["id"], xy, per)
        if not best: print(f"[no match] {name}: no closed loop near {t.get('length_mi')} mi found in OSM", file=sys.stderr); continue
        h, straight = heading(best[2])
        out[name] = {"heading_deg": round(h), "source": "osm", "way_id": best[1], "perimeter_mi": round(best[3] / 1609.34, 2), "longest_straight_m": round(straight)}
        print(f"{name}: heading {round(h)} deg (OSM way {best[1]}, loop {out[name]['perimeter_mi']} mi, straight {round(straight)} m)")
        time.sleep(5)
    json.dump(out, open(os.path.join(root, "config", "orientation.json"), "w"), indent=1)

if __name__ == "__main__": run()
