"""Learning loop (run weekly after the finishes refresh):
 1. Backtest: re-predict every past race using only earlier data, measure error.
 2. Tune factor weights (needs 30+ backtest races; result is blended 50/50 with the defaults to limit overfitting).
 3. Live report card: grade the predictions ingest.py logged before each race against what actually happened.
Writes config/weights.json (read by ingest.py) and data/model_metrics.json (read by the dashboard)."""
import os, sys, json, datetime as dt
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); import model
ROOT = os.path.join(os.path.dirname(__file__), "..")

def live_eval(f, root):
    p = os.path.join(root, "data", "predictions_log.csv")
    if not os.path.exists(p): return []
    log = pd.read_csv(p); log["race_date"] = pd.to_datetime(log["race_date"]); out = []
    for rid, g in log.groupby("race_id"):
        d0, trk = g.race_date.iloc[0], g.track.iloc[0]
        act = f[(f.track == trk) & ((f.date - d0).abs() <= pd.Timedelta(days=1))].drop_duplicates("key")
        if act.empty: continue                                   # race not finished/scraped yet
        g = g.assign(key=g.driver.map(model.nkey)).drop_duplicates("key").set_index("key")
        a = act.set_index("key").finish.astype(float); both = g.index.intersection(a.index)
        mm = model.race_metrics(g.loc[both, "score"], a.loc[both])
        if not mm: continue
        top10 = len(set(g.loc[both].sort_values("pred").head(10).index) & set(a.loc[both].sort_values().head(10).index))
        win = a.loc[both].idxmin()
        out.append({"race_id": rid, "track": trk, "date": d0.strftime("%Y-%m-%d"), "drivers": int(len(both)),
                    "mae": round(mm[0], 2), "spearman": round(mm[1], 3), "top10_hits": int(top10),
                    "winner_pred_rank": int(g.loc[win, "pred"]) if win in g.index else None})
    return sorted(out, key=lambda x: x["date"])

def run(root):
    f = model.load_finishes(os.path.join(root, "data", "finishes.csv"))
    bt = model.build_backtest(f); n = len(bt)
    mae_d, sp_d = model.score_weights(bt, model.DEFAULT_W) if n else (None, None)
    W, mae_t, src = model.DEFAULT_W, mae_d, "default"
    if n >= 30:
        best, _ = model.tune(bt); W = tuple(round(0.5 * b + 0.5 * d, 3) for b, d in zip(best, model.DEFAULT_W))
        s = sum(W); W = tuple(round(w / s, 3) for w in W); mae_t = model.score_weights(bt, W)[0]; src = "tuned"
        if mae_t is None or mae_d is None or mae_t > mae_d - 0.05:     # only switch for a meaningful gain
            W, mae_t, src = model.DEFAULT_W, mae_d, "default"
    avg_field = float(np.mean([len(b["actual"]) for b in bt])) if n else 36
    json.dump({"weights": list(W), "source": src, "races": n, "updated": dt.date.today().isoformat()}, open(os.path.join(root, "config", "weights.json"), "w"), indent=1)
    live = live_eval(f, root)
    json.dump({"updated": dt.datetime.now(dt.timezone.utc).isoformat(), "weights": list(W), "weights_source": src,
               "backtest": {"races": n, "mae_default": mae_d, "mae_used": mae_t, "spearman_default": sp_d,
                            "random_mae": round(model.random_mae(avg_field), 2)}, "live": live},
              open(os.path.join(root, "data", "model_metrics.json"), "w"), indent=1)
    print(f"backtest races: {n}; avg rank error default={mae_d}, used={mae_t}; weights={W} ({src}); graded live races: {len(live)}")

if __name__ == "__main__": run(ROOT)
