"""Grade the past week's lock-time boards against what actually happened.

    python grade_lock.py                  # the last 8 days of archived boards
    python grade_lock.py --since 2026-11-20

The same grading the NBA/NHL boards get every morning (DFS-ALLSPORTS
grade_nightly.py), run weekly because the NFL's stat lines arrive in a batch
early in the week. For every archived classic board (docs/data/archive/nfl,
the last version written before kickoff):

  players  - how far off the final projection, the model alone and the props
             alone were, and the regression of (actual - model) on
             (props - model): the weight the props deserved that slate.
  lineups  - the server's cash and tournament lineups' actual scores.

Each board is written to docs/data/grades/nfl/<date>-<draft group>.json and
docs/data/grades/nfl/summary.json pools them. props_weight.py reads that
summary, so the props blend learns its weight once enough is graded.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
import data
import scoring

log = logging.getLogger("grade_lock")
OUT = config.DOCS / "data" / "grades" / "nfl"
ARCH = config.DOCS / "data" / "archive" / "nfl"


def actuals(seasons: list[int]) -> pd.DataFrame:
    w = data.player_weeks(seasons)
    w["dk"] = scoring.score(w, "dk")
    return w


def grade_board(b: dict, act: pd.DataFrame) -> dict | None:
    starts = pd.Timestamp(b.get("starts")).tz_convert("America/New_York")
    # The NFL week a board belongs to: the stat rows whose game falls on the
    # board's days. nflverse rows carry season/week, not dates, so the week is
    # the one whose rows include the board's teams most recently before now.
    season = starts.year if starts.month >= 3 else starts.year - 1
    a = act[act["season"] == season]
    if a.empty:
        return None
    teams = {p["team"] for p in b["players"]}
    wk = int(a[a["team"].isin(teams)]["week"].max())
    a = a[a["week"] == wk]
    by = {}
    for r in a.itertuples(index=False):
        by[(data.normalise_name(r.name if hasattr(r, "name") else ""), str(r.team))] = float(r.dk)
    P = b["players"]
    for p in P:
        p["_act"] = by.get((data.normalise_name(p["name"]), p["team"]))
    played = [p for p in P if p["_act"] is not None and p.get("pos") not in ("DST", "K")]
    if len(played) < 10:
        return None

    def mae(rows, k):
        v = [abs(p["_act"] - float(p[k])) for p in rows if p.get(k) is not None]
        return round(float(np.mean(v)), 3) if v else None

    for p in played:
        pp = float(p.get("pplay") or 1.0)
        p["_final"] = float(p.get("mean") or 0) / max(pp, 0.05)
    out = {"date": str(starts.date()), "week": wk, "draft_group": b.get("draft_group"),
           "slate": b.get("label"), "graded": len(played),
           "mae_final": mae(played, "_final")}
    pr = [p for p in played if p.get("props") is not None and p.get("model") is not None]
    if pr:
        m = np.array([float(p["model"]) for p in pr])
        k = np.array([float(p["props"]) for p in pr])
        y = np.array([p["_act"] for p in pr])
        x, z = k - m, y - m
        out["props"] = {"n": len(pr), "mae_model": round(float(np.abs(y - m).mean()), 3),
                        "mae_props": round(float(np.abs(y - k).mean()), 3),
                        "sxy": float(x @ z), "sxx": float(x @ x),
                        "weight": round(float(x @ z / (x @ x)), 3) if x @ x > 0 else None}
    names = {p["name"]: p for p in P}
    for obj, lu in (b.get("server_lineups") or {}).items():
        sc = [names.get(n, {}).get("_act") for n in lu.get("players", [])]
        if all(v is not None for v in sc):
            out[f"lineup_{obj}"] = round(sum(sc), 2)
    return out


def summarise() -> dict:
    nights = []
    for f in sorted(OUT.glob("20*.json")):
        try:
            nights.append(json.loads(f.read_text()))
        except Exception:                                      # noqa: BLE001
            continue
    pr = [n["props"] for n in nights if n.get("props")]
    sxy, sxx = sum(p["sxy"] for p in pr), sum(p["sxx"] for p in pr)
    avg = lambda rows, k: (round(float(np.mean([r[k] for r in rows if r.get(k) is not None])), 3)
                           if any(r.get(k) is not None for r in rows) else None)
    s = {"sport": "nfl", "nights": len(nights), "props_nights": len(pr),
         "props_players": sum(p["n"] for p in pr),
         "props_weight": round(sxy / sxx, 3) if sxx > 0 else None,
         "mae_model": avg(pr, "mae_model"), "mae_props": avg(pr, "mae_props"),
         "updated": datetime.now(timezone.utc).isoformat(timespec="minutes")}
    (OUT / "summary.json").write_text(json.dumps(s, indent=1))
    return s


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--since")
    a = ap.parse_args(argv)
    since = a.since or (datetime.now(timezone.utc).date() - timedelta(days=8)).isoformat()
    boards = []
    for d in sorted(ARCH.glob("20*")):
        if d.name >= since:
            for f in d.glob("dk-classic-*.json"):
                boards.append(f)
    if not boards:
        print("no archived classic boards to grade")
        return 0
    yrs = sorted({int(f.parent.name[:4]) for f in boards} | {int(f.parent.name[:4]) - 1 for f in boards})
    act = actuals([y for y in yrs if y >= config.TRAIN_START_SEASON])
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in boards:
        try:
            g = grade_board(json.loads(f.read_text()), act)
        except Exception as exc:                               # noqa: BLE001
            log.warning("%s: %s", f.name, exc)
            continue
        if not g:
            log.info("%s: no stat lines yet", f.name)
            continue
        (OUT / f"{g['date']}-{g['draft_group']}.json").write_text(json.dumps(g, indent=1))
        n += 1
        pr = g.get("props") or {}
        print(f"- {g['date']} {g['slate']}: {g['graded']} players, final MAE {g['mae_final']}"
              + (f"; props {pr['n']}: model {pr['mae_model']} / props {pr['mae_props']}, "
                 f"weight {pr['weight']}" if pr else "")
              + f"; server lineups cash {g.get('lineup_cash')} / gpp {g.get('lineup_gpp')}")
    s = summarise()
    print(f"graded {n} boards; running props weight {s['props_weight']} over "
          f"{s['props_players']} players ({s['props_nights']} boards)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
