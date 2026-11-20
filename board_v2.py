"""The board in the same shape as the NBA/NHL/CFB site (DFS-ALLSPORTS), so
the NFL page looks and works exactly like the others: a pool of lineups
(engine lineup_pool, with QB-stack and bring-back families), each ranked by
its simulated Top-1%, Top-10% and Cash rates against a field that rosters
players as often as our ownership estimates say.

Written to docs/data/v2/ beside the original files, which the in-browser
simulator page (advanced.html) still reads.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
import lineup_pool as LP

log = logging.getLogger("board_v2")
OUT = config.DOCS / "data" / "v2"
FIELD = 5000
RANK_SIMS = 30000   # same as the NHL board; 25-30k made the top-5 stable


def _opp(team: str, game: str) -> str:
    g = str(game or "").replace(" vs ", " @ ").replace("@", " @ ")
    parts = [x.strip() for x in g.split("@") if x.strip()]
    if len(parts) == 2:
        return parts[1] if parts[0] == team else parts[0]
    return ""


def players(pool: pd.DataFrame, own: pd.Series, chalk: list[str]) -> list[dict]:
    out = []
    for i, r in pool.reset_index(drop=True).iterrows():
        pp = float(r.get("p_play")) if r.get("p_play") == r.get("p_play") and r.get("p_play") is not None else 1.0
        med = float(r.get("median") or 0)
        mean = float(r.get("mean") if r.get("mean") == r.get("mean") and r.get("mean") is not None else med * pp)
        st = str(r.get("playing") or "clear")
        news = str(r.get("news") or "")
        inj = str(r.get("injury_status") or "")
        if inj and inj.lower() not in ("nan", "none"):
            news = "; ".join(x for x in (inj, news) if x)
        pid = r.get("dk_player_id")
        out.append({
            "name": str(r["name"]), "pos": str(r["position"]), "team": str(r["team"]),
            "opp": _opp(str(r["team"]), str(r.get("game") or "")),
            "game": str(r.get("game") or ""),
            "pid": (str(int(pid)) if pid == pid and pid is not None and str(pid) not in ("", "nan") else None),
            "salary": int(r["salary"]),
            "q": [round(float(r[f"q{int(q * 100)}"]), 2) for q in config.QUANTILES],
            "med": round(med, 2), "proj": round(mean, 2), "mean": round(mean, 2),
            "ceil": round(float(r.get("ceiling") or med), 2),
            "own": round(float(own.iloc[i]), 5),
            "lev": round(float(r.get("leverage") or 0), 3),
            "pp": round(pp, 3),
            "status": "out" if st == "out" else "clear",
            "doubtful": st in ("questionable", "doubtful") or pp < 0.7,
            "chalk": str(r["name"]) in chalk,
            "news": news or None,
            "model": (round(float(r["model"]), 2) if "model" in r and r["model"] == r["model"] else None),
            "props": (round(float(r["props"]), 2) if "props" in r and r["props"] == r["props"] else None),
        })
    return out


def _field(P: list[dict], roster: dict, size: int, rng, adjust=None) -> np.ndarray:
    """Lineups drawn slot by slot with probability in proportion to ownership,
    legal on salary. Rough, but it rosters players about as often as our
    estimates, which is what the rates are measured against."""
    slots = roster["slots"]
    flexp = set(roster.get("flex_positions") or [])
    own = np.array([max(p["own"], 0.0) for p in P]) + 1e-5
    if adjust is not None:
        own = own * adjust
    sal = np.array([p["salary"] for p in P])
    pos = np.array([p["pos"] for p in P])
    ok = np.array([p["status"] != "out" for p in P])
    cap = roster.get("salary_cap", 50000)
    elig = {s: np.where(ok & ((pos == s) if s != "FLEX" else np.isin(pos, list(flexp))))[0]
            for s in set(slots)}
    cheapest = int(sal[ok].min()) if ok.any() else 0
    out, tries = [], 0
    order = sorted(range(len(slots)), key=lambda k: slots[k] == "FLEX")
    while len(out) < size and tries < size * 40:
        tries += 1
        pick, spent = [], 0
        for n_done, k in enumerate(order):
            c = elig[slots[k]]
            c = c[~np.isin(c, pick)]
            left = len(order) - n_done - 1
            c = c[sal[c] <= cap - spent - cheapest * left]
            if not len(c):
                break
            w = own[c] / own[c].sum()
            j = int(rng.choice(c, p=w))
            pick.append(j)
            spent += sal[j]
        if len(pick) == len(slots) and 0.92 * cap <= spent <= cap:
            out.append(pick)
    return np.array(out, dtype=int)


def calibrated_field(P, roster, rng):
    """A field whose rostering matches the ownership estimates: sampling
    lineups drawn by ownership, then raked (reweighted until the weighted
    frequencies match the estimates - salary rules otherwise starve the
    chalk: Gibbs came out at 33% against a 65% estimate) and redrawn. Same as the NBA/NHL sims."""
    target = np.array([max(p["own"], 0.0) for p in P])
    adjust = np.ones(len(P))
    for _ in range(5):
        F0 = _field(P, roster, 1500, rng, adjust)
        if len(F0) < 300:
            break
        freq = np.bincount(F0.ravel(), minlength=len(P)) / len(F0)
        adjust *= np.clip((target + 0.003) / (freq + 0.003), 0.25, 4.0) ** 0.8
    F = _field(P, roster, FIELD, rng, adjust)
    if len(F) < 300:
        F = _field(P, roster, FIELD, rng)
    if len(F) < 300:
        return F
    w = np.ones(len(F))
    for _ in range(80):
        freq = np.bincount(F.ravel(), weights=np.repeat(w, F.shape[1]), minlength=len(P)) / w.sum()
        r = np.log(np.clip((target + 0.003) / (freq + 0.003), 0.2, 5.0))
        w *= np.exp(r[F].mean(axis=1))
        w = np.clip(w / w.mean(), 0.02, 50.0)
    F = F[rng.choice(len(F), size=len(F), p=w / w.sum())]
    freq = np.bincount(F.ravel(), minlength=len(P)) / len(F)
    top = np.argsort(-target)[:12]
    log.info("v2 field ownership vs estimate (top 12): mean |gap| %.1f pts",
             float(np.abs(freq[top] - target[top]).mean() * 100))
    return F


def rank(lineups: list[dict], P: list[dict], roster: dict, draws: np.ndarray, rng) -> dict:
    sims = min(RANK_SIMS, draws.shape[1])
    D = draws[:, :sims]
    F = calibrated_field(P, roster, rng)
    if len(F) < 300:
        log.warning("v2: field too small (%d); lineups unranked", len(F))
        return {}
    q99, q90, q55 = np.empty(sims), np.empty(sims), np.empty(sims)
    for a in range(0, sims, 500):
        fs = D[:, a:a + 500][F].sum(axis=1)            # field x chunk
        q99[a:a + 500] = np.quantile(fs, 0.99, axis=0)
        q90[a:a + 500] = np.quantile(fs, 0.90, axis=0)
        q55[a:a + 500] = np.quantile(fs, 0.55, axis=0)
    for L in lineups:
        s = D[L["p"]].sum(axis=0)
        L["top1"] = round(float((s > q99).mean() * 100), 2)
        L["top10"] = round(float((s > q90).mean() * 100), 1)
        L["cash"] = round(float((s > q55).mean() * 100), 1)
    return {"sims": int(sims), "field": int(len(F))}


def write(base: dict, pool: pd.DataFrame, own: pd.Series, chalk: list[str],
          draws: np.ndarray) -> dict | None:
    """One classic slate in the v2 shape. Returns its manifest row."""
    roster = {"slots": base["roster"]["slots"], "salary_cap": base["roster"]["salary_cap"],
              "flex_positions": base["roster"].get("flex_positions") or ["RB", "WR", "TE"],
              "max_per_team": base["roster"].get("max_per_team"),
              "no_hitter_vs_pitcher": False}
    P = players(pool, own, chalk)
    lineups = LP.build(P, roster, int(base.get("field_size") or 100_000))
    lineups = [L for L in lineups if len(L["p"]) == len(roster["slots"])]
    sim = rank(lineups, P, roster, draws, np.random.default_rng(17)) if lineups else {}
    payload = {
        "generated_at": base["generated_at"], "sport": "nfl", "site": "dk", "kind": "classic",
        "draft_group": base["draft_group"], "label": base.get("label"),
        "locks": pd.Timestamp(base["starts"]).isoformat() if base.get("starts") else None,
        "field_size": base.get("field_size"), "roster": roster, "players": P,
        "lineups": lineups, "sim": sim or None,
        "props": base.get("props"),
        "market": {"source": "nflverse / live odds"},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"nfl_dk_classic_{base['draft_group']}.json"
    (OUT / name).write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    log.info("v2: %s - %d players, %d lineups, sim %s", name, len(P), len(lineups), sim)
    return {"sport": "nfl", "site": "dk", "kind": "classic", "label": base.get("label"),
            "draft_group": base["draft_group"], "file": f"data/v2/{name}",
            "starts": base.get("starts"), "players": len(P),
            "generated_at": base["generated_at"]}


def manifest(rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "manifest.json"
    prior = {}
    if path.exists():
        try:
            prior = {s["file"]: s for s in json.loads(path.read_text()).get("slates", [])}
        except Exception:                                      # noqa: BLE001
            prior = {}
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=12)

    def live(s):
        try:
            return pd.Timestamp(s.get("starts")).tz_convert("UTC") > cutoff
        except Exception:                                      # noqa: BLE001
            return False
    slates = {f: s for f, s in prior.items() if live(s)}
    for r in rows:
        slates[r["file"]] = r
    keep = {config.DOCS / f for f in slates}
    for f in OUT.glob("nfl_dk_*.json"):
        if f not in keep:
            f.unlink()
    order = sorted(slates.values(), key=lambda s: (str(s.get("starts")), -int(s.get("players") or 0)))
    path.write_text(json.dumps({"updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "slates": order}, indent=1))
