"""NFL player props from Bovada, blended into the projections.

Same idea as the NBA and NHL boards (DFS-ALLSPORTS): the books' player lines
are turned into an expected DraftKings score and blended with the model, the
props carrying most of the weight. Graded on NBA (2026-11-18/19, 167 players)
the props were the more accurate of the two.

From each game Bovada posts, per player (full game only):
  passing yards / TDs / interceptions, rushing yards, receptions, receiving
  yards (each as an over/under plus an alternate ladder), and the anytime /
  2+ / 3+ touchdown scorer prices.

Counts (TDs, receptions, interceptions) use the posted tail probabilities
directly, E[X] = sum P(X >= k), with a Poisson fill for rungs not posted.
Yards fit a normal to every posted rung (probit on the line), which also
gives the chance of the 100-yard (300 passing) bonus.

DraftKings classic: 0.04/pass yd, 4/pass TD, -1/INT, +3 at 300 pass yds;
0.1/rush or rec yd, 1/reception, 6/rush or rec TD, +3 at 100 rush or rec yds.

Each day's props are saved under docs/data/props/nfl/<date>.json, as a
fallback and as the record the weekly grade learns the weight from.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from statistics import NormalDist

log = logging.getLogger("nfl_props")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "docs", "data", "props", "nfl")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124 Safari/537.36")
LIST_URL = ("https://www.bovada.lv/services/sports/event/coupon/events/A/description/"
            "football/nfl?marketFilterId=def&preMatchOnly=true&lang=en")
EVENT_URL = "https://www.bovada.lv/services/sports/event/v2/events/A/description{}?lang=en"
N = NormalDist()
ONE_SIDED_POW = 1.12          # power de-vig for "yes"-only prices

import props_weight as _PW    # noqa: E402
W_MARKET = _PW.weight("nfl", 0.85)
W_GAP_MAX = min(0.97, max(W_MARKET + 0.10, 0.95))

# "Total Passing Yards - Carson Beck (ARI)"; "Alternate Receptions - X (KC)"
_WHO = re.compile(r"^(?P<name>.+?)\s*\((?P<team>[A-Z]{2,4})\)\s*$")
STATS = [("passing yards", "PY"), ("passing touchdowns", "PTD"),
         ("interceptions thrown", "INT"), ("rushing & receiving yards", None),
         ("rushing yards", "RY"), ("receiving yards", "RECY"),
         ("receptions", "REC")]
YARD_STATS = {"PY", "RY", "RECY"}
_CACHE: dict = {}


def _prob(a) -> float | None:
    if a in (None, ""):
        return None
    a = str(a).upper()
    if a == "EVEN":
        return 0.5
    v = float(a)
    return 100 / (v + 100) if v > 0 else -v / (-v + 100)


def _who(text: str):
    text = text.replace("\xa0", " ").strip()
    m = _WHO.match(text)
    return (m["name"].strip(), m["team"]) if m else (None, None)


def _stat(text: str):
    t = text.lower()
    for key, code in STATS:
        if key in t:
            return code
    return None


def parse_event(ev: dict) -> dict:
    """{(name, team): {stat: [(k, P(X >= k)), ...]}} for one game. Yards are
    stored the same way, k being the yardage."""
    out: dict = {}

    def add(name, team, stat, k, p):
        if name and p is not None and 0.003 < p < 0.997:
            out.setdefault((name, team), {}).setdefault(stat, []).append((float(k), p))

    for g in ev.get("displayGroups", []):
        for m in g.get("markets", []):
            if (m.get("period") or {}).get("description") not in ("Game", None):
                continue
            d = m.get("description", "").replace("\xa0", " ").strip()
            low = d.lower()
            if "longest" in low or "attempts" in low or "completions" in low \
                    or "first touchdown" in low or "last touchdown" in low:
                continue
            # Touchdown scorer lists: outcomes are the players.
            tdk = (1 if low.startswith("anytime touchdown") else
                   2 if "2 or more touchdowns" in low else
                   3 if "3 or more touchdowns" in low else None)
            if tdk:
                for o in m.get("outcomes", []):
                    nm, tm = _who(o.get("description", ""))
                    p = _prob((o.get("price") or {}).get("american"))
                    add(nm, tm, "TD", tdk, p ** ONE_SIDED_POW if p else None)
                continue
            if " - " not in d:
                continue
            kind, who = d.split(" - ", 1)
            stat = _stat(kind)
            if not stat:
                continue
            name, team = _who(who)
            if not name:
                continue
            sides: dict = {}
            for o in m.get("outcomes", []):
                pr = o.get("price") or {}
                desc = (o.get("description") or "").strip()
                p = _prob(pr.get("american"))
                mm = re.search(r"(\d+)\+", desc)
                if mm and not desc.lower().startswith(("over", "under")):
                    add(name, team, stat, int(mm.group(1)), p ** ONE_SIDED_POW if p else None)
                    continue
                hc = pr.get("handicap")
                if hc in (None, ""):
                    continue
                side = "O" if desc.lower().startswith("over") or o.get("type") == "O" else "U"
                sides.setdefault(float(hc), {})[side] = p
            for ln, s in sides.items():
                if s.get("O") and s.get("U"):
                    # P(X > line) = P(X >= floor(line) + 1) for a half-point line
                    add(name, team, stat, math.floor(ln) + 1, s["O"] / (s["O"] + s["U"]))
    return out


# --------------------------------------------------------------- the maths
def poisson_tail(lam: float, k: int) -> float:
    if k <= 0:
        return 1.0
    term, cdf = math.exp(-lam), 0.0
    for i in range(int(k)):
        cdf += term
        term *= lam / (i + 1)
    return max(0.0, 1.0 - cdf)


def count_expected(points) -> float | None:
    """E[X] = sum_k P(X >= k): posted tails where they exist, a fitted
    Poisson elsewhere."""
    pts = [(int(k), p) for k, p in points if 0.003 < p < 0.997]
    if not pts:
        return None

    def logit(x):
        x = min(max(x, 1e-6), 1 - 1e-6)
        return math.log(x / (1 - x))
    best, err_best, lam = 0.5, float("inf"), 0.01
    while lam < 40:
        err = sum((logit(poisson_tail(lam, k)) - logit(p)) ** 2 for k, p in pts)
        if err < err_best:
            best, err_best = lam, err
        lam *= 1.02
    obs: dict = {}
    for k, p in pts:
        obs.setdefault(k, []).append(p)
    e, prev = 0.0, 1.0
    for k in range(1, max(max(obs), int(best * 3) + 6) + 1):
        p = sum(obs[k]) / len(obs[k]) if k in obs else poisson_tail(best, k)
        p = min(p, prev)
        e += p
        prev = p
    return e


DEFAULT_SD = {"PY": lambda m: 0.22 * m + 25, "RY": lambda m: 0.50 * m + 8,
              "RECY": lambda m: 0.55 * m + 8}


def yards_fit(stat: str, points) -> tuple[float, float] | None:
    """(mean, sd) of a normal fitted to P(X >= k) at each posted yardage."""
    pts = [(k, N.inv_cdf(1 - p)) for k, p in points if 0.01 < p < 0.99]
    if not pts:
        return None
    if len(pts) >= 2:
        xs = [k for k, _ in pts]
        zs = [z for _, z in pts]
        mx, mz = sum(xs) / len(xs), sum(zs) / len(zs)
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx > 0:
            b = sum((x - mx) * (z - mz) for x, z in zip(xs, zs)) / sxx
            if b > 0:
                a = mz - b * mx
                mu, sd = -a / b, 1 / b
                lo, hi = min(xs), max(xs)
                return max(0.0, min(max(mu, lo - 60), hi + 60)), max(sd, 3.0)
    k, z = min(pts, key=lambda t: abs(t[1]))
    mu = k
    for _ in range(4):                  # z = (k - mu) / sd(mu)
        mu = k - z * DEFAULT_SD[stat](max(mu, 1))
    return max(0.0, mu), DEFAULT_SD[stat](max(mu, 1))


def market_dk(stats: dict, pos: str) -> dict | None:
    """Expected DraftKings points from a player's props, and how complete the
    picture was (`share`, 0-1)."""
    pos = (pos or "").upper()
    py = yards_fit("PY", stats["PY"]) if "PY" in stats else None
    ry = yards_fit("RY", stats["RY"]) if "RY" in stats else None
    rcy = yards_fit("RECY", stats["RECY"]) if "RECY" in stats else None
    rec = count_expected(stats["REC"]) if "REC" in stats else None
    ptd = count_expected(stats["PTD"]) if "PTD" in stats else None
    it = count_expected(stats["INT"]) if "INT" in stats else None
    td = count_expected(stats["TD"]) if "TD" in stats else None

    def over(fit, line):
        return 0.0 if not fit else 1 - N.cdf((line - fit[0]) / fit[1])

    if pos == "QB":
        if py is None or ptd is None:
            return None
        need = [ry, it, td]
    elif pos == "RB":
        if ry is None or td is None:
            return None
        need = [rec, rcy]
    elif pos in ("WR", "TE"):
        if rcy is None or rec is None or td is None:
            return None
        need = [ry]                     # rarely posted; usually ~0 anyway
        need = [] if pos == "TE" else need
    else:
        return None
    dk = 0.0
    if py:
        dk += 0.04 * py[0] + 3 * over(py, 300)
    if ptd is not None:
        dk += 4 * ptd
    if it is not None:
        dk -= it
    elif pos == "QB":
        dk -= 0.7                       # league-ish INTs per start
    if ry:
        dk += 0.1 * ry[0] + 3 * over(ry, 100)
    if rcy:
        dk += 0.1 * rcy[0] + 3 * over(rcy, 100)
    if rec is not None:
        dk += rec
    if td is not None:
        dk += 6 * td
    missing = sum(1 for x in need if x is None)
    share = 1.0 if pos in ("WR", "TE") else max(0.5, 1.0 - 0.2 * missing)
    return {"dk": round(dk, 2), "share": share}


# ------------------------------------------------------------- fetching
def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def fetch() -> dict:
    lst = _get(LIST_URL)
    out = {}
    for e in (e for blk in lst for e in blk.get("events", [])):
        try:
            full = _get(EVENT_URL.format(e["link"]))
            for ev in (ev for blk in full for ev in blk.get("events", [])):
                for (nm, tm), st in parse_event(ev).items():
                    out[f"{nm}|{tm or ''}"] = st
        except Exception as exc:                               # noqa: BLE001
            log.warning("bovada event %s failed (%s)", e.get("description"), str(exc)[:60])
        time.sleep(0.3)
    return out


def load(day: str | None = None) -> dict:
    day = day or datetime.now(timezone(timedelta(hours=-5))).date().isoformat()
    if day in _CACHE:
        return _CACHE[day]
    path = os.path.join(CACHE_DIR, f"{day}.json")
    props: dict = {}
    for attempt in range(3):            # Bovada now and then answers empty
        try:
            props = fetch()
        except Exception as exc:                               # noqa: BLE001
            log.warning("bovada unreachable (%s)", str(exc)[:80])
        if props:
            break
        time.sleep(5 * (attempt + 1))
    old: dict = {}
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f).get("players", {})
    if not props and not old:
        # Props for a Sunday are posted all week: the newest earlier file is
        # still this week's lines.
        prior = sorted(p for p in os.listdir(CACHE_DIR) if p < f"{day}.json") \
            if os.path.isdir(CACHE_DIR) else []
        if prior and prior[-1] >= (datetime.fromisoformat(day) - timedelta(days=3)).date().isoformat():
            with open(os.path.join(CACHE_DIR, prior[-1])) as f:
                old = json.load(f).get("players", {})
    if props:
        old.update(props)
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "players": old}, f, separators=(",", ":"), sort_keys=True)
        log.info("bovada: NFL props for %d players", len(props))
    elif old:
        log.info("bovada: using saved NFL props (%d players)", len(old))
    else:
        log.warning("bovada: no NFL props")
    _CACHE[day] = old
    return old


# ------------------------------------------------------------- blending
def weight(model: float, market: float) -> float:
    if model <= 0:
        return W_MARKET
    gap = abs(market - model) / model
    if gap <= 0.10:
        return W_MARKET
    return W_MARKET + (W_GAP_MAX - W_MARKET) * min(1.0, (gap - 0.10) / 0.15)


def _norm(s: str) -> str:
    s = re.sub(r"[.'`’]", "", str(s).lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def apply(pool, qcols: list[str]) -> int:
    """Blend props into a project.run() pool, in place. Returns the count.

    The quantile curves, median, ceiling and means are scaled together, so
    the simulation and the page see the blended number. The model's own
    conditional mean and the props number are kept in `model` / `props`,
    which the simulation uses for its 'model was right about this team'
    scenario."""
    import numpy as np
    props = load()
    pool["model"] = np.nan
    pool["props"] = np.nan
    if not props:
        return 0
    idx: dict = {}
    for key, st in props.items():
        nm, tm = key.rsplit("|", 1)
        idx.setdefault(_norm(nm), []).append((tm, st))
    n, gaps = 0, []
    for i, r in pool.iterrows():
        pos = str(r.get("position") or "")
        if pos in ("DST", "K") or str(r.get("playing") or "clear") == "out":
            continue
        cands = idx.get(_norm(r["name"]), [])
        if not cands:
            continue
        st = next((s for t, s in cands if t == str(r.get("team"))), cands[0][1])
        mk = market_dk(st, pos)
        if not mk:
            continue
        cond = float(r.get("cond_mean") if r.get("cond_mean") == r.get("cond_mean")
                     and r.get("cond_mean") is not None else r.get("median") or 0)
        if cond <= 0:
            continue
        w = weight(cond, mk["dk"]) * min(1.0, mk["share"])
        target = (1 - w) * cond + w * mk["dk"]
        f = target / cond
        for c in list(qcols) + ["median", "ceiling", "cond_mean", "mean"]:
            if c in pool.columns and pool.at[i, c] == pool.at[i, c]:
                pool.at[i, c] = float(pool.at[i, c]) * f
        pool.at[i, "model"] = round(cond, 2)
        pool.at[i, "props"] = round(mk["dk"], 2)
        gaps.append(mk["dk"] - cond)
        n += 1
    if n:
        log.info("props: blended %d players, props - model mean %+.2f, mean |gap| %.2f",
                 n, sum(gaps) / n, sum(abs(g) for g in gaps) / n)
    return n
