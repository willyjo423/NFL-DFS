"""Write what the page needs, for every live slate.

Why the page does not just receive finished lineups
---------------------------------------------------
A precomputed lineup is a dead object. The moment you want to lock a player,
fade one, cap an exposure or change the entry count, it is worth nothing and
you are waiting on a server round trip that GitHub Pages cannot give you.

So this writes the INPUTS - each player's fitted quantiles, his ownership, and
the factor loadings that generate the correlation - and the page regenerates
the simulation and re-solves in the browser. That is only possible because the
correlation is a factor model: the pairwise matrix would have been 200x200
floats per slate and the raw simulation 40MB, while the loadings are a few
dozen numbers and the browser rebuilds 50,000 correlated draws from them in
about a second.

The server's own answer ships alongside, from the exact integer program, so
the page opens on a real lineup rather than an empty form and you can see
whether the browser's search found the same thing.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
import data
import factors
import ownership as OWN
import project as P
import simulate as S

log = logging.getLogger(__name__)

OUT = config.DOCS / "data"

# Quantiles are written at this precision. Two decimals on a fantasy score is
# a hundredth of a point - far finer than the model can distinguish - and it
# roughly halves the file.
ROUND = 2


def slate_payload(draft_group: int, site: str, showdown: bool,
                  sims: int, field_size: int) -> dict:
    roster = config.ROSTERS[(site, "Showdown Captain Mode" if showdown
                             else "Classic")]
    out = P.run(draft_group, site=site)
    pool = out["players"].reset_index(drop=True)

    own = OWN.project(pool, roster)
    pool["ownership"] = own.round(4)
    pool["leverage"] = OWN.leverage(pool, own)

    draws = S.simulate(pool, config.QUANTILES, sims)
    corr = S.correlation_report(pool, draws)

    import optimise as O
    lineups = {}
    for objective in ("cash", "gpp"):
        built = O.build(pool, roster, draws, objective=objective, entries=1,
                        own=own, field_size=field_size, n_candidates=120)
        lineups[objective] = {
            "players": built["name"].tolist(),
            "slots": built["slot"].tolist(),
            "salary": int(built["charged"].sum()),
            "mean": round(float(built["mean"].iloc[0]), 2),
            "floor": round(float(built["floor"].iloc[0]), 2),
            "ceiling": round(float(built["ceiling"].iloc[0]), 2),
            "p_cash": round(float(built["p_cash"].iloc[0]), 4),
            "p_gpp": round(float(built["p_gpp"].iloc[0]), 5),
            "duplicates": round(float(built["duplicates"].iloc[0]), 2),
        }

    qcols = [f"q{int(q * 100)}" for q in config.QUANTILES]
    players = []
    for r in pool.itertuples(index=False):
        players.append({
            "name": str(r.name),
            "pos": str(r.position),
            "team": str(r.team),
            "game": str(getattr(r, "game", "") or ""),
            "salary": int(r.salary),
            "cpt": (int(r.captain_salary)
                    if getattr(r, "captain_salary", None) == getattr(
                        r, "captain_salary", None) else None),
            "q": [round(float(getattr(r, c)), ROUND) for c in qcols],
            "med": round(float(r.median), ROUND),
            "ceil": round(float(r.ceiling), ROUND),
            "own": round(float(r.ownership), 4),
            "lev": round(float(r.leverage), 3),
            "status": str(getattr(r, "playing", "clear")),
        })

    # The loadings, so the browser can rebuild the same correlation. This is
    # the whole reason the page can re-simulate: three numbers per position
    # instead of a matrix with forty thousand entries in it.
    table = factors.LOADINGS.get("nfl", {})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sport": "nfl",
        "site": site,
        "draft_group": int(draft_group),
        "game_type": roster_name(roster),
        "roster": {
            "slots": roster["slots"],
            "flex_positions": roster.get("flex_positions", []),
            "salary_cap": roster["salary_cap"],
            "max_per_team": roster.get("max_per_team"),
            "captain_multiplier": roster.get("captain_multiplier"),
        },
        "quantiles": config.QUANTILES,
        "loadings": table,
        "field_size": field_size,
        "sims": sims,
        "players": players,
        "server_lineups": lineups,
        "correlation": corr.to_dict("records"),
        "notes": {
            "ownership": "modelled from priors, not measured - DraftKings' "
                         "standings are behind a login",
            "graded": False,
        },
    }


def roster_name(roster: dict) -> str:
    return "showdown" if "CPT" in roster["slots"] else "classic"


def run(kinds: list[str], site: str, sims: int, field_size: int) -> list[dict]:
    written = []
    for kind in kinds:
        try:
            dg, label = P.pick_slate(kind)
        except SystemExit as exc:
            log.warning("no live %s slate (%s)", kind, exc)
            continue
        log.info("building %s: %s (draft group %s)", kind, label, dg)
        try:
            payload = slate_payload(dg, site, kind == "showdown", sims,
                                    field_size)
        except Exception as exc:                      # noqa: BLE001
            # One dead slate must not take the whole publish down. The page
            # shows what it has, and the manifest records what failed.
            log.error("%s slate failed: %s: %s", kind, type(exc).__name__, exc)
            written.append({"kind": kind, "draft_group": dg, "label": label,
                            "error": f"{type(exc).__name__}: {exc}"})
            continue
        payload["label"] = label

        folder = OUT / payload["sport"]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{site}-{kind}-{dg}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")))
        log.info("wrote %s (%.0f KB, %d players)", path,
                 path.stat().st_size / 1024, len(payload["players"]))
        written.append({
            "kind": kind, "sport": payload["sport"], "site": site,
            "draft_group": dg, "label": label,
            "file": f"data/{payload['sport']}/{path.name}",
            "players": len(payload["players"]),
            "generated_at": payload["generated_at"],
        })
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sport", default="nfl")
    p.add_argument("--site", default="dk", choices=["dk", "fd"])
    p.add_argument("--kinds", nargs="*", default=["classic", "showdown"])
    p.add_argument("--sims", type=int, default=config.SIMS_GPP)
    p.add_argument("--field", type=int, default=100_000)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    OUT.mkdir(parents=True, exist_ok=True)
    written = run(args.kinds, args.site, args.sims, args.field)

    manifest = OUT / "manifest.json"
    prior = {}
    if manifest.exists():
        try:
            prior = json.loads(manifest.read_text())
        except json.JSONDecodeError:
            prior = {}

    # Keep yesterday's entries for sports not built in this run, so a football
    # publish does not silently delete basketball from the page.
    slates = {s["file"]: s for s in prior.get("slates", []) if "file" in s}
    for s in written:
        if "file" in s:
            slates[s["file"]] = s

    manifest.write_text(json.dumps({
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "slates": sorted(slates.values(),
                         key=lambda s: (s.get("sport", ""), s.get("kind", ""))),
        "failures": [s for s in written if "error" in s],
    }, indent=2))
    print(f"manifest -> {manifest}")
    for s in written:
        print(f"  {s.get('kind'):<10}{s.get('error') or s.get('file')}")
    return 0 if any("file" in s for s in written) else 1


if __name__ == "__main__":
    sys.exit(main())
