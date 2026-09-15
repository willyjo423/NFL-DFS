"""Projections to lineups, for one live slate.

    python lineups.py --draft-group 153086 --entries 1
    python lineups.py --draft-group 153086 --entries 3 --objective gpp

Runs the whole chain: fit the quantile models, pull the slate, simulate it
fifty thousand times with the players correlated, enumerate legal lineups, and
choose between them by what each contest actually pays for.

Cash and GPP are built in parallel and printed together, because the most
useful thing about them is the difference: if the two lineups look the same,
either the slate has no leverage in it or something upstream is broken.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

import pandas as pd

import config
import optimise as O
import ownership as OWN
import project as P
import simulate as S

log = logging.getLogger(__name__)


def game_type(showdown: bool) -> str:
    return "Showdown Captain Mode" if showdown else "Classic"


def run(draft_group: int, site: str = "dk", showdown: bool = True,
        entries: int = 1, sims: int | None = None,
        n_candidates: int = 150, field_size: int = 100_000) -> dict:
    roster = config.ROSTERS[(site, game_type(showdown))]

    out = P.run(draft_group, site=site)
    pool = out["players"].reset_index(drop=True)
    log.info("%d projectable players in the pool", len(pool))

    size = len(roster["slots"])
    if len(pool) < size:
        raise SystemExit(f"only {len(pool)} players projected; a lineup needs "
                         f"{size}")

    sims = sims or config.SIMS_GPP
    log.info("simulating the slate %d times", sims)
    draws = S.simulate(pool, config.QUANTILES, sims)

    report = S.correlation_report(pool, draws)
    log.info("correlation delivered vs asked:\n%s",
             report.to_string(index=False,
                              float_format=lambda v: f"{v:6.3f}"))

    own = OWN.project(pool, roster)
    pool["ownership"] = own
    pool["leverage"] = OWN.leverage(pool, own)
    log.info("ownership is MODELLED, not measured - DraftKings' standings are "
             "behind a login, so every leverage number below is an estimate")

    built = {}
    for objective in ("cash", "gpp"):
        built[objective] = O.build(pool, roster, draws, objective=objective,
                                   entries=entries, n_candidates=n_candidates,
                                   own=own, field_size=field_size)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "draft_group": draft_group,
        "site": site,
        "roster": roster,
        "sims": sims,
        "pool": pool,
        "correlation": report,
        "ownership": pool[["name", "position", "salary", "median",
                           "ownership", "leverage"]].copy(),
        "lineups": built,
    }


def render(res: dict) -> str:
    roster = res["roster"]
    lines = [
        f"LINEUPS  draft group {res['draft_group']}  ({res['site'].upper()} "
        f"{'showdown' if 'CPT' in roster['slots'] else 'classic'})",
        "=" * 70,
        f"{res['sims']:,} correlated simulations of the slate",
        "",
        "Correlation assumed by the factor model, and delivered by the sims:",
        res["correlation"].head(14).to_string(
            index=False, float_format=lambda v: f"{v:6.3f}"),
        "",
        "These two columns should agree to a few thousandths. They are built",
        "from factors - a game shock, a team shock, competition within a team",
        "and position - so the matrix is a real distribution by construction",
        "and nothing is shrunk to make it one. A large gap here is a bug.",
    ]
    for objective in ("cash", "gpp"):
        lines.append(O.report(res["lineups"][objective], roster))

    cash = set(res["lineups"]["cash"]["name"])
    gpp = set(res["lineups"]["gpp"]["name"])
    lines += [
        "",
        "-" * 70,
        f"The two builds share {len(cash & gpp)} of {len(cash)} players.",
        "A large overlap is not a bug - on a slate with one obvious value",
        "play both objectives will want him. An IDENTICAL pair would be.",
        "",
        "Ownership here is MODELLED, not measured - DraftKings' standings sit",
        "behind a login. The scale is pinned by arithmetic (every entry fields",
        "exactly one quarterback, so quarterback ownership sums to 100%); only",
        "the concentration is a guess. Treat leverage as indicative.",
        "",
        "Nothing here has been graded against a settled contest yet.",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draft-group", type=int,
                   help="exact slate; omit to pick the busiest of --slate")
    p.add_argument("--slate", default=None, choices=["showdown", "classic"],
                   help="find the busiest live slate of this kind")
    p.add_argument("--site", default="dk", choices=["dk", "fd"])
    p.add_argument("--classic", action="store_true",
                   help="classic roster rules instead of showdown")
    p.add_argument("--entries", type=int, default=1)
    p.add_argument("--sims", type=int, default=None)
    p.add_argument("--candidates", type=int, default=150)
    p.add_argument("--field", type=int, default=100_000,
                   help="entries in the contest, for the duplication estimate")
    p.add_argument("--out", default=str(config.LINEUPS / "latest.csv"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # On a Sunday you do not know the draft group id, and looking it up is
    # exactly the kind of manual step that gets done wrong at 12:55.
    kind = args.slate or ("classic" if args.classic else "showdown")
    if args.draft_group:
        dg, label = args.draft_group, "(given)"
    else:
        dg, label = P.pick_slate(kind)
        print(f"slate: {label}  (draft group {dg})\n")

    # The roster rules follow the slate that was actually chosen, so asking for
    # a classic slate cannot silently build a showdown lineup for it.
    showdown = kind == "showdown" and not args.classic

    res = run(dg, site=args.site, showdown=showdown,
              entries=args.entries, sims=args.sims,
              n_candidates=args.candidates, field_size=args.field)
    print(render(res))

    frames = [df for df in res["lineups"].values()]
    allof = pd.concat(frames, ignore_index=True)
    keep = [c for c in ("objective", "entry", "slot", "name", "position",
                        "team", "salary", "charged", "q50", "median", "mean",
                        "floor", "ceiling", "ownership", "leverage",
                        "p_cash", "p_gpp", "duplicates", "edge", "own_sum")
            if c in allof.columns]
    allof[keep].to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    own_path = str(config.LINEUPS / "ownership.csv")
    res["ownership"].sort_values("ownership", ascending=False).to_csv(
        own_path, index=False)
    print(f"wrote {own_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
