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
import project as P
import simulate as S

log = logging.getLogger(__name__)


def game_type(showdown: bool) -> str:
    return "Showdown Captain Mode" if showdown else "Classic"


def run(draft_group: int, site: str = "dk", showdown: bool = True,
        entries: int = 1, sims: int | None = None,
        n_candidates: int = 150) -> dict:
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

    built = {}
    for objective in ("cash", "gpp"):
        built[objective] = O.build(pool, roster, draws, objective=objective,
                                   entries=entries, n_candidates=n_candidates)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "draft_group": draft_group,
        "site": site,
        "roster": roster,
        "sims": sims,
        "pool": pool,
        "correlation": report,
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
        "Correlation the simulator actually delivered:",
        res["correlation"].to_string(
            index=False, float_format=lambda v: f"{v:6.3f}"),
        "",
        "The delivered column is lower than asked because the priors cannot",
        "all hold at once - six receivers each tied to one quarterback must",
        "correlate with each other through him, which contradicts the",
        "competition term. The repair finds the nearest possible matrix.",
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
        "Still not modelled: ownership, and therefore leverage. Nothing here",
        "knows who the field will crowd into, which is most of a tournament",
        "edge. Nothing here has been graded against a settled contest either.",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draft-group", type=int, required=True)
    p.add_argument("--site", default="dk", choices=["dk", "fd"])
    p.add_argument("--classic", action="store_true",
                   help="classic roster rules instead of showdown")
    p.add_argument("--entries", type=int, default=1)
    p.add_argument("--sims", type=int, default=None)
    p.add_argument("--candidates", type=int, default=150)
    p.add_argument("--out", default=str(config.LINEUPS / "latest.csv"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    res = run(args.draft_group, site=args.site, showdown=not args.classic,
              entries=args.entries, sims=args.sims,
              n_candidates=args.candidates)
    print(render(res))

    frames = [df for df in res["lineups"].values()]
    allof = pd.concat(frames, ignore_index=True)
    keep = [c for c in ("objective", "entry", "slot", "name", "position",
                        "team", "salary", "charged", "q50", "median", "mean",
                        "floor", "ceiling", "p_cash", "p_gpp")
            if c in allof.columns]
    allof[keep].to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
