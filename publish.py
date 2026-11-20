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
    out = P.run(draft_group, site=site,
                captain_multiplier=roster.get("captain_multiplier"))
    pool = out["players"].reset_index(drop=True)
    qcols = [f"q{int(q * 100)}" for q in config.QUANTILES]

    # Bovada player props, blended over the model (see nfl_props.py).
    try:
        import nfl_props
        n_props = nfl_props.apply(pool, qcols)
    except Exception as exc:                                   # noqa: BLE001
        log.warning("props unavailable (%s)", str(exc)[:100])
        n_props = 0
        pool["model"] = np.nan
        pool["props"] = np.nan

    # Out of the rotation (project.mark_rotation): no props, and his team has
    # played STALE_GAMES+ games this season without him.
    pool["news"] = ""
    if "team_games_missed" in pool.columns:
        import project as PR
        stale = (pd.to_numeric(pool["team_games_missed"], errors="coerce") >= PR.STALE_GAMES) \
            & pool["props"].isna()
        for i in pool.index[stale]:
            pool.at[i, "playing"] = "out"
            if "p_play" in pool.columns:
                pool.at[i, "p_play"] = 0.0
            pool.at[i, "news"] = (f"no game in his team's last "
                                  f"{int(pool.at[i, 'team_games_missed'])} - not projected")
        if stale.any():
            log.info("not in the rotation, not projected: %s",
                     ", ".join(pool.loc[stale, "name"].astype(str).head(15)))
        pool = pool[~stale].reset_index(drop=True)

    own = OWN.project(pool, roster)
    chalk = []
    try:
        own, chalk = OWN.chalk_adjust(pool, own, roster)
        if chalk:
            log.info("chalk: %s", ", ".join(chalk))
    except Exception as exc:                                   # noqa: BLE001
        log.warning("chalk check failed (%s)", str(exc)[:80])
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
            # The two-part model's other half, published so the page can show
            # it. A projection of 18 points at a 55% chance of playing is a
            # completely different bet from 18 points at 99%, and a page that
            # shows only the first is hiding the more important number.
            "pplay": (round(float(r.p_play), 3)
                      if getattr(r, "p_play", None) == getattr(
                          r, "p_play", None) else None),
            "mean": round(float(getattr(r, "mean", r.median)), ROUND),
            "inj": str(getattr(r, "injury_status", "") or ""),
            "model": (round(float(r.model), 2) if getattr(r, "model", None) == getattr(r, "model", None)
                      and getattr(r, "model", None) is not None else None),
            "props": (round(float(r.props), 2) if getattr(r, "props", None) == getattr(r, "props", None)
                      and getattr(r, "props", None) is not None else None),
            "sgp": (int(r.season_games) if getattr(r, "season_games", None) == getattr(r, "season_games", None)
                   and getattr(r, "season_games", None) is not None else None),
            "chalk": str(r.name) in chalk,
            "news": str(getattr(r, "news", "") or "") or None,
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
        "props": {"source": "Bovada player props", "players": int(n_props)} if n_props else None,
        "notes": {
            "ownership": "modelled from priors, not measured - DraftKings' "
                         "standings are behind a login",
            "graded": False,
        },
    }


def roster_name(roster: dict) -> str:
    return "showdown" if "CPT" in roster["slots"] else "classic"


def run(kinds: list[str], site: str, sims: int, field_size: int,
        max_slates: int = 12, min_contests: int = 1,
        soon: int | None = None) -> list[dict]:
    """Build EVERY live slate we have roster rules for, not one per kind.

    The first version took the busiest Classic and the busiest Showdown and
    stopped. That is why the page's dropdown only ever had two entries while
    DraftKings was selling twenty-eight: the manifest is the dropdown, and the
    manifest had two rows in it. The page was never broken - it was faithfully
    showing everything it had been given.

    Every draft group DraftKings lists is now built, subject to two limits
    that exist for real reasons rather than tidiness:

    * **Roster rules must exist.** Best Ball, Snake, Tiers and Madden are all
      in the same lobby and none of them is a salary-cap lineup. Building one
      with Classic rules would produce a confident, illegal entry. A slate
      whose game type has no rules is recorded as skipped, with its name, so
      the gap is visible rather than silent.
    * **Time.** Each slate is a full fit, fifty thousand correlated draws and
      two optimisations, so the count is capped and the busiest slates are
      built first. The cap is a parameter, not a belief.
    """
    import project as PR
    try:
        table = data.slates()
    except Exception as exc:                                   # noqa: BLE001
        log.error("could not list slates: %s", exc)
        return []

    wanted = {k.lower() for k in kinds} if kinds else set()
    if soon:
        # Lock-time run: only slates starting within `soon` minutes, so it
        # finishes in minutes rather than rebuilding all twelve.
        now = pd.Timestamp.now(tz="UTC")
        st = pd.to_datetime(table["starts"], utc=True, errors="coerce")
        table = table[(st > now) & (st <= now + pd.Timedelta(minutes=soon))]
        log.info("lock-time run: %d slate(s) start within %d minutes", len(table), soon)
    written, skipped = [], []
    built = 0
    for row in table.itertuples(index=False):
        game_type = str(row.game_type)
        key = (site, game_type)
        if key not in config.ROSTERS:
            skipped.append((game_type, int(row.contests)))
            continue
        kind = "showdown" if "showdown" in game_type.lower() else "classic"
        if wanted and kind not in wanted:
            continue
        if int(row.contests) < min_contests:
            continue
        if built >= max_slates:
            skipped.append((f"{game_type} (over the cap of {max_slates})",
                            int(row.contests)))
            continue

        dg = int(row.draft_group)
        label = str(getattr(row, "example", "") or game_type)
        log.info("building %s: %s (draft group %s, %d contests)",
                 kind, label, dg, int(row.contests))
        try:
            payload = slate_payload(dg, site, kind == "showdown", sims,
                                    field_size)
        except Exception as exc:                               # noqa: BLE001
            # One dead slate must not take the whole publish down. The page
            # shows what it has, and the manifest records what failed.
            log.error("%s slate %s failed: %s: %s", kind, dg,
                      type(exc).__name__, exc)
            written.append({"kind": kind, "draft_group": dg, "label": label,
                            "error": f"{type(exc).__name__}: {exc}"})
            continue
        payload["label"] = label
        payload["contests"] = int(row.contests)
        payload["starts"] = str(getattr(row, "starts", "") or "")

        folder = OUT / payload["sport"]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{site}-{kind}-{dg}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")))
        # The last version written before a slate starts is its lock board;
        # the weekly grade reads it from here.
        try:
            day = str(pd.Timestamp(payload["starts"]).tz_convert("America/New_York").date())
            arch = OUT / "archive" / payload["sport"] / day
            arch.mkdir(parents=True, exist_ok=True)
            (arch / path.name).write_text(path.read_text())
        except Exception as exc:                               # noqa: BLE001
            log.warning("archive skipped (%s)", str(exc)[:60])
        built += 1
        log.info("wrote %s (%.0f KB, %d players)", path,
                 path.stat().st_size / 1024, len(payload["players"]))
        written.append({
            "kind": kind, "sport": payload["sport"], "site": site,
            "draft_group": dg, "label": label,
            "contests": int(row.contests),
            "starts": payload["starts"],
            "file": f"data/{payload['sport']}/{path.name}",
            "players": len(payload["players"]),
            "generated_at": payload["generated_at"],
        })

    if skipped:
        log.info("skipped %d slates with no roster rules: %s", len(skipped),
                 ", ".join(f"{g} ({n})" for g, n in skipped[:8]))
    log.info("built %d slates", built)
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sport", default="nfl")
    p.add_argument("--site", default="dk", choices=["dk", "fd"])
    p.add_argument("--kinds", nargs="*", default=[],
                   help="blank means every kind we have roster rules for")
    p.add_argument("--sims", type=int, default=config.SIMS_GPP)
    p.add_argument("--field", type=int, default=100_000)
    p.add_argument("--soon", type=int, default=None,
                   help="only slates starting within N minutes (lock runs)")
    p.add_argument("--max-slates", type=int, default=12,
                   help="how many slates to build; each is a full fit plus "
                        "50k correlated draws, so this is a time budget")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    OUT.mkdir(parents=True, exist_ok=True)
    written = run(args.kinds, args.site, args.sims, args.field,
                  max_slates=args.max_slates, soon=args.soon)
    if args.soon and not written:
        print("no slate starts within the window; nothing to do")
        return 0

    manifest = OUT / "manifest.json"
    prior = {}
    if manifest.exists():
        try:
            prior = json.loads(manifest.read_text())
        except json.JSONDecodeError:
            prior = {}

    # Keep yesterday's entries for sports not built in this run, so a football
    # publish does not silently delete basketball from the page.
    # Started slates leave the dropdown (the manifest had grown to 207 rows,
    # back to September): kept for 12 hours after their start, then dropped.
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=12)

    def live(sl):
        try:
            return pd.Timestamp(sl.get("starts")).tz_convert("UTC") > cutoff
        except Exception:                                      # noqa: BLE001
            return False
    slates = {s["file"]: s for s in prior.get("slates", [])
              if "file" in s and live(s)}
    for s in written:
        if "file" in s:
            slates[s["file"]] = s
    keep = {(config.DOCS / f).resolve() for f in slates}
    for f in (OUT / "nfl").glob("*.json"):
        if f.resolve() not in keep:
            f.unlink()

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
