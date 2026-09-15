"""Can we see what the field actually played?

This decides how good the tournament side can be, so it gets answered before
anything is built on top of it.

Why it matters more than projections
------------------------------------
Tournaments are not won by being right. They are won by being right where the
field is wrong. A lineup of six correctly-projected players who are each owned
by 40% of the field wins nothing, because forty thousand other people have the
same lineup and the prize is split. The quantity that decides a GPP is
leverage - projected points relative to projected ownership - and without
ownership it cannot be computed at all, only guessed at.

The path
--------
DraftKings publishes standings for completed contests, and standings contain
every entrant's lineup. Ownership is then not estimated, it is counted. A few
months of that is enough to train an ownership model - ownership is strongly
predictable from salary, projected points, value, and how well known a player
is - and a trained ownership model is what lets a realistic synthetic field be
generated for slates that have not happened yet.

If the endpoint is closed, the GPP side still works, but the field has to be
modelled from priors rather than measured, and everything built on it is
weaker. That is a real difference in what can be claimed, so it is worth
knowing now rather than after the optimiser is written.

    python ownership_probe.py
    python ownership_probe.py --contest 123456789
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from collections import Counter

import pandas as pd
import requests

import config

log = logging.getLogger(__name__)

UA = {"User-Agent": "Mozilla/5.0 (dfs research)",
      "Accept": "text/csv,application/json,*/*"}
TIMEOUT = 45

# Every form worth trying, most informative first. The CSV export is the prize:
# it carries one row per entry with the lineup spelled out, which is ownership
# by direct count rather than by inference.
FORMS = [
    ("full standings CSV",
     "https://www.draftkings.com/contest/exportfullstandingscsv?contestId={id}"),
    ("gamecenter participants",
     "https://api.draftkings.com/scores/v1/megacontests/{id}/participants?format=json"),
    ("entries by contest",
     "https://api.draftkings.com/scores/v1/entries?contestId={id}&format=json"),
    # The first three are the ones that were tried, and all three are shut:
    # the CSV returns a login page (identically 139,263 bytes for a 182,000
    # entry contest and a 52,000 entry one, which is how you know it is one
    # generic wall rather than data), and the other two 404.
    #
    # Everything below is a second sweep. These are the other public shapes
    # DraftKings has served lineups and leaderboards from; each is cheap to
    # ask and the answer is printed either way, so a closed door costs one
    # line of output and an open one changes what the tournament side can
    # claim. Listing them is not a prediction that any will work.
    ("leaderboard v1",
     "https://api.draftkings.com/scores/v1/leaderboards/{id}?format=json"),
    ("leaderboard v2",
     "https://api.draftkings.com/scores/v2/leaderboards/{id}?format=json"),
    ("contest lineups",
     "https://api.draftkings.com/lineups/v1/contests/{id}/lineups?format=json"),
    ("contest entries",
     "https://api.draftkings.com/contests/v1/contests/{id}/entries?format=json"),
    ("live contest scores",
     "https://api.draftkings.com/scores/v2/entries/livemegacontest?contestId={id}&format=json"),
    ("details popup",
     "https://www.draftkings.com/contest/detailspop?contestId={id}"),
    # Last, because it is the one that already answered. It carries no
    # ownership, but it should carry the PAYOUT CURVE - which both the cash
    # line and the tournament bar are currently estimating from a synthetic
    # field, and which would be a measured fact instead.
    ("contest detail",
     "https://api.draftkings.com/contests/v1/contests/{id}?format=json"),
]


def head(t):
    print(f"\n{t}\n{'=' * 72}")


def sub(t):
    print(f"\n{t}\n{'-' * 72}")


def archived_contests() -> pd.DataFrame:
    """Contests the capture job has already written down.

    Using the archive rather than the live lobby is deliberate: the lobby only
    lists contests that have not started, and the standings of a contest that
    has not been played do not exist. The capture job records the id and the
    start time, which is exactly what is needed to find one that has finished.
    """
    rows = []
    for folder in sorted(config.SLATES.glob("*")):
        for path in sorted(folder.glob("contests_*.csv")):
            try:
                rows.append(pd.read_csv(path))
            except Exception as exc:  # noqa: BLE001
                log.warning("unreadable %s: %s", path, exc)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True).drop_duplicates("contest_id")
    df["starts_utc"] = pd.to_datetime(df["starts_utc"], errors="coerce",
                                      utc=True)
    return df


def pick_targets(df: pd.DataFrame, limit: int = 3) -> list[dict]:
    """Contests most likely to have finished and to be worth reading.

    Large fields first. A 3,000-entry tournament gives a far better ownership
    estimate than a 20-person double-up, and if only one contest can be read
    it should be the one that says the most about the field.
    """
    if df.empty:
        return []

    # Never conclude from a column that might be empty. The start times in the
    # archive were written by a parser that could not read DraftKings' .NET
    # date format, so every one of 3,705 contests came back NaT - and this
    # function dutifully reported that none had started, which sounded like an
    # answer about the world and was actually an answer about a regex.
    #
    # So when the column is unusable, say so and probe the biggest contests
    # anyway. The endpoint is the authority on whether a contest has standings;
    # a start time is only a hint about which ones to try first, and a probe
    # that refuses to probe has failed at its one job.
    usable = df["starts_utc"].notna()
    if not usable.any():
        log.warning("no archived contest has a readable start time - probing "
                    "the largest contests regardless, since the endpoint is "
                    "what actually decides this")
        candidates = df
    else:
        now = pd.Timestamp.now(tz="UTC")
        candidates = df[usable & (df["starts_utc"] < now)]
        if candidates.empty:
            log.warning("no archived contest has started yet; falling back to "
                        "the largest regardless")
            candidates = df

    entries = pd.to_numeric(candidates.get("entries"), errors="coerce")
    candidates = candidates.assign(_n=entries.fillna(0))
    candidates = candidates.sort_values("_n", ascending=False)
    return candidates.head(limit).drop(columns="_n").to_dict("records")


def try_contest(contest_id: int) -> dict:
    """Every endpoint against one contest, reporting what each said."""
    sub(f"contest {contest_id}")
    found = {}
    for label, url in FORMS:
        u = url.format(id=contest_id)
        try:
            r = requests.get(u, headers=UA, timeout=TIMEOUT,
                             allow_redirects=True)
        except requests.RequestException as exc:
            print(f"  {label:<26} error - {str(exc)[:60]}")
            continue

        ctype = r.headers.get("Content-Type", "")[:40]
        print(f"  {label:<26} HTTP {r.status_code}  {len(r.content):>9,} bytes"
              f"  {ctype}")

        if r.status_code != 200 or not r.content:
            continue
        # A login wall returns 200 and an HTML page, which is the failure most
        # likely to be mistaken for success.
        body = r.content[:400].decode("utf-8", "replace").lower()
        if "<html" in body and "csv" not in ctype:
            print("      -> HTML, not data. Almost certainly a login wall.")
            continue
        found[label] = r.content
    return found


LINEUP_HINTS = ("lineup", "roster", "draftable", "entries", "entrant",
                "userName", "username", "playerId")
PAYOUT_HINTS = ("payout", "prize", "placepaid", "tier", "minposition",
                "maxposition")


def _walk(obj, path="", depth=0):
    """Every nested key, flattened, so a useful field cannot hide two levels
    down inside a response nobody read past the top of."""
    if depth > 4:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = f"{path}.{k}" if path else str(k)
            yield here, v
            yield from _walk(v, here, depth + 1)
    elif isinstance(obj, list) and obj:
        yield from _walk(obj[0], f"{path}[]", depth + 1)


def describe_json(payload) -> None:
    """Say whether this response carries lineups or a payout curve.

    Printing the top-level keys was not enough. `contest detail` answered on
    every contest and the probe reported `['contestDetail', 'errorStatus']` and
    stopped - which says nothing about whether the payout structure everything
    downstream is currently GUESSING at was sitting one level below. Both the
    cash line and the tournament bar are estimated from a synthetic field of
    randomly assembled lineups; a real payout curve replaces that estimate with
    a fact, and it costs nothing to look.
    """
    found_lineup, found_payout = [], []
    for path, value in _walk(payload):
        leaf = path.split(".")[-1].split("[")[0].lower()
        if any(h.lower() in leaf for h in LINEUP_HINTS):
            found_lineup.append((path, value))
        if any(h.lower() in leaf for h in PAYOUT_HINTS):
            found_payout.append((path, value))

    if found_payout:
        print(f"      -> PAYOUT fields ({len(found_payout)}):")
        for path, value in found_payout[:8]:
            v = str(value)
            print(f"           {path} = {v[:90]}")
    if found_lineup:
        print(f"      -> LINEUP-ish fields ({len(found_lineup)}):")
        for path, value in found_lineup[:8]:
            v = str(value)
            print(f"           {path} = {v[:90]}")
    if not found_payout and not found_lineup:
        sample = [p for p, _ in _walk(payload)][:14]
        print(f"      -> nothing lineup or payout shaped. paths seen: {sample}")


def read_standings(raw: bytes) -> pd.DataFrame | None:
    """The standings CSV, if that is what came back."""
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        print(f"      -> not parseable as CSV: {str(exc)[:70]}")
        return None
    print(f"      -> {len(df):,} rows, columns: {list(df.columns)[:8]}")
    return df


def ownership_from(df: pd.DataFrame) -> pd.DataFrame:
    """Count how often each player appears. That is ownership, exactly.

    DraftKings writes a lineup as one string per entry, with the roster slot
    labels inline. Splitting on those labels is what turns a column of text
    into the single most valuable number in tournament play.
    """
    col = next((c for c in df.columns
                if str(c).strip().lower() in ("lineup", "entry lineup")), None)
    if col is None:
        print("      -> no lineup column; ownership cannot be counted")
        return pd.DataFrame()

    slots = ("QB", "RB", "WR", "TE", "FLEX", "DST", "CPT", "K", "D")
    counts: Counter = Counter()
    entries = 0
    for line in df[col].dropna().astype(str):
        entries += 1
        text = line
        for s in slots:
            text = text.replace(f" {s} ", "|")
        for name in text.split("|"):
            name = name.strip()
            if name:
                counts[name] += 1
    if not entries or not counts:
        print("      -> lineup column present but nothing parsed out of it")
        return pd.DataFrame()

    out = (pd.DataFrame({"player": list(counts), "entries": list(counts.values())})
           .assign(ownership=lambda d: 100.0 * d["entries"] / entries)
           .sort_values("ownership", ascending=False)
           .reset_index(drop=True))
    print(f"      -> ownership counted across {entries:,} entries, "
          f"{len(out)} distinct players")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contest", type=int, action="append",
                   help="probe a specific contest id (repeatable)")
    p.add_argument("--limit", type=int, default=3)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    head("CAN THE FIELD BE SEEN")
    print("  Tournaments are won on leverage, not on accuracy. Leverage needs")
    print("  ownership. Ownership needs the standings of contests that have")
    print("  already been played. This asks whether those are readable.")

    if args.contest:
        targets = [{"contest_id": c, "name": "(given)", "entries": None}
                   for c in args.contest]
    else:
        archive = archived_contests()
        print(f"\n  contests in the local archive: {len(archive):,}")
        readable = int(archive["starts_utc"].notna().sum()) if len(archive) else 0
        print(f"  of those, with a readable start time: {readable:,}")
        if len(archive) and not readable:
            print("  (start times are unreadable in this archive - probing the")
            print("   largest contests anyway, since the endpoint decides this,")
            print("   not the timestamp)")
        targets = pick_targets(archive, args.limit)
        if not targets:
            print("\n  The archive is empty. Run the capture job first, or pass")
            print("  --contest with an id from a contest you know is over.")
            return 0

    any_csv = False
    for t in targets:
        entries = t.get("entries")
        print(f"\n  {t.get('name', '')[:60]}"
              f"{f'  ({int(entries):,} entries)' if entries else ''}")
        found = try_contest(int(t["contest_id"]))
        for label, raw in found.items():
            if "CSV" in label:
                df = read_standings(raw)
                if df is not None and not df.empty:
                    own = ownership_from(df)
                    if not own.empty:
                        any_csv = True
                        sub("The ten most-owned players in that contest")
                        for r in own.head(10).itertuples(index=False):
                            print(f"  {r.player[:32]:<32} {r.ownership:5.1f}%")
            else:
                try:
                    payload = json.loads(raw)
                    keys = (sorted(payload)[:10] if isinstance(payload, dict)
                            else f"list of {len(payload)}")
                    print(f"      -> JSON, keys: {keys}")
                    describe_json(payload)
                except json.JSONDecodeError:
                    print("      -> not JSON either")

    head("WHAT THIS MEANS")
    if any_csv:
        print("  Ownership is readable. That is the good outcome, and it is")
        print("  the difference between a tournament model that computes")
        print("  leverage and one that guesses at it.")
        print()
        print("  Next: archive standings for every contest as it finishes,")
        print("  build an ownership history, and train a model to project it")
        print("  forward - ownership is strongly predictable from salary,")
        print("  projected points and value, so a few months is enough.")
    else:
        print("  No contest returned a readable lineup list. The endpoints may")
        print("  now require a signed-in session.")
        print()
        print("  This does not stop the tournament model, but it changes what")
        print("  can be claimed for it: the field would have to be generated")
        print("  from priors - chalk is cheap and popular, value plays are")
        print("  heavily owned - rather than from measurement, and the")
        print("  leverage numbers would be indicative rather than real.")
        print()
        print("  The cash game side is entirely unaffected. It does not need")
        print("  the field at all, only the score that beats half of it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
