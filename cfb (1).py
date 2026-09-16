"""College football: the board, the history, and the map between them.

The explorer answered every question that had to be answered before writing
this file, and the answers set its shape:

* **The board comes from the lobby, not the API.** `api.draftkings.com`
  answers 403 Access Denied to GitHub's runners on every URL shape tried.
  `www.draftkings.com/lineup/getavailableplayers` answers 200 with the whole
  slate. It speaks an abbreviated dialect - `fn`/`ln`, `s`, `pn` - and carries
  no team abbreviation for the player, only for the two sides of his fixture.

* **The names join at 99.6%.** 847 of 850, and all three misses sat at the
  $3,000 minimum against a matched median of $3,300. That is the shape a join
  is allowed to fail in: the bottom of the price range, where a missing
  projection costs nothing because nobody was rostering them anyway.

* **The team codes do NOT join.** 2 of 24. DraftKings writes BAMA, TA&M, RU,
  SCAR; CFBD writes Alabama, Texas A&M, Rutgers, South Carolina. This is the
  dangerous half, because a wrong team code does not drop a player - it hands
  him the wrong opponent, the wrong implied total and the wrong correlation
  group, and produces a full board of confident numbers that are wrong in
  every row.

The team map, and why it is not written by hand
-----------------------------------------------
A hand-written map is 24 guesses that look like knowledge, and at least one of
them is wrong in a way nobody can see. `UL` is the example that matters: it is
Louisiana to DraftKings and could plausibly be read as Louisville, and the two
are different teams playing different opponents. Nothing about the string
settles it.

The fixture list does. DraftKings publishes each player's game as `AWAY @
HOME`, and CFBD publishes the same week's schedule with full school names. A
DraftKings fixture has to match exactly one CFBD fixture, and matching BOTH
sides at once is a far stronger constraint than matching either alone - `UL`
is resolved not by what it looks like but by who it is playing. So the map is
SOLVED from the schedule and then printed for inspection, and anything that
cannot be solved is reported rather than guessed.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
import requests

from data import normalise_name

log = logging.getLogger(__name__)

UA = {"User-Agent": "Mozilla/5.0 (dfs research)"}
TIMEOUT = 40
CFBD = "https://api.collegefootballdata.com"
DK_CONTESTS = "https://www.draftkings.com/lobby/getcontests?sport=CFB"
DK_PLAYERS = ("https://www.draftkings.com/lineup/getavailableplayers"
              "?draftGroupId={dg}")

# DraftKings college football, Classic. Same shape as the NFL game with one
# difference that matters for pricing: college awards a full point per
# reception, so slot receivers on pass-happy teams are worth more here than
# their yardage suggests.
SCORING = {
    "pass_yards": 0.04, "pass_td": 4.0, "interception": -1.0,
    "rush_yards": 0.1, "rush_td": 6.0,
    "rec": 1.0, "rec_yards": 0.1, "rec_td": 6.0,
    "return_td": 6.0, "fumble_lost": -1.0,
}
# The bonuses are why a projection has to be a distribution rather than a
# mean. A 95-yard rusher and a 105-yard rusher are one yard of talent apart
# and three points apart, and only a simulation prices that step.
BONUS = {"pass_yards": (300, 3.0), "rush_yards": (100, 3.0),
         "rec_yards": (100, 3.0)}


class Unavailable(RuntimeError):
    """A source this build cannot proceed without did not answer."""


# --------------------------------------------------------------------- draftkings
def _get(url: str, headers=None):
    r = requests.get(url, headers=headers or UA, timeout=TIMEOUT)
    if r.status_code != 200:
        raise Unavailable(f"{url[:70]} -> HTTP {r.status_code} "
                          f"({r.content[:80]!r})")
    return r.json()


def slates() -> pd.DataFrame:
    """Every college football draft group DraftKings is currently selling."""
    payload = _get(DK_CONTESTS)
    contests = payload.get("Contests") or []
    if not contests:
        raise Unavailable("the college football lobby listed no contests")
    rows = {}
    for c in contests:
        dg = c.get("dg")
        if not dg:
            continue
        r = rows.setdefault(dg, {"draft_group": dg, "contests": 0,
                                 "game_type": c.get("gameType"),
                                 "starts_text": c.get("sdstring"),
                                 "biggest_prize": 0, "example": c.get("n")})
        r["contests"] += 1
        r["biggest_prize"] = max(r["biggest_prize"], c.get("po") or 0)
    out = pd.DataFrame(rows.values())
    return out.sort_values("contests", ascending=False).reset_index(drop=True)


def board(draft_group: int) -> pd.DataFrame:
    """Who is priced on a slate, from the endpoint that is not blocked.

    Field mapping is from a row printed verbatim by the explorer, not from
    memory. The one that bit already: `pn` is the position ("QB"), while `pp`
    is an integer that is zero for every player on the board. Mapping position
    to `pp` produced 850 players all sharing a single position called `0` -
    which would have made every positional constraint in the optimiser
    vacuous, and a lineup of eight quarterbacks perfectly legal.
    """
    payload = _get(DK_PLAYERS.format(dg=draft_group))
    raw = (payload.get("playerList") or payload.get("draftables")
           or payload.get("players") or [])
    if not raw:
        raise Unavailable(f"draft group {draft_group} returned no players "
                          f"(top-level keys: {sorted(payload)[:10]})")

    rows = []
    for p in raw:
        home, away = p.get("htabbr"), p.get("atabbr")
        tid, htid, atid = p.get("tid"), p.get("htid"), p.get("atid")
        team = home if tid == htid else away if tid == atid else None
        opponent = away if tid == htid else home if tid == atid else None
        rows.append({
            "dk_player_id": p.get("pid"),
            "name": " ".join(x for x in (p.get("fn"), p.get("ln")) if x),
            "position": p.get("pn"),
            "team": team,
            "opponent": opponent,
            "is_home": 1 if tid == htid else 0 if tid == atid else np.nan,
            "salary": pd.to_numeric(p.get("s"), errors="coerce"),
            "dk_points_per_game": pd.to_numeric(p.get("ppg"), errors="coerce"),
            "disabled": bool(p.get("IsDisabledFromDrafting")),
            "roster_slot": p.get("rosposid"),
            "game": f"{away} @ {home}" if home and away else None,
        })
    df = pd.DataFrame(rows)
    df["norm"] = df["name"].map(normalise_name)

    # Say out loud what the mapping produced. A board where every player
    # shares one position, or where every salary is NaN, is a mapping failure
    # that looks exactly like a successful parse from the outside.
    if df["position"].nunique() <= 1:
        raise Unavailable(
            f"every player on draft group {draft_group} has position "
            f"{df['position'].iloc[0]!r}. The position field moved; look at a "
            f"raw row before trusting anything downstream.")
    if df["team"].isna().any():
        n = int(df["team"].isna().sum())
        log.warning("%d of %d players could not be assigned to a side of "
                    "their fixture - they are dropped rather than guessed",
                    n, len(df))
        df = df.dropna(subset=["team"])

    log.info("draft group %s: %d players, %d teams, %d games, $%s-$%s",
             draft_group, len(df), df["team"].nunique(), df["game"].nunique(),
             int(df["salary"].min()), int(df["salary"].max()))
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- cfbd
def cfbd(path: str, key: str, **params):
    hdr = dict(UA)
    hdr["Authorization"] = f"Bearer {key}"
    r = requests.get(f"{CFBD}/{path}", headers=hdr, params=params,
                     timeout=TIMEOUT)
    if r.status_code != 200:
        raise Unavailable(f"CFBD {path} -> HTTP {r.status_code} "
                          f"{r.text[:100]}")
    return r.json()


def live_week(key: str, season: int) -> int:
    """The week holding the next kickoff.

    Not "the earliest week with an unplayed game" - that picked week 2 while
    DraftKings was selling week 3, because two postponed games sat unplayed in
    a week that was otherwise finished. Choosing by the clock cannot make that
    mistake.
    """
    now = pd.Timestamp.now("UTC")
    best, best_week = None, None
    for week in range(1, 17):
        games = cfbd("games", key, year=season, week=week, seasonType="regular")
        if not games:
            continue
        starts = pd.to_datetime(
            pd.Series([g.get("startDate") for g in games]), utc=True,
            errors="coerce").dropna()
        future = starts[starts > now]
        if len(future) and (best is None or future.min() < best):
            best, best_week = future.min(), week
        if len(future) == len(games) and best_week is not None:
            break
    if best_week is None:
        raise Unavailable(f"no unplayed {season} game found in weeks 1-16")
    log.info("week %d is live: next kickoff %s", best_week, best)
    return best_week


# ---------------------------------------------------------------- the team map
_STOP = {"state", "st", "university", "of", "the", "at", "a&m", "am"}


def _tokens(school: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9&]+", str(school).lower()) if t]


def _candidates(school: str, abbreviation, alternates) -> set[str]:
    """Every string this school might plausibly be called by DraftKings."""
    out = set()
    for a in ([abbreviation] + list(alternates or [])):
        if a:
            out.add(re.sub(r"[^A-Z0-9&]", "", str(a).upper()))
    toks = _tokens(school)
    flat = "".join(toks)
    out.add(flat.upper())
    # Prefixes: BAMA is not a prefix of Alabama, but CLEM is of Clemson and
    # STAN is of Stanford, and prefixes cover most of DraftKings' shortenings.
    for n in (2, 3, 4, 5, 6):
        if len(flat) >= n:
            out.add(flat[:n].upper())
    # Initials: Ohio State -> OS, Mississippi State -> MS, plus the common
    # habit of appending the division letter (OSU, MSST).
    initials = "".join(t[0] for t in toks if t not in _STOP)
    if initials:
        out.add(initials.upper())
        out.add((initials + "U").upper())
    meaningful = [t for t in toks if t not in _STOP]
    if meaningful:
        out.add((meaningful[0][:4] + "ST").upper())
        out.add((meaningful[0][:3] + "ST").upper())
        out.add(("U" + meaningful[0][0]).upper())
    return {c for c in out if c}


def _similarity(code: str, school: str, abbreviation, alternates) -> float:
    """How much a DraftKings code looks like a school. Deliberately blunt.

    This is only ever a PRIOR. The fixture constraint below is what actually
    decides, because a score that looks confident on one side of a game can
    still be the wrong team - and the opponent is the evidence that settles
    it. Returning a soft score rather than a hard match is what lets the
    pairing overrule a tempting single-sided guess.
    """
    code = re.sub(r"[^A-Z0-9&]", "", str(code).upper())
    if not code:
        return 0.0
    cands = _candidates(school, abbreviation, alternates)
    if code in cands:
        return 1.0
    flat = "".join(_tokens(school)).upper()
    if flat.startswith(code):
        return 0.75
    # Every letter of the code appearing in order in the school name is weak
    # evidence - it is what makes BAMA reachable from alABAMA - so it scores
    # low enough that any real match beats it.
    i = 0
    for ch in flat:
        if i < len(code) and ch == code[i]:
            i += 1
    if i == len(code):
        return 0.45
    return 0.0


def team_map(board_df: pd.DataFrame, games: list[dict],
             teams: list[dict]) -> tuple[dict, list]:
    """Solve DraftKings code -> CFBD school from the fixture list.

    Each DraftKings fixture must be exactly one CFBD fixture, and both sides
    have to agree. That pairing is what disambiguates codes no string
    comparison can settle - UL is Louisiana rather than Louisville because
    Louisiana is the team playing this opponent this week, and Louisville is
    not.

    Returns the map and a list of (dk_code, school, score, opponent_evidence)
    rows so the caller can print the whole thing and look at it. A map nobody
    looks at is 24 silent assumptions.
    """
    meta = {t["school"]: (t.get("abbreviation"), t.get("alternateNames") or [])
            for t in teams}
    schools = list(meta)

    fixtures = (board_df.dropna(subset=["team", "opponent"])
                [["team", "opponent", "is_home"]]
                .drop_duplicates())
    dk_games = {(r.opponent, r.team) if r.is_home == 1 else (r.team, r.opponent)
                for r in fixtures.itertuples(index=False)}

    cfbd_games = [(g.get("awayTeam"), g.get("homeTeam")) for g in games
                  if g.get("awayTeam") and g.get("homeTeam")]

    resolved, report = {}, []
    for away_code, home_code in sorted(dk_games):
        scored = []
        for away_school, home_school in cfbd_games:
            a = _similarity(away_code, away_school, *meta.get(away_school,
                                                             (None, [])))
            h = _similarity(home_code, home_school, *meta.get(home_school,
                                                              (None, [])))
            # Both sides must be plausible. A fixture that matches one team
            # perfectly and the other not at all is not a match; it is a
            # coincidence, and accepting it is how a whole team ends up
            # pointed at the wrong game.
            if a > 0 and h > 0:
                scored.append((a + h, away_school, home_school))
        scored.sort(reverse=True)
        if not scored:
            report.append((f"{away_code} @ {home_code}", None, 0.0,
                           "no fixture matched both sides"))
            continue
        top = scored[0]
        margin = top[0] - (scored[1][0] if len(scored) > 1 else 0.0)
        resolved[away_code] = top[1]
        resolved[home_code] = top[2]
        report.append((f"{away_code} @ {home_code}",
                       f"{top[1]} @ {top[2]}", round(top[0], 2),
                       f"margin {margin:.2f} over "
                       f"{scored[1][1] + ' @ ' + scored[1][2] if len(scored) > 1 else 'nothing'}"))

    missing = sorted({c for g in dk_games for c in g} - set(resolved))
    if missing:
        log.error("%d DraftKings team codes could not be resolved: %s. "
                  "Their players are dropped rather than pointed at a guessed "
                  "school.", len(missing), missing)
    return resolved, report


# ------------------------------------------------------------------- the history
# CFBD names its stats the way a box score does, not the way a scoring rule
# does. This is the translation, and it is written out rather than inferred so
# that a stat CFBD renames shows up as a missing column instead of a silent
# zero.
STAT_MAP = {
    ("passing", "YDS"): "pass_yards", ("passing", "TD"): "pass_td",
    ("passing", "INT"): "interception", ("passing", "C/ATT"): "completions",
    ("rushing", "YDS"): "rush_yards", ("rushing", "TD"): "rush_td",
    ("rushing", "CAR"): "carries",
    ("receiving", "YDS"): "rec_yards", ("receiving", "TD"): "rec_td",
    ("receiving", "REC"): "rec",
    ("fumbles", "LOST"): "fumble_lost",
    ("kickReturns", "TD"): "kick_return_td",
    ("puntReturns", "TD"): "punt_return_td",
}


def _number(value):
    """A stat as a number. CFBD sends every value as a string.

    `C/ATT` arrives as "18/25", which float() rejects; taking the part before
    the slash gives completions. A bare float() here would have turned every
    quarterback's completion count into NaN, and NaN is not zero - it would
    have propagated into the feature and quietly deleted the row.
    """
    if value is None:
        return np.nan
    s = str(value).strip().replace(",", "")
    if "/" in s:
        s = s.split("/")[0]
    try:
        return float(s)
    except ValueError:
        return np.nan


def flatten_player_games(payload: list[dict]) -> pd.DataFrame:
    """One row per player per game, columns named for the scoring rule."""
    rows = []
    for game in payload:
        for team in game.get("teams") or []:
            school = team.get("team") or team.get("school")
            for cat in team.get("categories") or []:
                cname = cat.get("name")
                for typ in cat.get("types") or []:
                    tname = typ.get("name")
                    field = STAT_MAP.get((cname, tname))
                    if field is None:
                        continue
                    for ath in typ.get("athletes") or []:
                        rows.append({
                            "game_id": game.get("id"), "school": school,
                            "athlete_id": ath.get("id"),
                            "name": ath.get("name"),
                            "field": field, "value": _number(ath.get("stat")),
                        })
    if not rows:
        return pd.DataFrame()
    long = pd.DataFrame(rows)
    wide = (long.pivot_table(index=["game_id", "school", "athlete_id", "name"],
                             columns="field", values="value",
                             aggfunc="sum")
            .reset_index())
    wide.columns.name = None
    for field in set(STAT_MAP.values()):
        if field not in wide:
            wide[field] = 0.0
    return wide.fillna({f: 0.0 for f in set(STAT_MAP.values())})


def fantasy_points(df: pd.DataFrame) -> pd.Series:
    """DraftKings college football points for each row."""
    pts = pd.Series(0.0, index=df.index)
    for field, weight in SCORING.items():
        if field == "return_td":
            pts = pts + weight * (df.get("kick_return_td", 0)
                                  + df.get("punt_return_td", 0))
        elif field in df:
            pts = pts + weight * df[field]
    for field, (threshold, bonus) in BONUS.items():
        if field in df:
            pts = pts + bonus * (df[field] >= threshold)
    return pts.round(2)


def history(key: str, season: int, through_week: int) -> pd.DataFrame:
    """Every settled game this season, scored.

    Only weeks strictly BEFORE the live one. Including the live week would put
    a player's own result into the features used to predict it, which is the
    leak that makes a model look brilliant in backtest and lose money on
    Saturday.
    """
    frames = []
    for week in range(1, max(1, through_week)):
        payload = cfbd("games/players", key, year=season, week=week,
                       seasonType="regular")
        wide = flatten_player_games(payload)
        if wide.empty:
            log.warning("week %d returned no player stats", week)
            continue
        wide["season"], wide["week"] = season, week
        frames.append(wide)
    if not frames:
        raise Unavailable(f"no {season} player stats before week {through_week}")
    out = pd.concat(frames, ignore_index=True)
    out["dk_points"] = fantasy_points(out)
    out["norm"] = out["name"].map(normalise_name)
    log.info("history: %d player-games, weeks 1-%d, %d athletes, "
             "points %.1f to %.1f", len(out), through_week - 1,
             out["athlete_id"].nunique(), out["dk_points"].min(),
             out["dk_points"].max())
    return out


def market(key: str, season: int, week: int,
           provider: str = "DraftKings") -> pd.DataFrame:
    """Implied team totals for the live week, from ONE named book.

    One book by name, never an average across providers. The explorer found
    the same book listed twice under two spellings - `DraftKings` and `Draft
    Kings`, 120 games each - so averaging "all providers" would have weighted
    it twice against Bovada's once. That is not the market's view, it is a
    blend nobody chose and nobody could describe.
    """
    payload = cfbd("lines", key, year=season, week=week, seasonType="regular")
    want = re.sub(r"[^a-z]", "", provider.lower())
    rows, skipped = [], 0
    for g in payload:
        chosen = None
        for ln in g.get("lines") or []:
            name = re.sub(r"[^a-z]", "", str(ln.get("provider", "")).lower())
            if name == want and ln.get("spread") is not None \
                    and ln.get("overUnder") is not None:
                chosen = ln
                break
        if chosen is None:
            skipped += 1
            continue
        total = float(chosen["overUnder"])
        # CFBD's spread is from the HOME team's perspective and negative when
        # the home side is favoured, which is the opposite sign convention to
        # a margin. Getting this backwards hands the favourite the underdog's
        # total in every game - plausible numbers, wrong in every row.
        spread = float(chosen["spread"])
        home_implied = total / 2.0 - spread / 2.0
        away_implied = total - home_implied
        for school, implied, opp, home in (
                (g.get("homeTeam"), home_implied, g.get("awayTeam"), 1),
                (g.get("awayTeam"), away_implied, g.get("homeTeam"), 0)):
            rows.append({"school": school, "opponent_school": opp,
                         "implied_total": round(implied, 2),
                         "opponent_implied": round(total - implied, 2),
                         "game_total": total,
                         "team_spread": spread if home else -spread,
                         "is_home": home})
    if skipped:
        log.warning("%d of %d games carry no %s line - those teams get NO "
                    "implied total rather than a league average",
                    skipped, len(payload), provider)
    out = pd.DataFrame(rows)
    if len(out):
        log.info("market: %d teams, implied totals %.1f to %.1f", len(out),
                 out["implied_total"].min(), out["implied_total"].max())
    return out


# ------------------------------------------------------------------- the join
def name_keys(name) -> set[str]:
    """Every spelling two sources might agree on, not just one.

    The single-key normaliser was built for the NFL and gets one thing exactly
    right: a period becomes a SPACE, so DraftKings' "St.Brown" and nflverse's
    "St. Brown" both reduce to "st brown". Amon-Ra St. Brown depends on it.

    College football breaks it in the other direction. Initials are everywhere
    - AJ, TJ, CJ, DJ, KJ - and the two sources disagree about the periods:
    DraftKings writes "AJ Swann", CFBD writes "A.J. Swann". Period-to-space
    turns those into "aj swann" and "a j swann", which do not match. That is
    not a bench problem: AJ Swann was priced at $8,500, the second-most
    expensive quarterback on the board, and he silently had no history at all.

    Both rules are right and they contradict each other, so the answer is to
    carry BOTH spellings and match on either. Deliberately still not fuzzy -
    every key here is an exact reduction, because an edit-distance match would
    pair two different players with similar names, which is worse than a miss.
    """
    raw = str(name).strip()
    if "," in raw:
        last, _, first = raw.partition(",")
        raw = f"{first.strip()} {last.strip()}"
    keys = set()
    for dot in (" ", ""):
        s = raw.lower().replace("-", " ").replace(".", dot)
        s = s.replace("'", "").replace("`", "")
        s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
        s = " ".join(s.split())
        if s:
            keys.add(s)
    return keys


def attach_history(board: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Give every priced player his own scored games, or nothing.

    Matched within school, never across it. Two players sharing a name is
    routine across 136 schools and rare inside one, so scoping the match to
    the school is what keeps the extra name spellings from buying false
    positives with the recall they add.
    """
    agg = (hist.groupby(["school", "name"])
           .agg(games=("dk_points", "size"), mean=("dk_points", "mean"),
                best=("dk_points", "max"), floor=("dk_points", "min"))
           .reset_index())

    lookup: dict[tuple[str, str], list] = {}
    collisions = 0
    for r in agg.itertuples(index=False):
        for key in name_keys(r.name):
            slot = (r.school, key)
            if slot in lookup:
                collisions += 1
                continue
            lookup[slot] = (r.games, r.mean, r.best, r.floor)
    if collisions:
        log.warning("%d name keys collided inside a school and were left "
                    "unmatched rather than resolved arbitrarily", collisions)

    rows, hit_variant = [], {"first": 0, "second": 0}
    for r in board.itertuples(index=False):
        found = None
        for i, key in enumerate(sorted(name_keys(r.name))):
            found = lookup.get((r.school, key))
            if found:
                hit_variant["first" if i == 0 else "second"] += 1
                break
        rows.append(found or (np.nan,) * 4)

    out = board.copy()
    out[["games", "mean", "best", "floor"]] = pd.DataFrame(rows,
                                                           index=board.index)
    log.info("history attached to %d of %d priced players (%d needed the "
             "alternate spelling)", int(out["games"].notna().sum()), len(out),
             hit_variant["second"])
    return out


def map_warnings(mapping: dict, teams: list[dict], report: list) -> list[str]:
    """Which bindings deserve a second look before Saturday.

    The solver got UL right and I got it wrong: I said in advance it would be
    Louisiana, and the fixture list says Louisville. That is the argument for
    solving rather than hand-writing, but it is also the reason to print the
    cases where a code COULD have gone more than one way - a binding that was
    genuinely contested is worth thirty seconds of human attention, and one
    that had only one candidate is not.
    """
    meta = {t["school"]: (t.get("abbreviation"), t.get("alternateNames") or [])
            for t in teams}
    out = []
    for code, school in sorted(mapping.items()):
        rivals = [s for s in meta
                  if s != school and _similarity(code, s, *meta[s]) >= 1.0]
        if rivals:
            out.append(f"{code} -> {school}, but also matched "
                       f"{rivals[:4]} on the name alone. The fixture decided "
                       f"it; check the opponent looks right.")
    for fixture, resolved, score, evidence in report:
        if resolved is None:
            out.append(f"{fixture} matched NO college fixture on both sides.")
            continue
        m = re.search(r"margin ([\d.]+)", evidence or "")
        if m and float(m.group(1)) < 0.75:
            out.append(f"{fixture} -> {resolved} won by only {m.group(1)}. "
                       f"A near-tie is the binding most likely to be wrong.")
    return out
