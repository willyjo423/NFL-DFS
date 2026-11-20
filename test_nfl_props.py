"""Offline checks for nfl_props (no network)."""
import nfl_props as P


def _ou(desc, line, o, u):
    return {"description": desc, "period": {"description": "Game"},
            "outcomes": [{"description": "Over", "type": "O", "price": {"handicap": str(line), "american": o}},
                         {"description": "Under", "type": "U", "price": {"handicap": str(line), "american": u}}]}


def _ms(desc, items):
    return {"description": desc, "period": {"description": "Game"},
            "outcomes": [{"description": t, "price": {"american": a}} for t, a in items]}


EV = {"displayGroups": [
    {"description": "TD Scorer Props", "markets": [
        _ms("Anytime Touchdown Scorer", [("Rashee Rice (KC)", "+160"), ("Patrick Mahomes (KC)", "+600")]),
        _ms("Player to Score 2 or More Touchdowns", [("Rashee Rice (KC)", "+1200")]),
        {"description": "Anytime Touchdown Scorer", "period": {"description": "First Half"},
         "outcomes": [{"description": "Rashee Rice (KC) - 1H", "price": {"american": "+300"}}]}]},
    {"description": "Passing Yards", "markets": [
        _ou("Total Passing Yards - Patrick Mahomes (KC)", 255.5, "-115", "-115"),
        _ms("Alternate Passing Yards - Patrick Mahomes (KC)", [("200+ Passing Yards", "-400"), ("300+ Passing Yards", "+250")])]},
    {"description": "Passing Props", "markets": [
        _ou("Total Passing Touchdowns - Patrick Mahomes (KC)", 1.5, "-140", "+110"),
        _ou("Total Interceptions Thrown - Patrick Mahomes (KC)", 0.5, "+120", "-150"),
        _ou("Total Passing Attempts - Patrick Mahomes (KC)", 35.5, "-115", "-115")]},
    {"description": "Rushing Yards", "markets": [_ou("Total Rushing Yards - Patrick Mahomes (KC)", 18.5, "-115", "-115")]},
    {"description": "Receiving Props", "markets": [
        _ou("Total Receptions - Rashee Rice (KC)", 6.5, "-110", "-120"),
        _ms("Alternate Receptions - Rashee Rice (KC)", [("5+ Receptions", "-300"), ("8+ Receptions", "+200")])]},
    {"description": "Receiving Yards", "markets": [
        _ou("Total Receiving Yards - Rashee Rice (KC)", 72.5, "-115", "-115"),
        _ms("Alternate Receiving Yards - Rashee Rice (KC)", [("50+ Receiving Yards", "-250"), ("100+ Receiving Yards", "+300")])]},
]}


def test_parse():
    out = P.parse_event(EV)
    qb, wr = out[("Patrick Mahomes", "KC")], out[("Rashee Rice", "KC")]
    assert {"PY", "PTD", "INT", "RY", "TD"} <= set(qb) and "attempts" not in str(qb)
    assert {"REC", "RECY", "TD"} <= set(wr)
    assert len(wr["TD"]) == 2                       # 1H list ignored


def test_market_dk():
    out = P.parse_event(EV)
    qb = P.market_dk(out[("Patrick Mahomes", "KC")], "QB")
    wr = P.market_dk(out[("Rashee Rice", "KC")], "WR")
    assert 17 < qb["dk"] < 26, qb
    assert 14 < wr["dk"] < 22, wr
    assert P.market_dk({"PY": [(256, 0.5)]}, "QB") is None


def test_weight():
    assert P.weight(10, 10.5) == P.W_MARKET and P.weight(10, 14) == P.W_GAP_MAX


if __name__ == "__main__":
    test_parse(); test_market_dk(); test_weight(); print("nfl_props ok")
