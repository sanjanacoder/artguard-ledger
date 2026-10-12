import pytest

import config
from core import scoring
from core.models import MatchResult, Piece


def piece(label="none", posted=True, opted_out=False, glazed=False, verified=True) -> Piece:
    p = Piece(filename="a.jpg", mime="image/jpeg", original_bytes=b"", sha256_original="x", width=1, height=1,
              publicly_posted=posted, opted_out=opted_out, style_protected=glazed)
    p.match = MatchResult(label, 0.5, [])
    p.tag_verified = verified
    return p


def test_max_score_is_100():
    worst = {"strong_match": True, "possible_match": False, "no_dnt_tag": True, "no_opt_out": True,
             "publicly_posted": True, "no_style_protection": True}
    assert scoring.score(worst)[0] == 100
    w = config.WEIGHTS
    assert w["strong_match"] + w["no_dnt_tag"] + w["no_opt_out"] + w["publicly_posted"] + w["no_style_protection"] == 100


def test_strong_and_possible_are_exclusive():
    with pytest.raises(ValueError):
        scoring.score({"strong_match": True, "possible_match": True})


@pytest.mark.parametrize("signal", list(config.WEIGHTS))
def test_each_signal_adds_its_weight(signal):
    total, breakdown = scoring.score({signal: True})
    assert total == config.WEIGHTS[signal]
    assert breakdown == {signal: config.WEIGHTS[signal]}


@pytest.mark.parametrize("total,label", [(0, "Lower"), (29, "Lower"), (30, "Medium"), (59, "Medium"),
                                         (60, "High"), (100, "High")])
def test_label_boundaries(total, label):
    assert scoring.label_for(total) == label


def test_apply_strong_match_worst_case():
    p = piece("strong")
    scoring.apply(p)
    assert (p.score, p.label) == (100, "High")
    assert p.next_steps[0].title == "Review the matched copies"


def test_apply_best_case():
    p = piece("none", posted=False, opted_out=True, glazed=True)
    scoring.apply(p)
    # The posted copy never has the tag, so 15 points always remain.
    assert (p.score, p.label) == (15, "Lower")
    statuses = {s.title: s.status for s in p.next_steps}
    assert statuses["Opt-out registration (self-reported)"] == "cant_verify"
    assert statuses["Style protection (self-reported)"] == "cant_verify"


def test_steps_sorted_by_points():
    p = piece("possible")
    scoring.apply(p)
    points = [s.points for s in p.next_steps]
    assert points == sorted(points, reverse=True)


def test_failed_tag_is_reported():
    p = piece(verified=False)
    scoring.apply(p)
    assert any(s.status == "failed" for s in p.next_steps)
