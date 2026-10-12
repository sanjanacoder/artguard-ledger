import itertools

import pytest

import config
from core import rights
from core.hashing import sha256_bytes

METHODS = list(rights.questions()["method"]["options"])
YN = ["yes", "no", "na"]


def test_sha256_known_vector():
    assert sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.mark.parametrize("method,modified,arranged", list(itertools.product(METHODS, YN, YN)))
def test_every_answer_combination_has_a_summary(method, modified, arranged):
    r = rights.summarize({"method": method, "modified": modified, "arranged": arranged})
    assert r["summary"] and r["status"]
    assert r["disclaimer"] == config.NOT_LEGAL_ADVICE
    assert len(r["sources"]) >= 2


def test_human_work():
    assert rights.summarize({"method": "human"})["key"] == "human"


def test_follow_up_answers_ignored_for_human_work():
    assert rights.summarize({"method": "human", "modified": "yes"})["key"] == "human"


def test_mostly_ai_variants():
    assert rights.summarize({"method": "ai_mostly", "modified": "no", "arranged": "no"})["key"] == "ai_mostly"
    assert rights.summarize({"method": "ai_mostly", "arranged": "yes"})["key"] == "ai_mostly_arranged"
    assert rights.summarize({"method": "ai_mostly", "modified": "yes"})["key"] == "ai_mostly_modified"


def test_empty_answers_default_to_human():
    assert rights.summarize(None)["key"] == "human"
