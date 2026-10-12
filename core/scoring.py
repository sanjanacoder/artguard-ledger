"""Exposure score: published weights over simple, explainable signals."""
from __future__ import annotations

import config
from core.models import NextStep, Piece
from core.resources import resource


def signals(piece: Piece) -> dict[str, bool]:
    label = piece.match.label if piece.match else "none"
    return {
        "strong_match": label == "strong",
        "possible_match": label == "possible",
        # The score describes the copy that is already online, which has no tag.
        # The tagged copy only helps once the artist re-uploads it.
        "no_dnt_tag": True,
        "no_opt_out": not piece.opted_out,
        "publicly_posted": piece.publicly_posted,
        "no_style_protection": not piece.style_protected,
    }


def score(sig: dict[str, bool], weights: dict[str, int] = config.WEIGHTS) -> tuple[int, dict[str, int]]:
    if sig.get("strong_match") and sig.get("possible_match"):
        raise ValueError("strong and possible matches are mutually exclusive")
    breakdown = {k: weights[k] for k, on in sig.items() if on}
    return sum(breakdown.values()), breakdown


def label_for(total: int) -> str:
    if total >= config.LABEL_HIGH:
        return "High"
    if total >= config.LABEL_MEDIUM:
        return "Medium"
    return "Lower"


def next_steps(piece: Piece, sig: dict[str, bool]) -> list[NextStep]:
    w = config.WEIGHTS
    steps: list[NextStep] = []
    if sig["strong_match"]:
        steps.append(NextStep(
            "Review the matched copies",
            "A near-copy of this piece is in the searched index. Check the neighbors below, then consider an "
            "opt-out or removal request to the dataset maintainer.",
            "needs_you", w["strong_match"], resource("haveibeentrained")["url"]))
    elif sig["possible_match"]:
        steps.append(NextStep(
            "Check the possible match",
            "Something similar is in the searched index. It may be an altered copy or just a similar image; "
            "compare the neighbors yourself.",
            "needs_you", w["possible_match"]))
    if piece.tag_verified:
        steps.append(NextStep(
            "Tagged copy created", "A copy with the Do Not Train metadata tag was created and verified.",
            "done_by_app", w["no_dnt_tag"]))
    else:
        steps.append(NextStep(
            "Tagged copy could not be verified",
            "The metadata tag could not be confirmed on this copy. See the action log.",
            "failed", w["no_dnt_tag"]))
    steps.append(NextStep(
        "Replace the posted file with the tagged copy",
        "Re-upload the tagged copy wherever this piece is posted. Many platforms strip metadata on upload, "
        "so the tag may not survive.",
        "needs_you", w["no_dnt_tag"]))
    if sig["no_opt_out"]:
        steps.append(NextStep(
            "Register an opt-out", "Add this work to a Do Not Train registry.",
            "needs_you", w["no_opt_out"], resource("spawning")["url"]))
    else:
        steps.append(NextStep(
            "Opt-out registration (self-reported)", "You said this piece is registered. The app can't check this.",
            "cant_verify", 0))
    if sig["no_style_protection"]:
        steps.append(NextStep(
            "Consider style protection before posting new work",
            "Tools such as Glaze add changes meant to disrupt style mimicry. They run on your own computer.",
            "needs_you", w["no_style_protection"], resource("glaze")["url"]))
    else:
        steps.append(NextStep(
            "Style protection (self-reported)", "You said this piece is protected. The app can't check this.",
            "cant_verify", 0))
    steps.append(NextStep(
        "Whether AI developers honor these signals",
        "Opt-outs and metadata tags only work if trainers respect them. See the coverage map.",
        "cant_verify", 0))
    order = {"failed": 0, "needs_you": 1, "done_by_app": 2, "cant_verify": 3}
    return sorted(steps, key=lambda s: (-s.points, order[s.status]))


def apply(piece: Piece) -> None:
    sig = signals(piece)
    piece.score, piece.breakdown = score(sig)
    piece.label = label_for(piece.score)
    piece.next_steps = next_steps(piece, sig)
