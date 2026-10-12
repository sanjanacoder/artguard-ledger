from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

Status = Literal["done_by_app", "needs_you", "cant_verify", "failed"]
MatchLabel = Literal["strong", "possible", "none"]

STATUS_TEXT = {
    "done_by_app": "Done by app",
    "needs_you": "Needs you",
    "cant_verify": "Can't verify",
    "failed": "Failed",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Action:
    ts_utc: str
    piece_id: str | None  # None = batch-level
    step: str
    status: Status
    detail: str


class ActionLog:
    """Append-only record of everything the app did. Reports render from this."""

    def __init__(self) -> None:
        self.actions: list[Action] = []

    def add(self, step: str, status: Status, detail: str, piece_id: str | None = None) -> Action:
        action = Action(utc_now(), piece_id, step, status, detail)
        self.actions.append(action)
        return action

    def for_piece(self, piece_id: str) -> list[Action]:
        return [a for a in self.actions if a.piece_id == piece_id]


@dataclass
class MatchResult:
    label: MatchLabel
    top_score: float
    neighbors: list[dict]  # [{id, score, source, title, thumb}]
    phash_distance: int | None = None
    upgraded_by_phash: bool = False


@dataclass
class NextStep:
    title: str
    detail: str
    status: Status
    points: int = 0
    link: str | None = None


@dataclass
class Piece:
    filename: str
    mime: str
    original_bytes: bytes
    sha256_original: str
    width: int
    height: int
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    publicly_posted: bool = True
    opted_out: bool = False  # self-reported
    style_protected: bool = False  # self-reported (Glaze etc.)
    ai_answers: dict = field(default_factory=dict)
    match: MatchResult | None = None
    tagged_bytes: bytes | None = None
    sha256_tagged: str | None = None
    tag_engine: str | None = None
    tag_verified: bool = False
    rights_key: str | None = None
    rights_summary: str | None = None
    score: int | None = None
    label: str | None = None
    breakdown: dict[str, int] = field(default_factory=dict)
    next_steps: list[NextStep] = field(default_factory=list)
