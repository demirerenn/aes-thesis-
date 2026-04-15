"""Shared state schema for the LangGraph AES orchestration.

Agent nodes read and write structured messages on this state. Each
artifact (EDA report, literature review, ADR, training summary,
evaluation report) is tracked as a typed entry so downstream agents
can reason about it programmatically.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages


ArtifactKind = Literal[
    "eda_report",
    "literature_review",
    "adr",
    "peer_review",
    "training_summary",
    "evaluation_report",
    "feedback_note",
    "config",
    "code_patch",
]


class Artifact(TypedDict):
    kind: ArtifactKind
    path: str
    version: str
    produced_by: str
    summary: str


class Decision(TypedDict):
    agent: str
    decision: Literal["go", "iterate", "revise", "rollback"]
    rationale: str
    target_run: str


class AESState(TypedDict, total=False):
    # Conversation log shared across agents
    messages: Annotated[list, add_messages]
    # Structured knowledge
    artifacts: list[Artifact]
    decisions: list[Decision]
    # Latest run context
    current_run_name: str | None
    current_config_path: str | None
    best_qwk: float
    # Sprint pointer (1 = ASAP1-P2 baseline; 2 = SHORT bucket; ...)
    sprint: int
    # Free-form scratchpad keyed by agent name
    scratch: dict[str, Any]
