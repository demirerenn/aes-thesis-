"""Shared state schema for the LangGraph AES orchestration.

Agent nodes read and write structured messages on this state. Each
artifact (EDA report, literature review, ADR, training summary,
evaluation report) is tracked as a typed entry so downstream agents
can reason about it programmatically.

Fan-in semantics
----------------
The pipeline runs peer-review as a parallel fan-out (3 reviewers) that
fans back into ``peer_coordinator``. LangGraph's default merge for
plain ``list`` / ``dict`` fields is last-write-wins, which silently
drops parallel updates. To keep all reviewer decisions and artifacts,
``decisions``, ``artifacts`` and ``scratch`` are annotated with
dedicated reducers below.
"""
from __future__ import annotations

import operator
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
    decision: Literal["go", "iterate", "iterate_a", "revise", "rollback", "approve", "reject"]
    rationale: str
    target_run: str


def _merge_scratch(a, b):
    """Deep-ish merge for the free-form scratch dict."""
    out = dict(a or {})
    for k, v in (b or {}).items():
        prev = out.get(k)
        if isinstance(v, dict) and isinstance(prev, dict):
            out[k] = {**prev, **v}
        elif isinstance(v, list) and isinstance(prev, list):
            out[k] = prev + v
        elif isinstance(v, set) and isinstance(prev, set):
            out[k] = prev | v
        else:
            out[k] = v
    return out


class AESState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    artifacts: Annotated[list[Artifact], operator.add]
    decisions: Annotated[list[Decision], operator.add]
    current_run_name: str | None
    current_config_path: str | None
    current_run_dir: str | None
    best_qwk: float
    sprint: int
    scratch: Annotated[dict[str, Any], _merge_scratch]
