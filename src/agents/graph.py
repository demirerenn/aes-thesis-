"""LangGraph orchestration — 13 core agents + 3 sub-reviewers (16 total).

This file implements the static skeleton of the pipeline described in
Agent_Sistem_Tasarimi_v1.2.docx §10. Each node is a placeholder that
wraps an LLM-backed agent function (to be implemented per-agent in
`src/agents/nodes/<name>.py`). The routing logic between nodes is
deterministic where possible; the feedback branch is conditional on
the evaluation report's `qwk` vs the 0.80 target.

Entry-point:
    python -m src.agents.graph --sprint 1

The skeleton intentionally avoids hard-coding LLM identities. Each
node receives a `llm_id` argument resolved from environment at
runtime (see `resolve_llm` below) so the model routing stays easy to
update when a new frontier model is released.
"""
from __future__ import annotations

import argparse
import os
from typing import Callable

from langgraph.graph import END, START, StateGraph

from .state import AESState


# -------------------------------------------------------------------
# LLM routing (stubs — to be replaced with real LangChain ChatModels)
# -------------------------------------------------------------------

LLM_ROUTING: dict[str, str] = {
    # Core agents (13)
    "orchestrator":      "claude-opus-4-6",
    "research":          "claude-opus-4-6",
    "data_analyst":      "claude-sonnet-4-6",
    "architect":         "claude-opus-4-6",
    "training_engineer": "gpt-5.3-codex",
    "evaluator":         "claude-sonnet-4-6",
    "feedback_strategy": "claude-opus-4-6",
    "code_reviewer":     "gpt-5.3-codex",
    "devops":            "claude-haiku-4-5",
    "thesis_writer":     "claude-opus-4-6",
    "fairness_auditor":  "claude-sonnet-4-6",
    "ops_monitor":       "claude-haiku-4-5",
    "peer_coordinator":  "claude-sonnet-4-6",
    # Sub-reviewers (3)
    "review_ml_logic":        "gemini-3.1-pro",
    "review_performance":     "gpt-5.4-pro",
    "review_reproducibility": "claude-sonnet-4-6",
}


def resolve_llm(agent_name: str) -> str:
    """Return the LLM id for a given agent; override via env var if set."""
    env_key = f"AES_LLM_{agent_name.upper()}"
    return os.getenv(env_key) or LLM_ROUTING[agent_name]


# -------------------------------------------------------------------
# Node placeholders
#
# Each real implementation lives in src/agents/nodes/<name>.py and
# exports a `run(state: AESState) -> AESState` callable. Until those
# are implemented, these stubs simply log the transition and pass
# state through unchanged. This keeps `python -m src.agents.graph`
# executable for DAG verification before the LLM calls are wired.
# -------------------------------------------------------------------

def _stub(agent_name: str) -> Callable[[AESState], AESState]:
    def node(state: AESState) -> AESState:
        print(f"[{agent_name:>22}] {resolve_llm(agent_name):>20}  —  entering node")
        state.setdefault("scratch", {}).setdefault("visits", {}).setdefault(agent_name, 0)
        state["scratch"]["visits"][agent_name] += 1
        return state
    return node


# -------------------------------------------------------------------
# Routing predicates
# -------------------------------------------------------------------

def route_after_eval(state: AESState) -> str:
    """Conditional edge after Evaluator: continue or iterate."""
    qwk = state.get("best_qwk", 0.0)
    if qwk >= 0.80:
        return "thesis_writer"  # GO
    if qwk >= 0.72:
        return "feedback_strategy"  # ITERATE / REVISE
    return "feedback_strategy"      # ROLLBACK still routes via feedback agent


def route_after_review(state: AESState) -> str:
    """After peer-review panel: back to architect for rev1 or forward to engineer."""
    approvals = state.get("scratch", {}).get("review_approvals", 0)
    return "training_engineer" if approvals >= 3 else "architect"


# -------------------------------------------------------------------
# Build graph
# -------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(AESState)

    # Register nodes (core + sub-reviewers)
    for name in [
        "orchestrator", "research", "data_analyst", "architect",
        "review_ml_logic", "review_performance", "review_reproducibility",
        "training_engineer", "evaluator", "feedback_strategy",
        "code_reviewer", "devops", "fairness_auditor",
        "ops_monitor", "peer_coordinator", "thesis_writer",
    ]:
        g.add_node(name, _stub(name))

    # Static edges — linear backbone
    g.add_edge(START, "orchestrator")
    g.add_edge("orchestrator", "research")
    g.add_edge("research", "data_analyst")
    g.add_edge("data_analyst", "architect")

    # Peer-review fan-out / fan-in via peer_coordinator
    g.add_edge("architect", "peer_coordinator")
    g.add_edge("peer_coordinator", "review_ml_logic")
    g.add_edge("peer_coordinator", "review_performance")
    g.add_edge("peer_coordinator", "review_reproducibility")
    g.add_edge("review_ml_logic", "peer_coordinator")
    g.add_edge("review_performance", "peer_coordinator")
    g.add_edge("review_reproducibility", "peer_coordinator")
    g.add_conditional_edges("peer_coordinator", route_after_review, {
        "architect": "architect",
        "training_engineer": "training_engineer",
    })

    # Training & evaluation loop
    g.add_edge("training_engineer", "code_reviewer")
    g.add_edge("code_reviewer", "devops")
    g.add_edge("devops", "evaluator")
    g.add_edge("evaluator", "fairness_auditor")
    g.add_edge("fairness_auditor", "ops_monitor")
    g.add_conditional_edges("ops_monitor", route_after_eval, {
        "thesis_writer": "thesis_writer",
        "feedback_strategy": "feedback_strategy",
    })

    # Feedback → architect (next ADR revision) OR direct thesis
    g.add_edge("feedback_strategy", "architect")
    g.add_edge("thesis_writer", END)

    return g


def main() -> int:
    ap = argparse.ArgumentParser(description="AES LangGraph DAG skeleton")
    ap.add_argument("--sprint", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="Validate graph without invoking nodes")
    args = ap.parse_args()

    g = build_graph()
    app = g.compile()

    if args.dry_run:
        # Export Mermaid diagram for the thesis appendix
        try:
            print(app.get_graph().draw_mermaid())
        except Exception as e:
            print(f"[WARN] could not draw graph: {e}")
        return 0

    state: AESState = {
        "messages": [],
        "artifacts": [],
        "decisions": [],
        "best_qwk": 0.0,
        "sprint": args.sprint,
        "scratch": {},
    }
    final = app.invoke(state, config={"recursion_limit": 50})
    print("\n=== Final state summary ===")
    print("visits:", final.get("scratch", {}).get("visits"))
    print("best_qwk:", final.get("best_qwk"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
