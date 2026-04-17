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
from pathlib import Path
from typing import Callable

from langgraph.graph import END, START, StateGraph

from .state import AESState
from .llm_factory import AGENT_CONFIG, get_chat_model, invoke_agent, make_agent_node
from .nodes.evaluator import EvaluatorInput, evaluate_run


# Paths used when the evaluator is invoked from inside a LangGraph run.
# All of them can be overridden per-state (state["current_run_dir"] …)
# or globally via environment variables. Keeping them as module-level
# constants avoids hardcoding any path inside make_evaluator_node().
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TEMPLATE_DIR = _REPO_ROOT / "src" / "agents" / "templates"
_DEFAULT_RUN_DIR = _REPO_ROOT / "runs"
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "baseline_asap1_p2.yaml"


# -------------------------------------------------------------------
# LLM routing — model atamaları (AGENT_CONFIG ile senkron tutulur)
# -------------------------------------------------------------------

LLM_ROUTING: dict[str, str] = {
    # Core agents (13) — rev3: Opus 4.7 upgrade (16 Nisan 2026)
    # Tier-1 Opus 4.7: stratejik muhakeme, mimari karar, kök-neden analizi
    # Aynı fiyat ($5/$25 per M token), 14 benchmark'ın 12'sinde 4.6'yı geçiyor
    "orchestrator":      "claude-opus-4-7",
    "research":          "claude-opus-4-7",
    "data_analyst":      "claude-opus-4-7",       # ↑ Sonnet→Opus: EDA içgörüleri Architect'i besliyor
    "architect":         "claude-opus-4-7",
    "training_engineer": "claude-opus-4-7",       # ↑ GPT-5.3→Opus: loss/HP stratejik kararlar
    "feedback_strategy": "claude-opus-4-7",
    "thesis_writer":     "claude-opus-4-7",
    "review_reproducibility": "claude-opus-4-7",  # ↑ Sonnet→Opus: tez metodolojik rigor
    # Tier-2 Sonnet: yapılandırılmış analiz, koordinasyon
    "evaluator":         "claude-sonnet-4-6",     # deterministik — LLM sadece opsiyonel cilalama
    "fairness_auditor":  "claude-sonnet-4-6",
    "devops":            "claude-sonnet-4-6",     # ↑ Haiku→Sonnet: ARM64+CUDA karmaşıklığı
    "ops_monitor":       "claude-sonnet-4-6",     # ↑ Haiku→Sonnet: anomali tespiti
    "peer_coordinator":  "claude-sonnet-4-6",
    # Tier-3 Cross-provider: kognitif çeşitlilik (peer review çapraz doğrulama)
    "code_reviewer":          "gpt-5.3-codex",    # farklı LLM → Claude kör noktalarını yakalar
    "review_ml_logic":        "gemini-3.1-pro",   # Gemini perspektifi
    "review_performance":     "gpt-5.4-pro",      # GPT perspektifi
}


def resolve_llm(agent_name: str) -> str:
    """Return the LLM id for a given agent; override via env var if set."""
    env_key = f"AES_LLM_{agent_name.upper()}"
    return os.getenv(env_key) or LLM_ROUTING[agent_name]


# -------------------------------------------------------------------
# Node factory
#
# Evaluator has a dedicated implementation (src/agents/nodes/evaluator.py)
# because it is deterministic (no LLM call). All other agents use
# make_agent_node() from llm_factory.py which:
#   1. Reads AGENT_CONFIG for system prompt + model
#   2. Builds context from AESState (decisions, QWK, run name)
#   3. Invokes the correct provider (Anthropic/OpenAI/Google)
#   4. Writes result back to state
#   5. Falls back to stub mode if API key is missing
# -------------------------------------------------------------------

def _stub(agent_name: str) -> Callable[[AESState], AESState]:
    """Fallback stub — only used for --dry-run DAG verification."""
    def node(state: AESState) -> AESState:
        print(f"[{agent_name:>22}] {resolve_llm(agent_name):>20}  —  stub (dry-run)")
        return {"scratch": {"visits": {agent_name: 1}}}
    return node


# -------------------------------------------------------------------
# Evaluator node factory — deterministic, no LLM call
# -------------------------------------------------------------------

def make_evaluator_node(
    run_dir: Path | str | None = None,
    config_path: Path | str | None = None,
    template_dir: Path | str | None = None,
) -> Callable[[AESState], AESState]:
    """Build a LangGraph node that calls the deterministic Evaluator.

    Resolution order for the three paths:
        1. Explicit argument to this factory.
        2. ``state["current_run_dir"]`` / ``state["current_config_path"]``.
        3. Environment variables (``AES_EVAL_RUN_DIR``, ``AES_EVAL_CONFIG``).
        4. Module-level defaults (``runs/``, ``configs/baseline_asap1_p2.yaml``).

    The node writes:
        * ``best_qwk``              — max(existing, new QWK)
        * ``decisions`` (append)    — one Decision with target_run set to the
                                      evaluator's next_agent (architect /
                                      feedback_strategy / thesis_writer)
        * ``artifacts`` (append)    — the evaluation_report artifact
        * ``scratch.last_evaluator_decision`` — the full raw result dict
    """
    tpl_default = Path(template_dir) if template_dir else _DEFAULT_TEMPLATE_DIR

    def node(state: AESState) -> AESState:
        _rd = (
            Path(run_dir) if run_dir else
            Path(state["current_run_dir"]) if state.get("current_run_dir") else
            Path(os.getenv("AES_EVAL_RUN_DIR", str(_DEFAULT_RUN_DIR)))
        )
        _cfg = (
            Path(config_path) if config_path else
            Path(state["current_config_path"]) if state.get("current_config_path") else
            Path(os.getenv("AES_EVAL_CONFIG", str(_DEFAULT_CONFIG)))
        )

        try:
            result = evaluate_run(EvaluatorInput(
                run_dir=_rd,
                config_path=_cfg,
                template_dir=tpl_default,
            ))
            qwk = float(result["qwk_mean"])
            next_agent = result["next_agent"]
            decision = result["decision"]
            print(
                f"[{'evaluator':>22}] {'(deterministic)':>20}  —  "
                f"QWK={qwk:.4f} decision={decision.upper()} → {next_agent}"
            )
            return {
                "best_qwk": max(state.get("best_qwk", 0.0), qwk),
                "decisions": [{
                    "agent": "evaluator",
                    "decision": decision,
                    "rationale": result["rationale"],
                    "target_run": next_agent,
                }],
                "artifacts": [{
                    "kind": "evaluation_report",
                    "path": result["report_path"],
                    "version": "1",
                    "produced_by": "evaluator",
                    "summary": f"QWK={qwk:.4f} → {decision.upper()}",
                }],
                "scratch": {
                    "last_evaluator_decision": {
                        "qwk_mean": qwk,
                        "decision": decision,
                        "next_agent": next_agent,
                        "rationale": result["rationale"],
                        "report_path": result["report_path"],
                    },
                    "visits": {"evaluator": 1},
                },
            }
        except (FileNotFoundError, KeyError, ValueError) as e:
            # No training artifacts yet — emit a ROLLBACK Decision so the
            # routing layer doesn't silently continue with stale QWK.
            print(f"[{'evaluator':>22}] ⚠ pre-condition failed: {e}")
            return {
                "decisions": [{
                    "agent": "evaluator",
                    "decision": "rollback",
                    "rationale": f"Evaluator pre-condition failed: {e}",
                    "target_run": "architect",
                }],
                "scratch": {
                    "last_evaluator_decision": {
                        "qwk_mean": 0.0,
                        "decision": "rollback",
                        "next_agent": "architect",
                        "rationale": str(e),
                        "report_path": "",
                    },
                    "visits": {"evaluator": 1},
                },
            }
    return node


# -------------------------------------------------------------------
# Routing predicates
# -------------------------------------------------------------------

_EVAL_ROUTE_TARGETS = ("thesis_writer", "feedback_strategy", "architect")
_REQUIRED_REVIEWERS = frozenset({
    "review_ml_logic", "review_performance", "review_reproducibility",
})


def route_after_eval(state: AESState) -> str:
    """Honour the evaluator's decision; fall back to QWK thresholds.

    Priority:
        1. ``state["scratch"]["last_evaluator_decision"]["next_agent"]``
           (populated by make_evaluator_node).
        2. Most recent Decision with agent=="evaluator" — use target_run.
        3. QWK thresholds (ADR-001-rev3):
             ≥ 0.82 → thesis_writer   (well above target → finalize)
             ≥ 0.78 → feedback_strategy (close → one more iterate-A pass)
             else   → architect         (rework)
    """
    last = state.get("scratch", {}).get("last_evaluator_decision") or {}
    next_agent = last.get("next_agent")
    if next_agent in _EVAL_ROUTE_TARGETS:
        return next_agent
    for dec in reversed(state.get("decisions", []) or []):
        if dec.get("agent") == "evaluator":
            tr = dec.get("target_run")
            if tr in _EVAL_ROUTE_TARGETS:
                return tr
            break
    qwk = state.get("best_qwk", 0.0)
    if qwk >= 0.82:
        return "thesis_writer"
    if qwk >= 0.78:
        return "feedback_strategy"
    return "architect"


def route_after_review(state: AESState) -> str:
    """Require unanimous approval from the three named reviewers.

    Any ``revise``/``reject``/``rollback`` from a required reviewer
    short-circuits back to the architect for rework. Duplicates from
    the same reviewer do NOT inflate the approval count because
    we key on the reviewer *agent name* via a set.
    """
    approved: set[str] = set()
    blocked = False
    for d in state.get("decisions", []) or []:
        agent = d.get("agent")
        if agent not in _REQUIRED_REVIEWERS:
            continue
        dec = (d.get("decision") or "").lower()
        if dec in {"go", "iterate", "iterate_a", "approve"}:
            approved.add(agent)
        elif dec in {"revise", "reject", "rollback"}:
            blocked = True
    if blocked:
        return "architect"
    return "training_engineer" if approved >= _REQUIRED_REVIEWERS else "architect"


# -------------------------------------------------------------------
# Graph builder
# -------------------------------------------------------------------

def _node_for(name: str, dry_run: bool) -> Callable[[AESState], AESState]:
    """Return the node callable for an agent name.

    * evaluator → deterministic make_evaluator_node (always, even in
      dry-run — it will emit a rollback Decision if the run_dir is
      empty, which exercises the error path).
    * All other agents:
        - dry_run=True  → _stub (prints + visit counter, no API call)
        - dry_run=False → make_agent_node (real LLM via llm_factory)
    """
    if name == "evaluator":
        return make_evaluator_node()
    if dry_run:
        return _stub(name)
    return make_agent_node(name)


def build_graph(dry_run: bool = False):
    """Assemble the 16-node LangGraph StateGraph."""
    g = StateGraph(AESState)

    # 13 core agents
    core_agents = [
        "orchestrator",
        "research",
        "data_analyst",
        "architect",
        "code_reviewer",
        "training_engineer",
        "evaluator",
        "feedback_strategy",
        "fairness_auditor",
        "devops",
        "ops_monitor",
        "peer_coordinator",
        "thesis_writer",
    ]
    for name in core_agents:
        g.add_node(name, _node_for(name, dry_run))

    # 3 peer sub-reviewers (fan-out from peer_coordinator)
    for name in ("review_ml_logic", "review_performance", "review_reproducibility"):
        g.add_node(name, _node_for(name, dry_run))

    # -------------------------------------------------------------------
    # Edges — linear sprint flow with two feedback branches
    # -------------------------------------------------------------------
    g.add_edge(START, "orchestrator")
    g.add_edge("orchestrator", "research")
    g.add_edge("research", "data_analyst")
    g.add_edge("data_analyst", "architect")
    g.add_edge("architect", "code_reviewer")
    g.add_edge("code_reviewer", "peer_coordinator")

    # Fan-out to 3 peer reviewers, then fan-in back to peer_coordinator
    # routing logic (handled by conditional edge below).
    for rev in ("review_ml_logic", "review_performance", "review_reproducibility"):
        g.add_edge("peer_coordinator", rev)

    # Fan-in: each reviewer's output is merged via the state reducers
    # (operator.add for decisions). We use a conditional edge keyed on
    # the merged state — all three must produce a Decision before we
    # advance. LangGraph naturally waits for all three parallel branches
    # to complete before evaluating the conditional predicate.
    # Each reviewer routes through route_after_review which returns the
    # next node once all three have approved.
    for rev in ("review_ml_logic", "review_performance", "review_reproducibility"):
        g.add_conditional_edges(
            rev,
            route_after_review,
            {
                "training_engineer": "training_engineer",
                "architect": "architect",
            },
        )

    g.add_edge("training_engineer", "devops")
    g.add_edge("devops", "ops_monitor")
    g.add_edge("ops_monitor", "evaluator")
    g.add_edge("evaluator", "fairness_auditor")

    # Main routing gate — after fairness_auditor we decide based on
    # the evaluator's decision (captured in scratch / decisions).
    g.add_conditional_edges(
        "fairness_auditor",
        route_after_eval,
        {
            "thesis_writer": "thesis_writer",
            "feedback_strategy": "feedback_strategy",
            "architect": "architect",
        },
    )

    # Feedback loop: feedback_strategy feeds back into architect for a
    # targeted iteration (ADR-001-rev3 "Rejim-A: iterate-A").
    g.add_edge("feedback_strategy", "architect")

    g.add_edge("thesis_writer", END)

    return g.compile()


# -------------------------------------------------------------------
# CLI — dry-run DAG verification + Mermaid export
# -------------------------------------------------------------------

def _render_mermaid(app, out_path: Path) -> None:
    """Write a Mermaid diagram of the compiled graph to disk."""
    try:
        mmd = app.get_graph().draw_mermaid()
    except Exception as e:  # pragma: no cover
        print(f"[graph] Mermaid export failed: {e}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(mmd, encoding="utf-8")
    print(f"[graph] Mermaid diagram written → {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sprint", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true",
                    help="Build graph with stub nodes; no API calls.")
    ap.add_argument("--mermaid", type=str, default=None,
                    help="If set, write a Mermaid diagram to this path.")
    ap.add_argument("--run-once", action="store_true",
                    help="Invoke the compiled graph once with an empty state.")
    args = ap.parse_args()

    app = build_graph(dry_run=args.dry_run)
    print(f"[graph] compiled — dry_run={args.dry_run} sprint={args.sprint}")

    if args.mermaid:
        _render_mermaid(app, Path(args.mermaid))

    if args.run_once:
        init_state: AESState = {
            "messages": [],
            "artifacts": [],
            "decisions": [],
            "best_qwk": 0.0,
            "sprint": args.sprint,
            "scratch": {"visits": {}},
        }
        final = app.invoke(init_state)
        visits = (final.get("scratch") or {}).get("visits", {})
        print(f"[graph] run-once complete — visits={visits}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
