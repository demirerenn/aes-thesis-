"""AES — Automated Essay Scoring package for thesis project.

Exposes the three pillars of ADR-001-rev1:
    * data       — ASAP 1.0 / ASAP 2.0 loaders + stratified CV
    * losses     — CORN, CORAL, MSE+rank
    * models     — DeBERTa-v3 / Longformer backbones with ordinal heads
    * metrics    — QWK + fold-level bootstrap CI
    * training   — single-run trainer orchestrating the above
"""
from __future__ import annotations

__version__ = "0.1.0"
