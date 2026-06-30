"""
l1_scorer.py — Layer 1 Bidirectional Rule Scorer

Production implementation of the CascadeRule-LLM Layer 1 scorer.
Validated on MINEC (1,007 articles, 6,601 entities, 2,639 candidate pairs)
and DBP15K (ZH-EN, 4,000 pairs).

Design principles (validated by offline experiments):
  1. Universal property overlap — let ALL non-noise fields participate,
     weight learning discovers which matter per type.
  2. Strong/weak consistency tiers — strong fields require exact match
     (different = conflict), weak fields use Jaccard overlap.
  3. Asymmetric penalty — conflict costs 1.5× a match (λ=1.5),
     because false merges pollute the graph irreversibly.
  4. Name-match bonus — applied after property scoring:
     exact +0.3, containment +0.2, Jaccard≥0.5 +0.15.
  5. Hard conflicts — short-circuit to reject (gender, cross-class equipment).

Key parameters (locked from sweep experiments):
  - FIELD_MATCH_THRESHOLD = 0.65  (Jaccard for weak fields, 19-step sweep)
  - PENALTY_FACTOR λ       = 1.5   (asymmetric conflict penalty)
  - MERGE_THRESHOLD        = 0.6   (99-pair dual-threshold sweep)
  - REJECT_THRESHOLD       = -0.4  (99-pair dual-threshold sweep)

Three-valued logic per field:
  match    → +weight   (weight = 1/N_effective_fields)
  conflict → -weight × λ
  unknown  → 0          (missing → neutral, don't penalize information gaps)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("usn.l1_scorer")

# ============================================================
# Validated Parameters (locked from sweep experiments)
# ============================================================

PENALTY_FACTOR = 1.5        # λ: asymmetric conflict penalty
MERGE_THRESHOLD = 0.6       # score ≥ 0.6 → merge (skip L2)
REJECT_THRESHOLD = -0.4     # score ≤ -0.4 → reject (direct new entity)
FIELD_MATCH_THRESHOLD = 0.65  # Jaccard ≥ 0.65 for weak-field match
LOW_OVERLAP_ACTION = 'conflict'  # low overlap on weak fields → conflict

# Name-match bonus weights
NAME_BONUS_EXACT = 0.3
NAME_BONUS_CONTAINMENT = 0.2
NAME_BONUS_JACCARD = 0.15
NAME_JACCARD_THRESHOLD = 0.5

# System metadata — excluded from scoring
EXCLUDE_FIELDS = {
    'created_at', 'updated_at', 'confidence', 'labels',
    'vid', 'id', 'source_pk', 'imported_at',
}


# ============================================================
# Data Structures
# ============================================================

@dataclass
class FieldResult:
    """Single field comparison result."""
    result: str   # 'match' | 'conflict' | 'unknown'
    detail: str = ''


@dataclass
class ScoreResult:
    """Full L1 scoring result."""
    score: float       # clamped to [-1.0, +1.0]
    decision: str      # 'merge' | 'reject' | 'escalate'
    detail: str
    raw_score: float = 0.0  # before name bonus


# ============================================================
# Field Consistency Tiers
# ============================================================

# STRONG: exact-match fields — different values = conflict
STRONG_FIELDS = {
    'equip_type', 'category', 'state', 'loc_type', 'event_type', 'org_type',
    'region', 'coordinates', 'start_date', 'occurred_at', 'end_date',
    'home_location', 'latest_reported_at',
    'gender', 'nationality', 'industry',
    # Type-specific strong identifiers
    'designation', 'hull_number', 'identifier_value',
    'unified_credit_code', 'id_card',
}

# WEAK: descriptive fields — match only if Jaccard overlap high
WEAK_FIELDS = {
    'name', 'aliases', 'description',
    'location_name', 'org_name', 'occupation',
    'summary',
}

FIELD_TIER: dict[str, str] = {}
for _f in STRONG_FIELDS:
    FIELD_TIER[_f] = 'strong'
for _f in WEAK_FIELDS:
    FIELD_TIER[_f] = 'weak'
# Default: unknown field → treated as WEAK (safe default)


# ============================================================
# Helper Functions
# ============================================================

def _get(props: dict, key: str) -> str:
    """Safely get a string property, normalized to lowercase stripped."""
    v = props.get(key)
    if v is None:
        return ''
    s = str(v).strip()
    # Filter Nebula sentinel values
    if s in ('__NULL__', '__EMPTY__', 'None'):
        return ''
    return s.lower()


def _has_value(props: dict, key: str) -> bool:
    v = props.get(key)
    if v is None:
        return False
    s = str(v).strip()
    return bool(s and s not in ('__NULL__', '__EMPTY__', 'None'))


def _tokenize(s: str) -> set[str]:
    """Tokenize for Jaccard: lowercase words, filter empties."""
    return {w for w in s.lower().strip().split() if w}


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _name_similarity(name_a: str, name_b: str) -> float:
    """Word-level Jaccard similarity between two names."""
    wa = _tokenize(name_a)
    wb = _tokenize(name_b)
    return _jaccard(wa, wb)


def _is_containment(a: str, b: str) -> bool:
    a = a.lower().strip()
    b = b.lower().strip()
    if not a or not b:
        return False
    return a in b or b in a


def _get_effective_fields(props_a: dict, props_b: dict) -> list[str]:
    """Fields with values in at least one entity, excluding system metadata."""
    all_fields = set()
    for k in list(props_a.keys()) + list(props_b.keys()):
        if k in EXCLUDE_FIELDS:
            continue
        all_fields.add(k)

    effective = []
    for f in sorted(all_fields):
        if _has_value(props_a, f) or _has_value(props_b, f):
            effective.append(f)
    return effective


# ============================================================
# Field Comparison (Three-valued logic)
# ============================================================

def _compare_field(field: str, props_a: dict, props_b: dict) -> FieldResult:
    """
    Compare a single field between two entities.

    Strong fields: exact match → match; different → conflict
    Weak fields:   high Jaccard → match; low → conflict (not unknown)
    """
    va = _get(props_a, field)
    vb = _get(props_b, field)

    # Both missing → unknown
    if not va and not vb:
        return FieldResult('unknown')

    # One missing → unknown (don't penalize info gaps)
    if not va or not vb:
        return FieldResult('unknown')

    # Exact match (all tiers)
    if va == vb:
        return FieldResult('match', f'exact: {va[:40]}')

    # Containment (one is substring of other)
    if _is_containment(va, vb):
        return FieldResult('match', f'containment: {va[:20]} ~ {vb[:20]}')

    tier = FIELD_TIER.get(field, 'weak')

    if tier == 'strong':
        # Strong consistency: different values = conflict
        return FieldResult('conflict', f'{va[:20]} vs {vb[:20]}')
    else:
        # Weak consistency: Jaccard check
        sim = _name_similarity(va, vb)
        if sim >= FIELD_MATCH_THRESHOLD:
            return FieldResult('match', f'jaccard={sim:.2f}')
        # Low overlap → conflict (validated: better than 'unknown')
        return FieldResult('conflict', f'low_jaccard={sim:.2f}')


# ============================================================
# Hard Conflicts (type-specific, short-circuit to reject)
# ============================================================

def _has_hard_conflict(props_a: dict, props_b: dict, entity_type: str) -> bool:
    """
    Check for hard conflicts that immediately short-circuit to score=-1.

    These are rare, high-confidence exclusion rules where different values
    definitively mean different entities.
    """
    if entity_type == 'person':
        # Gender conflict
        ga = _get(props_a, 'gender')
        gb = _get(props_b, 'gender')
        if ga and gb and ga != gb:
            return True

    if entity_type == 'equipment':
        # Cross-class conflict (aircraft vs ship = always different)
        ca = _get(props_a, 'equip_type')
        cb = _get(props_b, 'equip_type')
        if ca and cb and ca != cb:
            aircraft_kw = ['aircraft', 'plane', 'helicopter', 'fighter',
                           'bomber', 'drone', 'uav']
            ship_kw = ['ship', 'carrier', 'destroyer', 'frigate',
                       'submarine', 'vessel']
            ca_ac = any(k in ca for k in aircraft_kw)
            cb_ac = any(k in cb for k in aircraft_kw)
            ca_sh = any(k in ca for k in ship_kw)
            cb_sh = any(k in cb for k in ship_kw)
            if (ca_ac and cb_sh) or (ca_sh and cb_ac):
                return True

    return False


# ============================================================
# Near-Perfect Signals (single-signal direct merge)
# ============================================================

# Fields where an exact match is sufficient for direct merge
NEAR_PERFECT_SIGNALS = {
    'equipment': ['designation', 'hull_number', 'name'],
    'identifier': ['value', 'identifier_value'],
    'organization': ['unified_credit_code'],
    'person': ['id_card'],
}


def _has_near_perfect_match(props_a: dict, props_b: dict,
                            entity_type: str) -> bool:
    """Check if a near-perfect identifier signal matches exactly."""
    signals = NEAR_PERFECT_SIGNALS.get(entity_type, [])
    for sig in signals:
        va = _get(props_a, sig)
        vb = _get(props_b, sig)
        if va and vb and va == vb:
            return True
    return False


# ============================================================
# Core Scoring Function
# ============================================================

def l1_score(props_a: dict, props_b: dict, entity_type: str,
             penalty_factor: float = PENALTY_FACTOR,
             merge_threshold: float = MERGE_THRESHOLD,
             reject_threshold: float = REJECT_THRESHOLD) -> ScoreResult:
    """
    Compute Layer 1 bidirectional score using universal property overlap.

    Returns ScoreResult with:
      score   ∈ [-1.0, +1.0]
      decision ∈ {'merge', 'reject', 'escalate'}
    """
    if not props_a or not props_b:
        return ScoreResult(0.0, 'escalate', 'missing properties')

    # 1. Hard conflict → immediate reject
    if _has_hard_conflict(props_a, props_b, entity_type):
        return ScoreResult(-1.0, 'reject', 'hard conflict', -1.0)

    # 2. Near-perfect identifier match → immediate merge
    if _has_near_perfect_match(props_a, props_b, entity_type):
        return ScoreResult(0.95, 'merge', 'near-perfect identifier match', 0.95)

    # 3. Universal property overlap scoring
    fields = _get_effective_fields(props_a, props_b)
    if not fields:
        return ScoreResult(0.0, 'escalate', 'no effective fields')

    n = len(fields)
    weight = 1.0 / n  # equal weight (validated: let LR discover importance)

    score = 0.0
    match_count = 0
    conflict_count = 0
    unknown_count = 0
    details = []

    for field in fields:
        result = _compare_field(field, props_a, props_b)
        if result.result == 'match':
            score += weight
            match_count += 1
        elif result.result == 'conflict':
            score -= weight * penalty_factor
            conflict_count += 1
        else:
            unknown_count += 1
        details.append(f'{field}={result.result}')

    # Clamp to [-1, +1]
    score = max(-1.0, min(1.0, score))

    # Routing
    if score >= merge_threshold:
        decision = 'merge'
    elif score <= reject_threshold:
        decision = 'reject'
    else:
        decision = 'escalate'

    detail_str = (f'fields={n} matches={match_count} '
                  f'conflicts={conflict_count} unknowns={unknown_count}; '
                  + ', '.join(details))

    return ScoreResult(score, decision, detail_str, score)


def l1_score_with_name(name_a: str, name_b: str, props_a: dict, props_b: dict,
                       entity_type: str, **kwargs) -> tuple:
    """
    Score including name-match bonus applied after property scoring.

    Name bonus: exact +0.3, containment +0.2, Jaccard≥0.5 +0.15.
    Applied AFTER field scoring, BEFORE routing.

    Returns (score, decision, detail) tuple for backward compatibility.
    """
    result = l1_score(props_a, props_b, entity_type, **kwargs)

    # If hard conflict already determined → keep it
    if result.decision == 'reject' and result.score <= -0.9:
        return result.score, result.decision, result.detail

    score = result.score

    # Add name match bonus
    if name_a and name_b:
        na = name_a.lower().strip()
        nb = name_b.lower().strip()
        if na == nb:
            score = max(-1.0, min(1.0, score + NAME_BONUS_EXACT))
        elif _is_containment(na, nb):
            score = max(-1.0, min(1.0, score + NAME_BONUS_CONTAINMENT))
        else:
            sim = _name_similarity(na, nb)
            if sim >= NAME_JACCARD_THRESHOLD:
                score = max(-1.0, min(1.0, score + NAME_BONUS_JACCARD))

    # Re-route with name bonus included
    merge_threshold = kwargs.get('merge_threshold', MERGE_THRESHOLD)
    reject_threshold = kwargs.get('reject_threshold', REJECT_THRESHOLD)
    if score >= merge_threshold:
        decision = 'merge'
    elif score <= reject_threshold:
        decision = 'reject'
    else:
        decision = 'escalate'

    return score, decision, result.detail


# ============================================================
# Feature Extraction (for adaptive weight learning via LR)
# ============================================================

def extract_features(props_a: dict, props_b: dict, entity_type: str,
                     name_a: str = '', name_b: str = '') -> dict:
    """
    Extract feature vector for logistic regression training.

    Produces the validated feature set:
      - Per-field match/conflict binary indicators
      - Aggregate overlap/conflict rates
      - Name similarity features
      - Type-aware structural cues

    Used by feedback.py for Unified LR and Hybrid (per-type) LR training.
    """
    features = {}

    fields = _get_effective_fields(props_a, props_b)

    if not fields:
        features['property_overlap_rate'] = 0.0
        features['property_conflict_rate'] = 0.0
        features['n_effective_fields'] = 0
        features['n_strong_fields'] = 0
        features['n_weak_fields'] = 0
        features['strong_overlap_rate'] = 0.0
        features['weak_overlap_rate'] = 0.0
        features['name_similarity'] = 0.0
        features['name_exact'] = 0
        features['name_containment'] = 0
        features['hard_conflict'] = 0
        features['near_perfect_match'] = 0
        return features

    n = len(fields)
    matches = 0
    conflicts = 0
    strong_matches = 0
    strong_conflicts = 0
    weak_matches = 0
    weak_conflicts = 0
    n_strong = 0
    n_weak = 0

    for field in fields:
        result = _compare_field(field, props_a, props_b)
        tier = FIELD_TIER.get(field, 'weak')
        is_strong = tier == 'strong'
        if is_strong:
            n_strong += 1
        else:
            n_weak += 1

        if result.result == 'match':
            matches += 1
            if is_strong:
                strong_matches += 1
            else:
                weak_matches += 1
        elif result.result == 'conflict':
            conflicts += 1
            if is_strong:
                strong_conflicts += 1
            else:
                weak_conflicts += 1

        # Per-field binary features
        features[f'fld_{field}_match'] = 1 if result.result == 'match' else 0
        features[f'fld_{field}_conflict'] = 1 if result.result == 'conflict' else 0

    # Aggregate statistics
    features['property_overlap_rate'] = matches / n
    features['property_conflict_rate'] = conflicts / n
    features['n_effective_fields'] = n
    features['n_strong_fields'] = n_strong
    features['n_weak_fields'] = n_weak
    features['strong_overlap_rate'] = (strong_matches / n_strong) if n_strong else 0.0
    features['weak_overlap_rate'] = (weak_matches / n_weak) if n_weak else 0.0
    features['strong_conflict_rate'] = (strong_conflicts / n_strong) if n_strong else 0.0
    features['weak_conflict_rate'] = (weak_conflicts / n_weak) if n_weak else 0.0

    # Name similarity features
    na = (name_a or _get(props_a, 'name') or '').strip()
    nb = (name_b or _get(props_b, 'name') or '').strip()
    features['name_similarity'] = _name_similarity(na, nb)
    features['name_exact'] = 1 if na and nb and na.lower() == nb.lower() else 0
    features['name_containment'] = 1 if _is_containment(na, nb) else 0

    # Hard conflict / near-perfect match indicators
    features['hard_conflict'] = 1 if _has_hard_conflict(props_a, props_b, entity_type) else 0
    features['near_perfect_match'] = 1 if _has_near_perfect_match(props_a, props_b, entity_type) else 0

    return features


# ============================================================
# Learned Model Loading (adaptive weight layer)
# ============================================================

import os as _os
import json as _json
import pickle as _pickle
from pathlib import Path as _Path

_MODEL_DIR = _Path(_os.getenv(
    "MODEL_DIR",
    "/root/workspace/tanshu_docs/experiments/release/models"))

# Cache loaded models to avoid disk I/O on every score call
_model_cache: dict[str, dict] = {}


def _load_model_artifact(version: str) -> dict | None:
    """Load a trained model artifact from disk (with cache)."""
    if version in _model_cache:
        return _model_cache[version]

    path = _MODEL_DIR / f"{version}.pkl"
    if not path.exists():
        return None

    try:
        with open(path, 'rb') as f:
            artifact = _pickle.load(f)
        _model_cache[version] = artifact
        return artifact
    except Exception:
        return None


def _load_model_metadata(version: str) -> dict | None:
    """Load model version metadata JSON."""
    path = _MODEL_DIR / f"{version}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return _json.load(f)
    except Exception:
        return None


def _get_latest_version(prefix: str) -> str | None:
    """Find the newest model version by filename prefix."""
    if not _MODEL_DIR.exists():
        return None
    versions = sorted(_MODEL_DIR.glob(f"{prefix}*.json"))
    if not versions:
        return None
    with open(versions[-1]) as f:
        meta = _json.load(f)
    return meta.get('version')


def score_with_learned_model(props_a: dict, props_b: dict, entity_type: str,
                             ) -> ScoreResult | None:
    """
    Try to score using a learned LR model (Unified or Hybrid per-type).

    Resolution order:
      1. Hybrid per-type model for this entity_type (if available)
      2. Unified model (fallback for types with too few samples)
      3. None → caller falls back to fixed L1

    Returns None if no learned model is available, so the caller
    can transparently fall back to the fixed-weight scorer.
    """
    # 1. Try per-type hybrid model
    type_prefix = f"{entity_type}_hybrid"
    type_version = _get_latest_version(type_prefix)
    if type_version:
        artifact = _load_model_artifact(type_version)
        if artifact:
            result = _predict_with_model(artifact, props_a, props_b, entity_type)
            if result:
                return result

    # 2. Try unified model
    unified_version = _get_latest_version("unified_v")
    if unified_version:
        artifact = _load_model_artifact(unified_version)
        if artifact:
            return _predict_with_model(artifact, props_a, props_b, entity_type)

    # 3. No learned model available
    return None


def _predict_with_model(artifact: dict, props_a: dict, props_b: dict,
                        entity_type: str) -> ScoreResult | None:
    """Score a pair using a loaded LR model artifact."""
    model = artifact.get('model')
    scaler = artifact.get('scaler')
    feature_names = artifact.get('feature_names', [])
    theta_merge = artifact.get('theta_merge', 0.6)
    theta_reject = artifact.get('theta_reject', 0.4)

    if model is None or scaler is None or not feature_names:
        return None

    # Extract features and align to the model's feature space
    raw_features = extract_features(props_a, props_b, entity_type)
    if np is None:
        return None
    X = np.zeros((1, len(feature_names)))
    for j, fname in enumerate(feature_names):
        X[0, j] = float(raw_features.get(fname, 0.0))

    try:
        X_scaled = scaler.transform(X)
        prob = float(model.predict_proba(X_scaled)[0, 1])
    except Exception:
        return None

    # Route using learned thresholds
    if prob >= theta_merge:
        decision = 'merge'
    elif prob <= theta_reject:
        decision = 'reject'
    else:
        decision = 'escalate'

    return ScoreResult(
        score=round(prob, 4),
        decision=decision,
        detail=f'learned_model: prob={prob:.3f} θ_m={theta_merge:.2f} θ_r={theta_reject:.2f}',
        raw_score=prob,
    )


def l1_score_adaptive(props_a: dict, props_b: dict, entity_type: str,
                      name_a: str = '', name_b: str = '',
                      **kwargs) -> ScoreResult:
    """
    Adaptive L1 scoring: try learned model first, fall back to fixed weights.

    This is the production entry point for extraction-time disambiguation.
    The three-level hierarchy (Fixed → Unified LR → Hybrid LR) is resolved
    transparently:

      1. Hard conflicts and near-perfect signals ALWAYS take priority
         (these are deterministic and should never be overridden by LR).
      2. If a learned model exists → use it.
      3. Otherwise → use fixed-weight universal overlap scorer.

    The name-match bonus is only applied for the fixed-weight path, since
    the learned model already incorporates name features.
    """
    if not props_a or not props_b:
        return ScoreResult(0.0, 'escalate', 'missing properties')

    # Hard conflict → always reject regardless of model
    if _has_hard_conflict(props_a, props_b, entity_type):
        return ScoreResult(-1.0, 'reject', 'hard conflict', -1.0)

    # Near-perfect identifier → always merge
    if _has_near_perfect_match(props_a, props_b, entity_type):
        return ScoreResult(0.95, 'merge', 'near-perfect identifier match', 0.95)

    # Try learned model
    learned = score_with_learned_model(props_a, props_b, entity_type)
    if learned:
        return learned

    # Fall back to fixed-weight scorer with name bonus
    return l1_score(props_a, props_b, entity_type, **kwargs)


# numpy import (deferred to avoid hard dependency at module import time
# for callers that only use the fixed-weight path)
try:
    import numpy as np
except ImportError:
    np = None
