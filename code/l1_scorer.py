#!/usr/bin/env python3
"""
l1_scorer.py — Layer 1 Bidirectional Rule Scorer (universal property overlap + strong/weak field tiers)

DESIGN CHANGE from v1:
  v1: 5 entity types × 4 hand-picked signals with fixed weights
  v2: Universal property overlap across ALL non-noise fields + name bonus + hard conflicts

v2 philosophy:
  - Don't guess which fields matter. Let all non-empty attributes participate equally.
  - Weight learning (LR) will discover which fields matter per type.
  - Name matching is a universal bonus (exact/containment/Jaccard).
  - Hard conflicts are rare explicit rules (gender, cross-domain equipment).

Three-valued logic per field:
  match    → +weight  (where weight = 1/N_effective_fields)
  conflict → -weight × λ  (λ=1.5, asymmetric penalty)
  unknown  → 0             (missing → neutral)

Threshold routing:
  score >= 0.5  → merge
  score <= -0.3 → reject
  -0.3 < score < 0.5 → escalate to L2
"""

import re
from dataclasses import dataclass
from typing import Optional

# ============================================================
# Configuration
# ============================================================

PENALTY_FACTOR = 1.5
MERGE_THRESHOLD = 0.6
REJECT_THRESHOLD = -0.4

# Field comparison params (locked from sweep experiments)
FIELD_MATCH_THRESHOLD = 0.65   # Jaccard ≥ 0.65 for weak fields → match (from fine sweep)
LOW_OVERLAP_ACTION = 'conflict'  # low overlap on weak fields → conflict (not unknown)

# System metadata fields — excluded from scoring (not entity attributes)
EXCLUDE_FIELDS = {
    'created_at', 'updated_at', 'confidence', 'labels',
    'vid', 'id', 'source_pk',
}

# ============================================================
# Data Structures
# ============================================================

@dataclass
class FieldResult:
    result: str   # 'match' | 'conflict' | 'unknown'
    detail: str = ''

@dataclass
class ScoreResult:
    score: float
    decision: str  # 'merge' | 'reject' | 'escalate'
    detail: str

# ============================================================
# Helper Functions
# ============================================================

def _get(props: dict, key: str) -> str:
    v = props.get(key, '') or ''
    return str(v).strip().lower()

def _has_value(props: dict, key: str) -> bool:
    v = props.get(key, '')
    return bool(v and str(v).strip())

def _name_similarity(name_a: str, name_b: str) -> float:
    wa = set(name_a.lower().split())
    wb = set(name_b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)

def _is_containment(a: str, b: str) -> bool:
    a = a.lower().strip()
    b = b.lower().strip()
    if not a or not b:
        return False
    return a in b or b in a

def _get_effective_fields(props_a: dict, props_b: dict) -> list[str]:
    """Get fields that have values in at least one entity, excluding noise."""
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

# Field consistency tiers:
#   STRONG — exact-match fields (different value = conflict)
#   WEAK   — descriptive fields (match only if overlap high; low overlap = unknown, never conflict)
#             Also applies to free-text fields that may express same concept differently
STRONG_FIELDS = {
    'equip_type', 'category', 'state', 'loc_type', 'event_type', 'org_type',
    'region', 'coordinates', 'start_date', 'occurred_at', 'end_date',
    'home_location', 'latest_reported_at', 'gender', 'nationality', 'industry',
}
WEAK_FIELDS = {
    'name', 'aliases', 'description', 'location_name', 'org_name', 'occupation',
}
# Default: unknown field → treat as WEAK (safe: don't penalize unfamiliar descriptive fields)

FIELD_TIER = {}
for f in STRONG_FIELDS:
    FIELD_TIER[f] = 'strong'
for f in WEAK_FIELDS:
    FIELD_TIER[f] = 'weak'


def _compare_field(field: str, props_a: dict, props_b: dict) -> FieldResult:
    """Compare a single field between two entities using three-valued logic.
    
    Strong fields: exact match → match; different → conflict (clear disagreement)
    Weak fields:   high overlap → match; low overlap → unknown (no penalty)
    """
    va = _get(props_a, field)
    vb = _get(props_b, field)
    
    # Both missing → unknown
    if not va and not vb:
        return FieldResult('unknown')
    
    # One missing → unknown
    if not va or not vb:
        return FieldResult('unknown')
    
    # Exact match (all tiers)
    if va == vb:
        return FieldResult('match', f'exact: {va[:40]}')
    
    # Containment (all tiers — one is substring of other)
    if _is_containment(va, vb):
        return FieldResult('match', f'containment: {va[:20]} ~ {vb[:20]}')
    
    tier = FIELD_TIER.get(field, 'weak')  # default weak for unknown fields
    
    if tier == 'strong':
        # Strong consistency: different values = conflict
        return FieldResult('conflict', f'{va[:20]} vs {vb[:20]}')
    else:
        # Weak consistency: Jaccard check with locked params
        sim = _name_similarity(va, vb)
        if sim >= FIELD_MATCH_THRESHOLD:
            return FieldResult('match', f'sim={sim:.2f}')
        # Low overlap → conflict (not unknown), per sweep experiments
        return FieldResult('conflict', f'low_sim={sim:.2f}')

# ============================================================
# Hard Conflicts (type-specific, rare)
# ============================================================

def _has_hard_conflict(props_a: dict, props_b: dict, entity_type: str) -> bool:
    """Check for hard conflicts that short-circuit to score=-1."""
    
    # Person: gender conflict (only if both have values)
    if entity_type == 'person':
        ga = _get(props_a, 'gender')
        gb = _get(props_b, 'gender')
        if ga and gb and ga != gb:
            return True
    
    # Equipment: cross-domain conflict (aircraft vs ship)
    if entity_type == 'equipment':
        ca = _get(props_a, 'equip_type')
        cb = _get(props_b, 'equip_type')
        if ca and cb and ca != cb:
            aircraft_kw = ['aircraft', 'plane', 'helicopter', 'fighter', 'bomber', 'drone']
            ship_kw = ['ship', 'carrier', 'destroyer', 'frigate', 'submarine', 'vessel']
            ca_ac = any(k in ca for k in aircraft_kw)
            cb_ac = any(k in cb for k in aircraft_kw)
            ca_sh = any(k in ca for k in ship_kw)
            cb_sh = any(k in cb for k in ship_kw)
            if (ca_ac and cb_sh) or (ca_sh and cb_ac):
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
    
    Returns: ScoreResult(score, decision, detail)
    """
    if props_a is None or props_b is None:
        return ScoreResult(0.0, 'escalate', 'missing properties')
    
    # Hard conflict check
    if _has_hard_conflict(props_a, props_b, entity_type):
        return ScoreResult(-1.0, 'reject', 'hard conflict')
    
    # Get effective fields (excluding noise, requiring at least one value)
    fields = _get_effective_fields(props_a, props_b)
    if not fields:
        return ScoreResult(0.0, 'escalate', 'no effective fields')
    
    n_strong = sum(1 for f in fields if FIELD_TIER.get(f, 'weak') == 'strong')
    n_weak = len(fields) - n_strong
    n = len(fields)
    
    # Equal weight (wwf=1.0, locked from sweep experiments)
    weight = 1.0 / n
    weight_strong = weight
    weight_weak = weight
    
    # Score accumulation
    score = 0.0
    match_count = 0
    conflict_count = 0
    unknown_count = 0
    details = []
    
    for field in fields:
        result = _compare_field(field, props_a, props_b)
        tier = FIELD_TIER.get(field, 'weak')
        w = weight_strong if tier == 'strong' else weight_weak
        
        if result.result == 'match':
            score += w
            match_count += 1
        elif result.result == 'conflict':
            score -= w * penalty_factor
            conflict_count += 1
        else:
            unknown_count += 1
        details.append(f'{field}={result.result}')
    
    # Clamp
    score = max(-1.0, min(1.0, score))
    
    # Routing
    if score >= merge_threshold:
        decision = 'merge'
    elif score <= reject_threshold:
        decision = 'reject'
    else:
        decision = 'escalate'
    
    detail_str = f'fields={n} matches={match_count} conflicts={conflict_count} unknowns={unknown_count}; ' + ', '.join(details)
    
    return ScoreResult(score, decision, detail_str)


def l1_score_with_name(name_a: str, name_b: str, props_a: dict, props_b: dict,
                       entity_type: str, **kwargs) -> tuple:
    """
    Score including name-match bonus applied after property scoring.
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
            score = max(-1.0, min(1.0, score + 0.3))
            score = max(-1.0, min(1.0, score))
        elif _is_containment(na, nb):
            score = max(-1.0, min(1.0, score + 0.2))
        else:
            sim = _name_similarity(na, nb)
            if sim >= 0.5:
                score = max(-1.0, min(1.0, score + 0.15))
    
    # Re-route
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
# Feature Extraction (for LR training in Level 1/2)
# ============================================================

def extract_features(props_a: dict, props_b: dict, entity_type: str,
                     name_a: str = '', name_b: str = '') -> dict:
    """
    Extract feature vector for LR training.
    Returns dict of feature_name → value.
    """
    features = {}
    
    fields = _get_effective_fields(props_a, props_b)
    
    if not fields:
        features['property_overlap_rate'] = 0.0
        features['property_conflict_rate'] = 0.0
        features['n_effective_fields'] = 0
        features['name_similarity'] = 0.0
        return features
    
    n = len(fields)
    matches = 0
    conflicts = 0
    
    for field in fields:
        result = _compare_field(field, props_a, props_b)
        if result.result == 'match':
            matches += 1
        elif result.result == 'conflict':
            conflicts += 1
        
        # Per-field feature (binary)
        features[f'fld_{field}_match'] = 1 if result.result == 'match' else 0
        features[f'fld_{field}_conflict'] = 1 if result.result == 'conflict' else 0
    
    features['property_overlap_rate'] = matches / n
    features['property_conflict_rate'] = conflicts / n
    features['n_effective_fields'] = n
    
    # Name similarity
    features['name_similarity'] = _name_similarity(name_a or _get(props_a, 'name'),
                                                    name_b or _get(props_b, 'name'))
    features['name_exact'] = 1 if name_a and name_b and name_a.lower().strip() == name_b.lower().strip() else 0
    features['name_containment'] = 1 if _is_containment(name_a or '', name_b or '') else 0
    
    return features


if __name__ == '__main__':
    # Self-test
    props1 = {'name': 'Capt. Paul Lorence', 'org_name': 'U.S. Air Force', 'occupation': 'pilot', 'equip_type': ''}
    props2 = {'name': 'Paul Lorence', 'org_name': 'U.S. Air Force', 'occupation': 'pilot'}
    
    score, decision, detail = l1_score_with_name('Capt. Paul Lorence', 'Paul Lorence', props1, props2, 'person')
    print(f"Test: Capt. Paul Lorence vs Paul Lorence")
    print(f"  score={score:.3f}, decision={decision}")
    print(f"  detail: {detail}")
    
    print()
    props3 = {'name': 'F-35', 'equip_type': 'fighter', 'category': 'aircraft'}
    props4 = {'name': 'USS Reagan', 'equip_type': 'carrier', 'category': 'ship'}
    score, decision, detail = l1_score_with_name('F-35', 'USS Reagan', props3, props4, 'equipment')
    print(f"Test: F-35 vs USS Reagan (hard conflict)")
    print(f"  score={score:.3f}, decision={decision}")
