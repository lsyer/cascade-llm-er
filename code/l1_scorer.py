#!/usr/bin/env python3
"""
l1_scorer.py — Layer 1 Bidirectional Rule Scorer
Implements the scoring formula from tanshu-graph-design-v1.6 §4.2.

Three-valued logic per signal:
  match    → +weight
  conflict → -weight × λ  (λ=1.5, asymmetric penalty)
  unknown  → 0             (missing data is neutral, NOT penalized)

Hard conflicts short-circuit to score=-1.0
Near-perfect signals (weight 0.9-1.0) can trigger immediate merge

Threshold routing:
  score >= 0.5  → merge
  score <= -0.3 → new entity (different)
  -0.3 < score < 0.5 → uncertain → escalate to L2
"""

import re
from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# Signal Definitions (code constants, per design doc GD §4.2)
# ============================================================

@dataclass
class SignalResult:
    result: str  # 'match' | 'conflict' | 'unknown'
    detail: str = ''

@dataclass
class Signal:
    name: str
    weight: float
    evaluator: callable  # (props_a, props_b) -> SignalResult

# ---- Helper functions for attribute comparison ----

def _get(props: dict, key: str) -> str:
    """Get a property value, cleaned and lowercased."""
    v = props.get(key, '') or ''
    return str(v).strip().lower()

def _name_similarity(name_a: str, name_b: str) -> float:
    """Jaccard similarity on word sets."""
    wa = set(name_a.lower().split())
    wb = set(name_b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)

def _is_containment(short: str, long: str) -> bool:
    """Check if one name is contained within the other."""
    short = short.lower().strip()
    long = long.lower().strip()
    if not short or not long:
        return False
    return short in long or long in short

# ---- Person signals ----

def sig_person_timeline(props_a, props_b):
    """Timeline: compare created_at/updated_at periods."""
    ca = _get(props_a, 'created_at')[:10]  # date part
    cb = _get(props_b, 'created_at')[:10]
    if not ca or not cb:
        return SignalResult('unknown')
    # If created within 90 days → likely same entity
    if ca == cb:
        return SignalResult('match', f'same date: {ca}')
    # Simple: same year → match (in news context, same entity references cluster)
    if ca[:4] == cb[:4]:
        return SignalResult('match', f'same year: {ca[:4]}')
    return SignalResult('conflict', f'different years: {ca} vs {cb}')

def sig_person_location(props_a, props_b):
    """Location: compare org_name as location proxy."""
    la = _get(props_a, 'org_name')
    lb = _get(props_b, 'org_name')
    if not la or not lb:
        return SignalResult('unknown')
    if la == lb:
        return SignalResult('match', f'same org: {la}')
    # Check containment (e.g., "US Navy" vs "U.S. Navy")
    if _is_containment(la, lb):
        return SignalResult('match', f'contained: {la} ~ {lb}')
    return SignalResult('conflict', f'different org: {la} vs {lb}')

def sig_person_org(props_a, props_b):
    """Organization: same as location but use occupation field."""
    oa = _get(props_a, 'occupation')
    ob = _get(props_b, 'occupation')
    if not oa or not ob:
        return SignalResult('unknown')
    if oa == ob:
        return SignalResult('match', f'same occupation: {oa}')
    if _is_containment(oa, ob):
        return SignalResult('match')
    return SignalResult('conflict', f'different occupation: {oa} vs {ob}')

def sig_person_social(props_a, props_b):
    """Social circle: use description/labels overlap."""
    da = set(_get(props_a, 'description').split())
    db = set(_get(props_b, 'description').split())
    if not da or not db:
        return SignalResult('unknown')
    overlap = len(da & db)
    if overlap >= 2:
        return SignalResult('match', f'overlap: {overlap} words')
    if overlap == 0:
        return SignalResult('conflict')
    return SignalResult('unknown')

# Person signals with weights (GD §4.2)
PERSON_SIGNALS = [
    Signal('timeline', 0.30, sig_person_timeline),
    Signal('location', 0.25, sig_person_location),
    Signal('org', 0.25, sig_person_org),
    Signal('social', 0.20, sig_person_social),
]

# ---- Equipment signals ----

def sig_equip_model(props_a, props_b):
    """Model/type match."""
    ma = _get(props_a, 'equip_type') or _get(props_a, 'name')
    mb = _get(props_b, 'equip_type') or _get(props_b, 'name')
    if not ma or not mb:
        return SignalResult('unknown')
    sim = _name_similarity(ma, mb)
    if sim >= 0.5 or ma == mb:
        return SignalResult('match', f'model sim={sim:.2f}')
    if _is_containment(ma, mb):
        return SignalResult('match', f'containment')
    if sim < 0.2:
        return SignalResult('conflict', f'low sim={sim:.2f}')
    return SignalResult('unknown')

def sig_equip_deploy(props_a, props_b):
    """Deployment location."""
    la = _get(props_a, 'home_location')
    lb = _get(props_b, 'home_location')
    if not la or not lb:
        return SignalResult('unknown')
    if la == lb or _is_containment(la, lb):
        return SignalResult('match')
    return SignalResult('conflict')

def sig_equip_affil(props_a, props_b):
    """Affiliation/parent unit."""
    pa = _get(props_a, 'parent_unit')
    pb = _get(props_b, 'parent_unit')
    if not pa or not pb:
        return SignalResult('unknown')
    if pa == pb or _is_containment(pa, pb):
        return SignalResult('match')
    return SignalResult('conflict')

def sig_equip_params(props_a, props_b):
    """Technical parameters: category + state."""
    ca = _get(props_a, 'category')
    cb = _get(props_b, 'category')
    if not ca or not cb:
        return SignalResult('unknown')
    if ca == cb:
        return SignalResult('match')
    return SignalResult('conflict')

EQUIPMENT_SIGNALS = [
    Signal('model', 0.40, sig_equip_model),
    Signal('deploy', 0.25, sig_equip_deploy),
    Signal('affiliation', 0.20, sig_equip_affil),
    Signal('params', 0.15, sig_equip_params),
]

# ---- Location signals ----

def sig_loc_region(props_a, props_b):
    """Region match."""
    ra = _get(props_a, 'region')
    rb = _get(props_b, 'region')
    if not ra or not rb:
        return SignalResult('unknown')
    if ra == rb:
        return SignalResult('match', f'same region: {ra}')
    return SignalResult('conflict', f'{ra} vs {rb}')

def sig_loc_type(props_a, props_b):
    """Location type."""
    ta = _get(props_a, 'loc_type')
    tb = _get(props_b, 'loc_type')
    if not ta or not tb:
        return SignalResult('unknown')
    if ta == tb:
        return SignalResult('match')
    return SignalResult('conflict')

def sig_loc_name(props_a, props_b):
    """Name containment (e.g., Norfolk vs Naval Station Norfolk)."""
    na = _get(props_a, 'name')
    nb = _get(props_b, 'name')
    if not na or not nb:
        return SignalResult('unknown')
    if na == nb:
        return SignalResult('match', f'exact: {na}')
    if _is_containment(na, nb):
        return SignalResult('match', f'containment: {na} ~ {nb}')
    sim = _name_similarity(na, nb)
    if sim >= 0.5:
        return SignalResult('match', f'sim={sim:.2f}')
    if sim < 0.15:
        return SignalResult('conflict', f'sim={sim:.2f}')
    return SignalResult('unknown')

def sig_loc_coords(props_a, props_b):
    """Coordinates match."""
    ca = _get(props_a, 'coordinates')
    cb = _get(props_b, 'coordinates')
    if not ca or not cb:
        return SignalResult('unknown')
    if ca == cb:
        return SignalResult('match', f'same coords')
    return SignalResult('conflict')

LOCATION_SIGNALS = [
    Signal('region', 0.35, sig_loc_region),
    Signal('admin', 0.30, sig_loc_type),
    Signal('name', 0.20, sig_loc_name),
    Signal('coords', 0.15, sig_loc_coords),
]

# ---- Event signals ----

def sig_event_time(props_a, props_b):
    """Time match."""
    ta = _get(props_a, 'occurred_at') or _get(props_a, 'start_date')
    tb = _get(props_b, 'occurred_at') or _get(props_b, 'start_date')
    if not ta or not tb:
        return SignalResult('unknown')
    if ta[:4] == tb[:4]:  # same year
        return SignalResult('match', f'same year: {ta[:4]}')
    return SignalResult('conflict', f'{ta} vs {tb}')

def sig_event_loc(props_a, props_b):
    """Location match."""
    la = _get(props_a, 'location_name')
    lb = _get(props_b, 'location_name')
    if not la or not lb:
        return SignalResult('unknown')
    if la == lb or _is_containment(la, lb):
        return SignalResult('match')
    return SignalResult('conflict')

def sig_event_type(props_a, props_b):
    """Event type match."""
    ta = _get(props_a, 'event_type')
    tb = _get(props_b, 'event_type')
    if not ta or not tb:
        return SignalResult('unknown')
    if ta == tb:
        return SignalResult('match')
    return SignalResult('conflict')

def sig_event_name(props_a, props_b):
    """Name similarity."""
    na = _get(props_a, 'name')
    nb = _get(props_b, 'name')
    if not na or not nb:
        return SignalResult('unknown')
    if na == nb:
        return SignalResult('match', 'exact')
    if _is_containment(na, nb):
        return SignalResult('match', 'containment')
    sim = _name_similarity(na, nb)
    if sim >= 0.4:
        return SignalResult('match', f'sim={sim:.2f}')
    if sim < 0.15:
        return SignalResult('conflict')
    return SignalResult('unknown')

EVENT_SIGNALS = [
    Signal('time', 0.35, sig_event_time),
    Signal('location', 0.30, sig_event_loc),
    Signal('participants', 0.20, sig_event_name),  # name as proxy
    Signal('type', 0.15, sig_event_type),
]

# ---- Organization signals (GD §4.2: hierarchy 0.30 + region 0.25 + timeline 0.25 + industry 0.20) ----

def sig_org_hierarchy(props_a, props_b):
    """Hierarchy/org_type match — same type of organization."""
    ta = _get(props_a, 'org_type')
    tb = _get(props_b, 'org_type')
    if not ta or not tb:
        return SignalResult('unknown')
    if ta == tb:
        return SignalResult('match', f'same org_type: {ta}')
    return SignalResult('conflict', f'{ta} vs {tb}')

def sig_org_region(props_a, props_b):
    """Region/country match."""
    ra = _get(props_a, 'region')
    rb = _get(props_b, 'region')
    if not ra or not rb:
        return SignalResult('unknown')
    if ra == rb or _is_containment(ra, rb):
        return SignalResult('match')
    return SignalResult('conflict', f'{ra} vs {rb}')

def sig_org_timeline(props_a, props_b):
    """Timeline: compare created_at."""
    ca = _get(props_a, 'created_at')[:10]
    cb = _get(props_b, 'created_at')[:10]
    if not ca or not cb:
        return SignalResult('unknown')
    if ca[:4] == cb[:4]:
        return SignalResult('match', f'same year: {ca[:4]}')
    return SignalResult('conflict', f'{ca} vs {cb}')

def sig_org_industry(props_a, props_b):
    """Industry/business scope match."""
    ia = _get(props_a, 'industry')
    ib = _get(props_b, 'industry')
    if not ia or not ib:
        return SignalResult('unknown')
    if ia == ib:
        return SignalResult('match', f'same industry: {ia}')
    return SignalResult('conflict', f'{ia} vs {ib}')

ORGANIZATION_SIGNALS = [
    Signal('org_type', 0.30, sig_org_hierarchy),
    Signal('region', 0.25, sig_org_region),
    Signal('timeline', 0.25, sig_org_timeline),
    Signal('industry', 0.20, sig_org_industry),
]

# ---- Signal map ----
SIGNAL_MAP = {
    'person': PERSON_SIGNALS,
    'equipment': EQUIPMENT_SIGNALS,
    'location': LOCATION_SIGNALS,
    'event': EVENT_SIGNALS,
    'organization': ORGANIZATION_SIGNALS,
}

# ============================================================
# Hard Conflicts (GD §4.2)
# ============================================================

def has_hard_conflict(props_a, props_b, entity_type):
    """Check for hard conflicts that short-circuit to score=-1."""
    # Person: gender conflict
    if entity_type == 'person':
        ga = _get(props_a, 'gender')
        gb = _get(props_b, 'gender')
        if ga and gb and ga != gb:
            return True
    
    # Equipment: different category entirely (e.g., aircraft vs ship)
    if entity_type == 'equipment':
        ca = _get(props_a, 'equip_type')
        cb = _get(props_b, 'equip_type')
        if ca and cb and ca != cb:
            # Check if fundamentally different
            aircraft_kw = ['aircraft', 'plane', 'helicopter', 'fighter', 'bomber']
            ship_kw = ['ship', 'carrier', 'destroyer', 'frigate', 'submarine']
            ca_ac = any(k in ca for k in aircraft_kw)
            cb_ac = any(k in cb for k in aircraft_kw)
            ca_sh = any(k in ca for k in ship_kw)
            cb_sh = any(k in cb for k in ship_kw)
            if (ca_ac and cb_sh) or (ca_sh and cb_ac):
                return True
    
    return False

# ============================================================
# Near-Perfect Signals (GD §4.2)
# ============================================================

def check_near_perfect(props_a, props_b, entity_type):
    """Check for near-perfect signals that trigger immediate merge."""
    # Equipment: exact name match with same equip_type
    if entity_type == 'equipment':
        na = _get(props_a, 'name')
        nb = _get(props_b, 'name')
        ca = _get(props_a, 'equip_type')
        cb = _get(props_b, 'equip_type')
        if na and nb and na == nb and ca and cb and ca == cb:
            return True
    
    return False

# ============================================================
# Core Scoring Function (GD §4.1.1)
# ============================================================

def l1_score(props_a, props_b, entity_type, 
             penalty_factor=1.5,
             merge_threshold=0.5,
             reject_threshold=-0.3,
             enable_hard_conflict=True,
             enable_near_perfect=True):
    """
    Compute Layer 1 bidirectional score.
    
    Returns: (score, decision, details)
      decision: 'merge' | 'reject' | 'escalate'
    """
    if props_a is None or props_b is None:
        return (0.0, 'escalate', 'missing properties')
    
    signals = SIGNAL_MAP.get(entity_type, [])
    if not signals:
        return (0.0, 'escalate', f'no signals for type {entity_type}')
    
    # Near-perfect signal check
    if enable_near_perfect and check_near_perfect(props_a, props_b, entity_type):
        return (1.0, 'merge', 'near-perfect signal')
    
    # Hard conflict check
    if enable_hard_conflict and has_hard_conflict(props_a, props_b, entity_type):
        return (-1.0, 'reject', 'hard conflict')
    
    # Score accumulation
    score = 0.0
    details = []
    match_count = 0
    conflict_count = 0
    unknown_count = 0
    
    for signal in signals:
        result = signal.evaluator(props_a, props_b)
        if result.result == 'match':
            score += signal.weight
            match_count += 1
        elif result.result == 'conflict':
            score -= signal.weight * penalty_factor
            conflict_count += 1
        else:
            unknown_count += 1
        details.append(f'{signal.name}={result.result}({result.detail})')
    
    # Clamp
    score = max(-1.0, min(1.0, score))
    
    # Routing
    if score >= merge_threshold:
        decision = 'merge'
    elif score <= reject_threshold:
        decision = 'reject'
    else:
        decision = 'escalate'
    
    detail_str = f'matches={match_count} conflicts={conflict_count} unknowns={unknown_count}; ' + ', '.join(details)
    
    return (score, decision, detail_str)


# ============================================================
# Wrapper: also check name match as implicit signal
# ============================================================

def l1_score_with_name(name_a, name_b, props_a, props_b, entity_type, **kwargs):
    """Score including a name-match signal as bonus."""
    score, decision, details = l1_score(props_a, props_b, entity_type, **kwargs)
    
    # If decision is already determined by hard conflict / near-perfect, keep it
    if decision in ('merge', 'reject') and abs(score) >= 0.9:
        return score, decision, details
    
    # Add name match bonus
    if name_a and name_b:
        na = name_a.lower().strip()
        nb = name_b.lower().strip()
        if na == nb:
            # Exact name match: strong signal, add 0.3
            score = max(-1.0, min(1.0, score + 0.3))
            details += f'; name_exact_match(+0.3)'
        elif _is_containment(na, nb):
            score = max(-1.0, min(1.0, score + 0.2))
            details += f'; name_containment(+0.2)'
        else:
            sim = _name_similarity(na, nb)
            if sim >= 0.5:
                score = max(-1.0, min(1.0, score + 0.15))
                details += f'; name_similar(+0.15, sim={sim:.2f})'
    
    # Re-route after name adjustment
    merge_threshold = kwargs.get('merge_threshold', 0.5)
    reject_threshold = kwargs.get('reject_threshold', -0.3)
    if score >= merge_threshold:
        decision = 'merge'
    elif score <= reject_threshold:
        decision = 'reject'
    else:
        decision = 'escalate'
    
    return score, decision, details


if __name__ == '__main__':
    # Quick self-test
    props1 = {'name': 'Capt. Paul Lorence', 'org_name': 'U.S. Air Force', 'created_at': '2026-05-07'}
    props2 = {'name': 'Paul Lorence', 'org_name': '', 'created_at': '2026-05-07'}
    score, dec, det = l1_score_with_name('Capt. Paul Lorence', 'Paul Lorence', props1, props2, 'person')
    print(f"Test: Capt. Paul Lorence vs Paul Lorence")
    print(f"  score={score:.3f}, decision={dec}")
    print(f"  details: {det}")
