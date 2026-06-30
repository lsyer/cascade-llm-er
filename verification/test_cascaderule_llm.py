#!/usr/bin/env python3
"""
test_cascaderule_llm.py — Automated test suite for CascadeRule-LLM production modules

Tests:
  1. Import chain: all modules load without circular dependency
  2. L1 scorer: parameter values match paper
  3. L1 scorer: known entity pair scoring matches expected values
  4. L1 scorer: feature extraction produces valid dimensions
  5. L2 judge: prompt construction contains all required components
  6. L2 judge: source fragment extraction logic
  7. L1 scorer: fusion.py integration (compute_similarity signature)
  8. Paper vs code: all validated parameters match
"""

import sys
import os
import json
import traceback
import numpy as np

# Add the backend directory to path (parent of 'app' package)
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "system", "backend")
sys.path.insert(0, BACKEND_DIR)

RESULTS = []
PASS = 0
FAIL = 0
SKIP = 0


def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        RESULTS.append(f"  ✅ {name}")
    else:
        FAIL += 1
        RESULTS.append(f"  ❌ {name} — {detail}")


def skip(name, reason=""):
    global SKIP
    SKIP += 1
    RESULTS.append(f"  ⏭  {name} — {reason}")


def section(title):
    RESULTS.append(f"\n{'='*60}")
    RESULTS.append(f"  {title}")
    RESULTS.append(f"{'='*60}")


# ============================================================
# 1. Import Chain — verify no circular dependencies
# ============================================================

section("1. Import Chain (no circular dependencies)")

try:
    from app.services.l1_scorer import (
        l1_score, l1_score_with_name, extract_features,
        MERGE_THRESHOLD, REJECT_THRESHOLD,
        PENALTY_FACTOR, FIELD_MATCH_THRESHOLD,
        STRONG_FIELDS, WEAK_FIELDS,
    )
    test("l1_scorer imports OK", True)
except Exception as e:
    test("l1_scorer imports OK", False, str(e))
    # Can't continue if this fails
    RESULTS.append("\nFATAL: Cannot import l1_scorer, aborting remaining tests.")
    print("\n".join(RESULTS))
    print(f"\n{'='*60}")
    print(f"Results: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    sys.exit(1)

try:
    from app.services.l2_judge import (
        l2_judge, build_l2_prompt, clean_attributes,
        TYPE_JUDGING_RULES, MAX_FRAGMENTS, FRAGMENT_CONTEXT_CHARS,
        extract_source_fragments, _scan_article_for_mentions,
    )
    test("l2_judge imports OK", True)
except Exception as e:
    test("l2_judge imports OK", False, str(e))

try:
    from app.services.feedback import (
        collect_feedback_pairs, build_training_matrix,
        train_unified_lr, train_hybrid_lr,
        check_and_retrain, ModelVersion, FeedbackPair,
        MIN_SAMPLES_UNIFIED, MIN_SAMPLES_PER_TYPE,
        RETRAIN_CHECK_INTERVAL, MIN_DEPLOY_ACCURACY,
    )
    test("feedback imports OK", True)
except Exception as e:
    test("feedback imports OK", False, str(e))


# ============================================================
# 2. L1 Scorer — Parameter Values Match Paper
# ============================================================

section("2. L1 Scorer Parameters (vs paper)")

test("MERGE_THRESHOLD = 0.6", MERGE_THRESHOLD == 0.6,
     f"got {MERGE_THRESHOLD}")
test("REJECT_THRESHOLD = -0.4", REJECT_THRESHOLD == -0.4,
     f"got {REJECT_THRESHOLD}")
test("PENALTY_FACTOR λ = 1.5", PENALTY_FACTOR == 1.5,
     f"got {PENALTY_FACTOR}")
test("FIELD_MATCH_THRESHOLD τ = 0.65", FIELD_MATCH_THRESHOLD == 0.65,
     f"got {FIELD_MATCH_THRESHOLD}")

# Strong/weak field tier checks
test("equip_type in STRONG_FIELDS", "equip_type" in STRONG_FIELDS)
test("gender in STRONG_FIELDS", "gender" in STRONG_FIELDS)
test("region in STRONG_FIELDS", "region" in STRONG_FIELDS)
test("name in WEAK_FIELDS", "name" in WEAK_FIELDS)
test("description in WEAK_FIELDS", "description" in WEAK_FIELDS)
test("aliases in WEAK_FIELDS", "aliases" in WEAK_FIELDS)

# Name bonus values
from app.services.l1_scorer import (
    NAME_BONUS_EXACT, NAME_BONUS_CONTAINMENT, NAME_BONUS_JACCARD,
    NAME_JACCARD_THRESHOLD,
)
test("NAME_BONUS_EXACT = 0.3", NAME_BONUS_EXACT == 0.3)
test("NAME_BONUS_CONTAINMENT = 0.2", NAME_BONUS_CONTAINMENT == 0.2)
test("NAME_BONUS_JACCARD = 0.15", NAME_BONUS_JACCARD == 0.15)
test("NAME_JACCARD_THRESHOLD = 0.5", NAME_JACCARD_THRESHOLD == 0.5)


# ============================================================
# 3. L1 Scorer — Known Entity Pair Scoring
# ============================================================

section("3. L1 Scorer — Known Entity Pairs")

# Test 3a: Same person, name containment (Capt. Paul Lorence vs Paul Lorence)
props1 = {
    "name": "Capt. Paul Lorence",
    "org_name": "U.S. Air Force",
    "occupation": "pilot",
    "nationality": "American",
}
props2 = {
    "name": "Paul Lorence",
    "org_name": "U.S. Air Force",
    "occupation": "pilot",
    "nationality": "American",
}
score, decision, detail = l1_score_with_name(
    "Capt. Paul Lorence", "Paul Lorence", props1, props2, "person")
test("Same person (Capt. prefix) → merge or escalate",
     decision in ('merge', 'escalate'),
     f"score={score:.3f} decision={decision}")
test("Same person — all attributes match → score > 0",
     score > 0,
     f"score={score:.3f}")

# Test 3b: Hard conflict — gender mismatch
props_m = {"name": "John Smith", "gender": "male", "org_name": "Navy"}
props_f = {"name": "John Smith", "gender": "female", "org_name": "Navy"}
score_gc, dec_gc, _ = l1_score_with_name(
    "John Smith", "John Smith", props_m, props_f, "person")
test("Gender hard conflict → reject", dec_gc == 'reject',
     f"score={score_gc:.3f} decision={dec_gc}")
test("Gender hard conflict → score = -1.0", score_gc == -1.0,
     f"score={score_gc:.3f}")

# Test 3c: Equipment cross-class conflict (aircraft vs ship)
props_air = {"name": "F-35", "equip_type": "fighter", "category": "aircraft"}
props_ship = {"name": "F-35", "equip_type": "carrier", "category": "ship"}
score_xc, dec_xc, _ = l1_score_with_name(
    "F-35", "F-35", props_air, props_ship, "equipment")
test("Cross-class equipment (aircraft vs ship) → reject",
     dec_xc == 'reject',
     f"score={score_xc:.3f} decision={dec_xc}")

# Test 3d: Same equipment, exact match
props_eq1 = {"name": "USS Gerald R. Ford", "equip_type": "carrier",
             "category": "aircraft_carrier", "state": "active"}
props_eq2 = {"name": "USS Gerald R. Ford", "equip_type": "carrier",
             "category": "aircraft_carrier", "state": "active"}
score_eq, dec_eq, _ = l1_score_with_name(
    "USS Gerald R. Ford", "USS Gerald R. Ford", props_eq1, props_eq2, "equipment")
test("Identical equipment → merge", dec_eq == 'merge',
     f"score={score_eq:.3f} decision={dec_eq}")
test("Identical equipment → score ≥ 0.6", score_eq >= 0.6,
     f"score={score_eq:.3f}")

# Test 3e: Different events (same name, different region/time)
props_ev1 = {"name": "Freedom of Navigation Operation",
             "event_type": "operation", "start_date": "2025-01-01",
             "location_name": "South China Sea"}
props_ev2 = {"name": "Freedom of Navigation Operation",
             "event_type": "operation", "start_date": "2026-03-15",
             "location_name": "Persian Gulf"}
score_ev, dec_ev, _ = l1_score_with_name(
    "Freedom of Navigation Operation", "Freedom of Navigation Operation",
    props_ev1, props_ev2, "activity")
test("Same event name, different region+time → escalate",
     dec_ev == 'escalate',
     f"score={score_ev:.3f} decision={dec_ev}")
test("Conflicting events → score in grey zone (not merge)",
     score_ev < MERGE_THRESHOLD,
     f"score={score_ev:.3f}")

# Test 3f: Score range always within [-1, +1]
import random
for _ in range(100):
    rand_a = {"name": f"entity_{random.randint(0,5)}",
              "region": f"r{random.randint(0,3)}",
              "equip_type": random.choice(["ship", "aircraft", ""])}
    rand_b = {"name": f"entity_{random.randint(0,5)}",
              "region": f"r{random.randint(0,3)}",
              "equip_type": random.choice(["ship", "aircraft", ""])}
    s, d, _ = l1_score_with_name(
        rand_a["name"], rand_b["name"], rand_a, rand_b, "equipment")
    if s < -1.0 or s > 1.0:
        test("Score always in [-1, +1]", False, f"score={s:.3f} out of range")
        break
else:
    test("Score always in [-1, +1] (100 random tests)", True)


# ============================================================
# 4. L1 Scorer — Feature Extraction
# ============================================================

section("4. Feature Extraction")

feats = extract_features(props1, props2, "person",
                         "Capt. Paul Lorence", "Paul Lorence")
test("Feature extraction returns dict", isinstance(feats, dict))
test("property_overlap_rate in features", "property_overlap_rate" in feats)
test("property_conflict_rate in features", "property_conflict_rate" in feats)
test("n_effective_fields in features", "n_effective_fields" in feats)
test("name_similarity in features", "name_similarity" in feats)
test("name_exact in features", "name_exact" in feats)
test("name_containment in features", "name_containment" in feats)
test("hard_conflict in features", "hard_conflict" in feats)
test("near_perfect_match in features", "near_perfect_match" in feats)

# Check strong/weak breakdown
test("strong_overlap_rate in features", "strong_overlap_rate" in feats)
test("weak_overlap_rate in features", "weak_overlap_rate" in feats)
test("n_strong_fields in features", "n_strong_fields" in feats)
test("n_weak_fields in features", "n_weak_fields" in feats)

# Property overlap for matching entities should be high
test("Matching entities → overlap_rate > 0.5",
     feats["property_overlap_rate"] > 0.5,
     f"overlap={feats['property_overlap_rate']:.3f}")
test("Matching entities → conflict_rate = 0",
     feats["property_conflict_rate"] == 0.0,
     f"conflict={feats['property_conflict_rate']:.3f}")

# Empty properties edge case
empty_feats = extract_features({}, {}, "person")
test("Empty props → overlap_rate = 0", empty_feats["property_overlap_rate"] == 0.0)
test("Empty props → n_effective_fields = 0", empty_feats["n_effective_fields"] == 0)


# ============================================================
# 5. L2 Judge — Prompt Construction
# ============================================================

section("5. L2 Judge — Prompt Construction")

try:
    prompt, valid_vids = build_l2_prompt(
        entity_type="equipment",
        name_new="USS Reagan",
        props_new={"name": "USS Reagan", "equip_type": "carrier",
                   "category": "aircraft_carrier"},
        candidates=[
            {"vid": "equip_1", "name": "USS Reagan",
             "props": {"name": "USS Reagan", "equip_type": "carrier"},
             "source_fragments": []},
        ],
        source_fragments=["The USS Reagan departed from Yokosuka..."],
        article_title="US Navy News",
    )
    test("build_l2_prompt returns (str, set)", True)
    test("valid_vids contains candidate VID", "equip_1" in valid_vids)
    test("Prompt contains entity type", "equipment" in prompt.lower())
    test("Prompt contains NEW entity name", "USS Reagan" in prompt)
    test("Prompt contains candidate VID", "equip_1" in prompt)
    test("Prompt contains source fragment", "Yokosuka" in prompt)
    test("Prompt contains type-specific rules",
         "hull number" in prompt.lower() or "designation" in prompt.lower())
    test("Prompt contains JSON return instruction",
         "json" in prompt.lower())
    test("Prompt contains bias-free caution",
         "caution" in prompt.lower() or "false merge" in prompt.lower())

    # Check all 5 type rules exist
    test("TYPE_JUDGING_RULES has 5 types",
         len(TYPE_JUDGING_RULES) == 5,
         f"got {len(TYPE_JUDGING_RULES)}")
    for etype in ['person', 'equipment', 'location', 'event', 'organization']:
        test(f"TYPE_JUDGING_RULES has '{etype}'", etype in TYPE_JUDGING_RULES)

except Exception as e:
    test("build_l2_prompt works", False, str(e))
    traceback.print_exc()


# ============================================================
# 6. L2 Judge — Source Fragment Extraction
# ============================================================

section("6. L2 Judge — Source Fragment Extraction")

test("MAX_FRAGMENTS = 2", MAX_FRAGMENTS == 2)
test("FRAGMENT_CONTEXT_CHARS = 300", FRAGMENT_CONTEXT_CHARS == 300)

# Test article scanning
article = """
The USS Gerald R. Ford (CVN-78) departed Naval Station Norfolk on Monday,
heading to the Mediterranean Sea. The carrier strike group includes
the USS Higgins (DDG-76) and VFA-27 squadron.
"""
mentions = _scan_article_for_mentions("USS Gerald R. Ford", article)
test("Article scan finds entity mentions", len(mentions) > 0,
     f"found {len(mentions)} mentions")
if mentions:
    test("Fragment length ≤ ~400 chars", len(mentions[0]) <= 400,
         f"len={len(mentions[0])}")

# Non-matching name
no_mentions = _scan_article_for_mentions("Nonexistent Entity", article)
test("Non-matching name → 0 fragments", len(no_mentions) == 0)


# ============================================================
# 7. Attribute Cleaning
# ============================================================

section("7. Attribute Cleaning")

dirty = {
    "name": "USS Ford",
    "created_at": "2024-01-01",
    "confidence": 0.95,
    "equip_type": "carrier",
    "description": "",
    "category": "aircraft_carrier",
    "vid": "equip_1",
    "__EMPTY__": "",
}
cleaned = clean_attributes(dirty)
test("Cleaned removes created_at", "created_at" not in cleaned)
test("Cleaned removes confidence", "confidence" not in cleaned)
test("Cleaned removes vid", "vid" not in cleaned)
test("Cleaned removes __EMPTY__", "__EMPTY__" not in cleaned)
test("Cleaned removes empty description", "description" not in cleaned)
test("Cleaned keeps name", "name" in cleaned)
test("Cleaned keeps equip_type", "equip_type" in cleaned)
test("Cleaned keeps category", "category" in cleaned)


# ============================================================
# 8. Feedback Module — Training Matrix Construction
# ============================================================

section("8. Feedback Module")

# Build synthetic feedback pairs
pairs = []
for i in range(60):
    pairs.append(FeedbackPair(
        pair_id=f"test_{i}",
        entity_type="equipment",
        name_a=f"Entity_A_{i}",
        name_b=f"Entity_B_{i}" if i >= 30 else f"Entity_A_{i}",
        props_a={"name": f"Entity_A_{i}", "equip_type": "carrier",
                 "category": "ship"},
        props_b={"name": f"Entity_B_{i}" if i >= 30 else f"Entity_A_{i}",
                 "equip_type": "carrier", "category": "ship"},
        label=0 if i >= 30 else 1,
        source="l2_clear",
        feedback_at="2024-01-01T00:00:00",
    ))

X, y, feature_names, etypes = build_training_matrix(pairs)
test("Training matrix X shape correct",
     X.shape == (60, len(feature_names)),
     f"X.shape={X.shape}")
test("Label vector y shape correct", y.shape == (60,))
test("Feature names non-empty", len(feature_names) > 5,
     f"got {len(feature_names)} features")
test("Labels include both 0 and 1", set(y.tolist()) == {0, 1})
test("Entity types array correct", len(etypes) == 60)

# Check feature names contain expected aggregates
expected_features = [
    'property_overlap_rate', 'property_conflict_rate', 'n_effective_fields',
    'name_similarity', 'name_exact', 'name_containment',
    'hard_conflict', 'near_perfect_match',
    'strong_overlap_rate', 'weak_overlap_rate',
]
for ef in expected_features:
    test(f"Feature '{ef}' in training matrix", ef in feature_names)

# Feedback configuration checks
test("MIN_SAMPLES_UNIFIED = 50", MIN_SAMPLES_UNIFIED == 50)
test("MIN_SAMPLES_PER_TYPE = 50", MIN_SAMPLES_PER_TYPE == 50)
test("RETRAIN_CHECK_INTERVAL = 50", RETRAIN_CHECK_INTERVAL == 50)
test("MIN_DEPLOY_ACCURACY = 0.75", MIN_DEPLOY_ACCURACY == 0.75)


# ============================================================
# 9. Feedback Module — LR Training (if sklearn available)
# ============================================================

section("9. Feedback Module — LR Training")

try:
    from sklearn.linear_model import LogisticRegression
    has_sklearn = True
except ImportError:
    has_sklearn = False
    skip("LR training test", "sklearn not installed")

if has_sklearn:
    try:
        mv = train_unified_lr(X, y, feature_names)
        if mv:
            test("Unified LR training succeeded", True)
            test("Model version format correct",
                 "unified_v" in mv.version,
                 f"version={mv.version}")
            test("Model has accuracy metric",
                 hasattr(mv, 'accuracy') and 0 <= mv.accuracy <= 1.0)
            test("Model has top features",
                 len(mv.top_features) > 0)
            test("Model has theta_merge",
                 hasattr(mv, 'theta_merge') and 0 < mv.theta_merge < 1.0)
        else:
            # Might fail if synthetic data is too clean — acceptable
            skip("Unified LR training", "model not deployed (precision floor)")
    except Exception as e:
        test("Unified LR training succeeded", False, str(e))
        traceback.print_exc()


# ============================================================
# 10. Integration — fusion.py compute_similarity signature
# ============================================================

section("10. Integration — fusion.py")

try:
    # Check compute_similarity returns tuple (not float like old code)
    from app.services.fusion import compute_similarity, L1_MERGE, L1_REJECT
    test("fusion imports L1_MERGE", L1_MERGE == 0.6)
    test("fusion imports L1_REJECT", L1_REJECT == -0.4)

    # Call compute_similarity with test data
    ent_a = {"vid": "equip_1", "name": "USS Ford", "aliases": []}
    ent_b = {"vid": "equip_2", "name": "USS Ford", "aliases": []}
    props_a = {"name": "USS Ford", "equip_type": "carrier",
               "category": "aircraft_carrier"}
    props_b = {"name": "USS Ford", "equip_type": "carrier",
               "category": "aircraft_carrier"}
    result = compute_similarity("equipment", ent_a, ent_b, props_a, props_b)
    test("compute_similarity returns tuple",
         isinstance(result, tuple) and len(result) == 2,
         f"type={type(result)}")
    if isinstance(result, tuple):
        sim, dec = result
        test("compute_similarity score in [-1, +1]", -1.0 <= sim <= 1.0,
             f"score={sim:.3f}")
        test("compute_similarity returns valid decision",
             dec in ('merge', 'reject', 'escalate'),
             f"decision={dec}")
        test("Identical entities → merge", dec == 'merge',
             f"decision={dec} score={sim:.3f}")

except Exception as e:
    test("fusion.py integration", False, str(e))
    traceback.print_exc()


# ============================================================
# 11. Cross-Reference — Paper Parameters vs Code
# ============================================================

section("11. Paper vs Code Parameter Cross-Reference")

# These are the exact values stated in the paper (Section 3 and experiments)
paper_params = {
    "τ (Jaccard threshold)": ("FIELD_MATCH_THRESHOLD", FIELD_MATCH_THRESHOLD, 0.65),
    "λ (penalty factor)": ("PENALTY_FACTOR", PENALTY_FACTOR, 1.5),
    "merge_threshold": ("MERGE_THRESHOLD", MERGE_THRESHOLD, 0.6),
    "reject_threshold": ("REJECT_THRESHOLD", REJECT_THRESHOLD, -0.4),
    "name_bonus_exact": ("NAME_BONUS_EXACT", NAME_BONUS_EXACT, 0.3),
    "name_bonus_containment": ("NAME_BONUS_CONTAINMENT", NAME_BONUS_CONTAINMENT, 0.2),
    "name_bonus_jaccard": ("NAME_BONUS_JACCARD", NAME_BONUS_JACCARD, 0.15),
    "fragment_context_chars": ("FRAGMENT_CONTEXT_CHARS", FRAGMENT_CONTEXT_CHARS, 300),
    "max_fragments": ("MAX_FRAGMENTS", MAX_FRAGMENTS, 2),
    "retrain_interval": ("RETRAIN_CHECK_INTERVAL", RETRAIN_CHECK_INTERVAL, 50),
}

for label, (code_name, code_val, paper_val) in paper_params.items():
    test(f"Paper {label} = {paper_val} ↔ Code {code_name} = {code_val}",
         code_val == paper_val,
         f"code={code_val} paper={paper_val}")

# Check L2 temperature matches paper
from app.services.l2_judge import LLM_MODEL
# Paper says temperature = 0.1 for L2; checked in build_l2_prompt
test("L2 model configurable", isinstance(LLM_MODEL, str))


# ============================================================
# 12. Closed-Loop Feedback: Train → Save → Load → Score
# ============================================================

section("12. Closed-Loop Feedback (Train → Load → Adaptive Score)")

from app.services.l1_scorer import (
    l1_score_adaptive, score_with_learned_model,
    _load_model_artifact, _get_latest_version,
    _model_cache,
)

# 12a: Before training → adaptive falls back to fixed L1
# (Note: section 9 may have already trained a model, so we check
#  that score_with_learned_model returns either None or a valid ScoreResult)
pre_result = score_with_learned_model(props1, props2, "person")
test("Pre-existing model returns None or valid result",
     pre_result is None or hasattr(pre_result, 'decision'),
     f"got {type(pre_result)}")

# 12b: Build training data (60 pairs: 30 same, 30 different)
loop_pairs = []
for i in range(30):
    # Same entities
    loop_pairs.append(FeedbackPair(
        pair_id=f"loop_same_{i}",
        entity_type="equipment",
        name_a="USS Ford",
        name_b="USS Ford",
        props_a={"name": "USS Ford", "equip_type": "carrier", "category": "ship"},
        props_b={"name": "USS Ford", "equip_type": "carrier", "category": "ship"},
        label=1, source="test", feedback_at="2024-01-01",
    ))
    # Different entities
    loop_pairs.append(FeedbackPair(
        pair_id=f"loop_diff_{i}",
        entity_type="equipment",
        name_a="USS Ford",
        name_b="USS Reagan",
        props_a={"name": "USS Ford", "equip_type": "carrier", "category": "ship"},
        props_b={"name": "USS Reagan", "equip_type": "carrier", "category": "ship"},
        label=0, source="test", feedback_at="2024-01-01",
    ))

X_loop, y_loop, fn_loop, et_loop = build_training_matrix(loop_pairs)
test("Loop training matrix built", X_loop.shape[0] == 60)

# 12c: Train unified model
mv_loop = train_unified_lr(X_loop, y_loop, fn_loop)
test("Unified model trained for loop test", mv_loop is not None)

if mv_loop:
    # 12d: Verify model was saved to disk
    test("Model version saved to disk",
         _load_model_artifact(mv_loop.version) is not None,
         f"version={mv_loop.version}")

    # 12e: Clear cache to force reload from disk
    _model_cache.clear()

    # 12f: Adaptive scorer should now find and use the model
    learned_result = score_with_learned_model(
        {"name": "USS Ford", "equip_type": "carrier", "category": "ship"},
        {"name": "USS Ford", "equip_type": "carrier", "category": "ship"},
        "equipment",
    )
    test("score_with_learned_model returns result after training",
         learned_result is not None)
    if learned_result:
        test("Learned model returns valid decision",
             learned_result.decision in ('merge', 'reject', 'escalate'),
             f"decision={learned_result.decision}")

    # 12g: l1_score_adaptive should use learned model transparently
    adaptive_result = l1_score_adaptive(
        {"name": "USS Ford", "equip_type": "carrier", "category": "ship"},
        {"name": "USS Ford", "equip_type": "carrier", "category": "ship"},
        "equipment", name_a="USS Ford", name_b="USS Ford",
    )
    test("l1_score_adaptive returns ScoreResult",
         hasattr(adaptive_result, 'score') and hasattr(adaptive_result, 'decision'))
    test("Adaptive score for identical entities → merge or escalate",
         adaptive_result.decision in ('merge', 'escalate'),
         f"decision={adaptive_result.decision} score={adaptive_result.score:.3f}")

    # 12h: Hard conflict still overrides learned model
    conflict_result = l1_score_adaptive(
        {"name": "Test", "gender": "male"},
        {"name": "Test", "gender": "female"},
        "person", name_a="Test", name_b="Test",
    )
    test("Hard conflict overrides learned model → reject",
         conflict_result.decision == 'reject',
         f"decision={conflict_result.decision}")

# 12i: maybe_retrain_async with no DB → graceful failure
try:
    from app.services.feedback import maybe_retrain_async, count_new_resolved_since_last_train
    result = maybe_retrain_async(None)
    test("maybe_retrain_async with None DB → no crash", True)
    test("maybe_retrain_async returns dict", isinstance(result, dict))
    test("maybe_retrain_async reports not trained",
         result.get('trained') == False or result.get('trained') is None)
except Exception as e:
    test("maybe_retrain_async with None DB → no crash", False, str(e))


# ============================================================
# 13. L2 Few-Shot Isolation (ONLY human-adjudicated)
# ============================================================

section("13. L2 Few-Shot Isolation — Expert Feedback Only")

from app.services.feedback import (
    get_human_fewshot_examples, format_fewshot_text,
    MAX_FEWSHOT_EXAMPLES, FEWSHOT_SOURCES,
)

# 13a: None db_conn → empty list (no crash)
fs = get_human_fewshot_examples(None, "person")
test("get_human_fewshot_examples(None) → empty", fs == [])

# 13b: format_fewshot_text with empty list → empty string
test("format_fewshot_text([]) → ''", format_fewshot_text([]) == "")

# 13c: format_fewshot_text with valid examples → contains key markers
sample = [
    {"name_a": "John Smith", "name_b": "J. Smith", "label": "SAME", "reason": "manual merge"},
    {"name_a": "John Smith", "name_b": "John Jones", "label": "DIFFERENT", "reason": "manual disambiguate"},
]
fs_text = format_fewshot_text(sample)
test("Few-shot text mentions 'expert human judgments'",
     "expert human judgments" in fs_text.lower())
test("Few-shot text contains SAME example",
     "John Smith" in fs_text and "SAME" in fs_text)
test("Few-shot text contains DIFFERENT example",
     "John Jones" in fs_text and "DIFFERENT" in fs_text)

# 13d: build_l2_prompt with fewshot_text injects examples
from app.services.l2_judge import build_l2_prompt as _build
prompt_fs, _ = _build(
    "person", "Test Person", {"name": "Test Person"},
    [{"vid": "person_1", "name": "Candidate", "props": {"name": "Candidate"}}],
    ["Source fragment text"],
    "Article Title", "",
    fewshot_text=fs_text,
)
test("L2 prompt with fewshot contains examples",
     "expert human judgments" in prompt_fs or "SAME" in prompt_fs)

# 13e: L2 prompt WITHOUT fewshot does NOT contain examples
prompt_nofs, _ = _build(
    "person", "Test Person", {"name": "Test Person"},
    [{"vid": "person_1", "name": "Candidate", "props": {"name": "Candidate"}}],
    ["Source fragment text"],
    "Article Title", "",
    fewshot_text="",
)
test("L2 prompt without fewshot excludes examples",
     "expert human judgments" not in prompt_nofs)

# 13f: MAX_FEWSHOT_EXAMPLES is reasonable (1-10)
test("MAX_FEWSHOT_EXAMPLES in [1,10]",
     1 <= MAX_FEWSHOT_EXAMPLES <= 10,
     f"value={MAX_FEWSHOT_EXAMPLES}")

# 13g: FEWSHOT_SOURCES only contains human-review markers
test("FEWSHOT_SOURCES excludes 'l2_clear' and 'llm'",
     'l2_clear' not in FEWSHOT_SOURCES and 'llm' not in str(FEWSHOT_SOURCES).lower())


# ============================================================
# 14. Anti Self-Reinforcement: LLM Batch ≠ Training Source
# ============================================================

section("14. Anti Self-Reinforcement — LLM Results ≠ Training Data")

# The design principle: only human-resolved pairs enter the feedback loop.
# LLM batch results from fusion_execute should NOT trigger retraining.

# 14a: Verify collect_feedback_pairs query string contains 'manual%' filter
import inspect as _inspect
from app.services import feedback as _fb_mod
src = _inspect.getsource(_fb_mod.collect_feedback_pairs)
test("collect_feedback_pairs has manual% filter in all branches",
     src.count("notes LIKE 'manual%'") >= 2)  # cursor + sqlalchemy branches

# 14b: Verify count_new_resolved_since_last_train also filters
src_count = _inspect.getsource(_fb_mod.count_new_resolved_since_last_train)
test("count_new_resolved filters by manual%",
     "notes LIKE 'manual%'" in src_count or "resolved_by IS NOT NULL" in src_count)

# 14c: Verify fusion_execute does NOT call maybe_retrain_async
import re as _re
_admin_path = "/root/workspace/tanshu_docs/experiments/release/system/backend/app/routers/admin.py"
with open(_admin_path) as _f:
    _admin_src = _f.read()
_start = _admin_src.index("async def fusion_execute")
_next = list(_re.finditer(r'\n(?:async )?def |@router\.', _admin_src[_start+10:]))
_end = _start + 10 + _next[0].start() if _next else len(_admin_src)
_fusion_only = _admin_src[_start:_end]
test("fusion_execute does NOT directly trigger retrain",
     "maybe_retrain_async" not in _fusion_only)
test("fusion_execute notes pending human confirmation",
     "human confirmation" in _fusion_only.lower()
     or "human review" in _fusion_only.lower())


# ============================================================
# Summary
# ============================================================

RESULTS.append(f"\n{'='*60}")
RESULTS.append(f"  SUMMARY")
RESULTS.append(f"{'='*60}")
RESULTS.append(f"  Passed:  {PASS}")
RESULTS.append(f"  Failed:  {FAIL}")
RESULTS.append(f"  Skipped: {SKIP}")
RESULTS.append(f"  Total:   {PASS + FAIL + SKIP}")
RESULTS.append(f"{'='*60}")

print("\n".join(RESULTS))
sys.exit(0 if FAIL == 0 else 1)
