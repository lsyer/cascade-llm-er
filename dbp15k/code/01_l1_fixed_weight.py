#!/usr/bin/env python3
"""
DBP15K (ZH-EN) Experiment: Apply CascadeRule-LLM L1 scorer to cross-lingual KG entity alignment.

Dataset: DBP15K ZH-EN from JAPE (ISWC 2017)
- 15,000 gold-standard entity alignment pairs (Chinese DBpedia ↔ English DBpedia)
- Each entity has: label, translated label, attribute set
- Relational triples + attribute triples available

Experiment design:
1. Load all 15K gold pairs → positive examples
2. Sample negatives (non-aligned pairs with similar names) → negative examples
3. Apply L1 signal scoring (adapted for cross-lingual: use translated labels)
4. Evaluate L1 accuracy, interception rate
5. Sample L1-escalated pairs → send to GLM-5.2 for L2 judgment
6. Report pipeline accuracy

Output: results saved to dbp15k_results.json
"""

import json, os, sys, random, re, time
from collections import defaultdict
from pathlib import Path

# ─── Paths ───
DATA_DIR = Path(__file__).parent / "JAPE" / "data" / "dbp15k" / "zh_en"
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── 1. Load DBP15K ───

def load_labels(filepath):
    """Load entity labels: URI → (label, translated_label)"""
    labels = {}
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                uri, label, trans = parts[0], parts[1], parts[2]
                labels[uri] = {'label': label, 'trans': trans}
            elif len(parts) == 2:
                uri, label = parts[0], parts[1]
                labels[uri] = {'label': label, 'trans': label}
    return labels

def load_ILLs(filepath):
    """Load inter-lingual links (gold alignment): list of (zh_uri, en_uri)"""
    pairs = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs

def load_attrs(filepath):
    """Load attribute sets: URI → set of attribute URIs"""
    attrs = {}
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                uri = parts[0]
                attr_set = set(parts[1:])
                attrs[uri] = attr_set
    return attrs

def load_rel_triples(filepath):
    """Load relational triples: URI → list of (relation, object_uri)"""
    triples = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                subj, rel, obj = parts
                triples[subj].append((rel, obj))
    return triples

print("Loading DBP15K ZH-EN...")
zh_labels = load_labels(DATA_DIR / "s_labels")
en_labels = load_labels(DATA_DIR / "t_labels")
gold_pairs = load_ILLs(DATA_DIR / "ent_ILLs")
zh_attrs = load_attrs(DATA_DIR / "training_attrs_1")
en_attrs = load_attrs(DATA_DIR / "training_attrs_2")
zh_triples = load_rel_triples(DATA_DIR / "s_triples")
en_triples = load_rel_triples(DATA_DIR / "t_triples")

print(f"  ZH entities with labels: {len(zh_labels)}")
print(f"  EN entities with labels: {len(en_labels)}")
print(f"  Gold alignment pairs:    {len(gold_pairs)}")
print(f"  ZH with attrs:           {len(zh_attrs)}")
print(f"  EN with attrs:           {len(en_attrs)}")
print(f"  ZH relational triples:   {sum(len(v) for v in zh_triples.values())}")
print(f"  EN relational triples:   {sum(len(v) for v in en_triples.values())}")

# ─── 2. Build evaluation set ───

random.seed(42)

# Positive pairs: sample from gold standard
pos_sample = random.sample(gold_pairs, min(2000, len(gold_pairs)))

# Negative pairs: entities with similar translated labels but NOT in gold
zh_uris = list(zh_labels.keys())
en_uris = list(en_labels.keys())
gold_set = set(gold_pairs)

# Build name-based blocking: match entities with similar translated names
def normalize_name(name):
    """Normalize entity name for comparison."""
    name = name.lower().strip()
    name = re.sub(r'[_\-\(\)]', ' ', name)
    name = re.sub(r'\s+', ' ', name)
    return name

# Index EN entities by first letter of normalized name for fast blocking
en_name_index = defaultdict(list)
for uri in en_uris:
    label = en_labels.get(uri, {}).get('label', '')
    norm = normalize_name(label)
    if norm:
        first_word = norm.split()[0] if norm.split() else norm[0]
        en_name_index[first_word[:3]].append(uri)  # first 3 chars of first word

neg_pairs = []
attempts = 0
while len(neg_pairs) < 2000 and attempts < 50000:
    attempts += 1
    zh_uri = random.choice(zh_uris)
    if zh_uri not in zh_labels:
        continue
    trans = zh_labels[zh_uri].get('trans', '')
    norm = normalize_name(trans)
    if not norm:
        continue
    # Find EN entities with similar name prefix but NOT aligned
    prefix = norm.split()[0][:3] if norm.split() else norm[:3]
    candidates = en_name_index.get(prefix, [])
    if not candidates:
        continue
    en_uri = random.choice(candidates)
    if (zh_uri, en_uri) not in gold_set and (en_uri, zh_uri) not in gold_set:
        neg_pairs.append((zh_uri, en_uri))

print(f"\nEvaluation set: {len(pos_sample)} positive + {len(neg_pairs)} negative = {len(pos_sample) + len(neg_pairs)} total")

# ─── 3. L1 Signal Scoring (adapted for cross-lingual KG) ───

def compute_l1_score(zh_uri, en_uri):
    """
    Compute L1 compatibility score for a ZH-EN entity pair.
    Signals adapted from CascadeRule-LLM L1 scorer:
    1. Name similarity (translated label vs EN label) — weight 0.40
    2. Attribute overlap (Jaccard on attribute property sets) — weight 0.35
    3. Relational degree similarity — weight 0.15
    4. URI similarity (string-level) — weight 0.10
    Returns score in [-1, +1]
    """
    signals = {}
    
    # Signal 1: Name similarity (0-1)
    zh_trans = zh_labels.get(zh_uri, {}).get('trans', '')
    en_label = en_labels.get(en_uri, {}).get('label', '')
    if zh_trans and en_label:
        n1 = normalize_name(zh_trans)
        n2 = normalize_name(en_label)
        # Token overlap ratio
        tokens1 = set(n1.split())
        tokens2 = set(n2.split())
        if tokens1 and tokens2:
            overlap = len(tokens1 & tokens2) / max(len(tokens1 | tokens2), 1)
            # Exact match bonus
            if n1 == n2:
                overlap = 1.0
            signals['name_similarity'] = overlap
        else:
            signals['name_similarity'] = 0.0
    else:
        signals['name_similarity'] = 0.0
    
    # Signal 2: Attribute overlap (Jaccard, 0-1)
    zh_a = zh_attrs.get(zh_uri, set())
    en_a = en_attrs.get(en_uri, set())
    if zh_a and en_a:
        # Normalize attribute URIs: strip namespace, compare local names
        zh_a_norm = set(a.split('/')[-1].lower() for a in zh_a)
        en_a_norm = set(a.split('/')[-1].lower() for a in en_a)
        jaccard = len(zh_a_norm & en_a_norm) / max(len(zh_a_norm | en_a_norm), 1)
        signals['attr_overlap'] = jaccard
    else:
        signals['attr_overlap'] = 0.0
    
    # Signal 3: Relational degree similarity (0-1)
    zh_deg = len(zh_triples.get(zh_uri, []))
    en_deg = len(en_triples.get(en_uri, []))
    if zh_deg > 0 and en_deg > 0:
        ratio = min(zh_deg, en_deg) / max(zh_deg, en_deg)
        signals['degree_similarity'] = ratio
    else:
        signals['degree_similarity'] = 0.0
    
    # Signal 4: URI local name similarity (0-1)
    zh_local = zh_uri.split('/')[-1].lower()
    en_local = en_uri.split('/')[-1].lower()
    zh_local = re.sub(r'%[0-9a-fA-F]{2}', '_', zh_local)  # URL-decode placeholder
    if zh_local and en_local:
        tokens1 = set(re.split(r'[_\-]', zh_local))
        tokens2 = set(re.split(r'[_\-]', en_local))
        if tokens1 and tokens2:
            signals['uri_similarity'] = len(tokens1 & tokens2) / max(len(tokens1 | tokens2), 1)
        else:
            signals['uri_similarity'] = 0.0
    else:
        signals['uri_similarity'] = 0.0
    
    # Weighted combination
    weights = {
        'name_similarity': 0.40,
        'attr_overlap': 0.35,
        'degree_similarity': 0.15,
        'uri_similarity': 0.10,
    }
    
    score = sum(signals[k] * weights[k] for k in weights)
    
    # Penalty for zero attribute overlap on high-degree entities
    if signals['attr_overlap'] == 0 and (zh_deg > 5 or en_deg > 5):
        score *= 0.7  # 30% penalty
    
    return score, signals

# ─── 4. Run L1 on all pairs ───

print("\nRunning L1 scoring...")
l1_results = []

for zh_uri, en_uri in pos_sample:
    score, signals = compute_l1_score(zh_uri, en_uri)
    l1_results.append({'zh_uri': zh_uri, 'en_uri': en_uri, 'label': 1, 'score': score, 'signals': signals})

for zh_uri, en_uri in neg_pairs:
    score, signals = compute_l1_score(zh_uri, en_uri)
    l1_results.append({'zh_uri': zh_uri, 'en_uri': en_uri, 'label': 0, 'score': score, 'signals': signals})

# ─── 5. Evaluate L1 ───

MERGE_THRESHOLD = 0.5
REJECT_THRESHOLD = 0.3  # pairs below this → confident reject

tp = fp = tn = fn = 0
escalated = 0
correct_intercept = 0

for r in l1_results:
    score = r['score']
    label = r['label']
    
    if score >= MERGE_THRESHOLD:
        predicted = 1
    elif score < REJECT_THRESHOLD:
        predicted = 0
    else:
        # Escalate to L2
        escalated += 1
        continue  # will be handled by L2
    
    if predicted == 1 and label == 1:
        tp += 1
        correct_intercept += 1
    elif predicted == 1 and label == 0:
        fp += 1
    elif predicted == 0 and label == 0:
        tn += 1
        correct_intercept += 1
    elif predicted == 0 and label == 1:
        fn += 1

intercepted = len(l1_results) - escalated
l1_accuracy = (tp + tn) / max(intercepted, 1)
interception_rate = intercepted / len(l1_results)

print(f"\n=== L1 Results (DBP15K ZH-EN) ===")
print(f"Total pairs:      {len(l1_results)}")
print(f"L1 intercepted:   {intercepted} ({interception_rate:.1%})")
print(f"Escalated to L2:  {escalated} ({1-interception_rate:.1%})")
print(f"L1 accuracy:      {l1_accuracy:.1%}")
print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
print(f"  Precision (merge): {tp/max(tp+fp,1):.1%}")
print(f"  Recall (merge):    {tp/max(tp+fn+escalated//2,1):.1%}")  # rough: half escalated are positive

# Signal statistics
print(f"\n=== Signal Statistics (positive pairs) ===")
pos_signals = defaultdict(list)
for r in l1_results:
    if r['label'] == 1:
        for k, v in r['signals'].items():
            pos_signals[k].append(v)

neg_signals = defaultdict(list)
for r in l1_results:
    if r['label'] == 0:
        for k, v in r['signals'].items():
            neg_signals[k].append(v)

for sig_name in ['name_similarity', 'attr_overlap', 'degree_similarity', 'uri_similarity']:
    pos_mean = sum(pos_signals[sig_name]) / max(len(pos_signals[sig_name]), 1)
    neg_mean = sum(neg_signals[sig_name]) / max(len(neg_signals[sig_name]), 1)
    print(f"  {sig_name:25s}  pos={pos_mean:.3f}  neg={neg_mean:.3f}  gap={pos_mean-neg_mean:+.3f}")

# ─── 6. Save results ───

results = {
    'dataset': 'DBP15K ZH-EN',
    'total_pairs': len(l1_results),
    'positive_pairs': len(pos_sample),
    'negative_pairs': len(neg_pairs),
    'l1': {
        'intercepted': intercepted,
        'escalated': escalated,
        'interception_rate': round(interception_rate, 4),
        'accuracy': round(l1_accuracy, 4),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'merge_precision': round(tp / max(tp + fp, 1), 4),
    },
    'signals': {
        sig_name: {
            'pos_mean': round(sum(pos_signals[sig_name]) / max(len(pos_signals[sig_name]), 1), 4),
            'neg_mean': round(sum(neg_signals[sig_name]) / max(len(neg_signals[sig_name]), 1), 4),
        }
        for sig_name in ['name_similarity', 'attr_overlap', 'degree_similarity', 'uri_similarity']
    },
    'weights': {'name_similarity': 0.40, 'attr_overlap': 0.35, 'degree_similarity': 0.15, 'uri_similarity': 0.10},
    'thresholds': {'merge': MERGE_THRESHOLD, 'reject': REJECT_THRESHOLD},
}

output_path = OUTPUT_DIR / "dbp15k_results.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_path}")
print(json.dumps(results, indent=2))
