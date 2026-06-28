#!/usr/bin/env python3
"""
Threshold sensitivity analysis: plot pipeline accuracy vs merge threshold.
Uses existing per-pair L1 scores + L2 labels — no new API calls needed.

Data sources:
- MINEC: l1_results_all.json (2217 pairs with score + label)
- DBP15K: re-compute from raw data (same logic as run_dbp15k_experiment.py)
"""
import json, re, random, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REL = Path("/root/workspace/tanshu_docs/experiments/release")
OUT = Path("/root/workspace/tanshu_docs/experiments/dbp15k")

# ─── MINEC ───
print("Loading MINEC per-pair data...")
with open(OUT / "minec_labeled_pairs.json") as f:
    minec_raw = json.load(f)

minec_pairs = [{'score': r['score'], 'label': r['label']} for r in minec_raw]

print(f"  MINEC: {len(minec_pairs)} pairs ({sum(p['label'] for p in minec_pairs)} positive)")

# MINEC L2 accuracy (GLM-5 = 97.1%, use as pipeline L2)
MINEC_L2_ACC = 0.971
# MINEC L1+feedback accuracy (unified LR, 80.1%)
MINEC_L1_FB_ACC = 0.801

# ─── DBP15K ───
print("Loading DBP15K data...")
DATA = OUT / "JAPE" / "data" / "dbp15k" / "zh_en"

def load_labels(fp):
    d = {}
    for line in open(fp):
        p = line.strip().split('\t')
        if len(p) >= 3: d[p[0]] = {'label': p[1], 'trans': p[2]}
        elif len(p) == 2: d[p[0]] = {'label': p[1], 'trans': p[1]}
    return d

def load_ILLs(fp):
    return set(tuple(l.strip().split('\t')) for l in open(fp) if len(l.strip().split('\t')) == 2)

def load_attrs(fp):
    d = {}
    for line in open(fp):
        p = line.strip().split('\t')
        if len(p) >= 2: d[p[0]] = set(x.split('/')[-1].lower() for x in p[1:])
    return d

def load_triples(fp):
    d = defaultdict(list)
    for line in open(fp):
        p = line.strip().split('\t')
        if len(p) == 3: d[p[0]].append((p[1].split('/')[-1], p[2].split('/')[-1]))
    return d

def norm(s):
    s = s.lower().strip()
    s = re.sub(r'[_\-\(\)]', ' ', s)
    return re.sub(r'\s+', ' ', s)

zh_labels = load_labels(DATA / "s_labels")
en_labels = load_labels(DATA / "t_labels")
gold = load_ILLs(DATA / "ent_ILLs")
zh_attrs = load_attrs(DATA / "training_attrs_1")
en_attrs = load_attrs(DATA / "training_attrs_2")
zh_tris = load_triples(DATA / "s_triples")
en_tris = load_triples(DATA / "t_triples")

# Build eval set (same seed as experiment)
random.seed(42)
pos = random.sample(list(gold), 2000)
zh_uris = list(zh_labels.keys())
en_uris = list(en_labels.keys())
idx = defaultdict(list)
for u in en_uris:
    n = norm(en_labels.get(u, {}).get('label', ''))
    if n:
        pre = n.split()[0][:3] if n.split() else n[:3]
        idx[pre].append(u)

neg = []
att = 0
while len(neg) < 2000 and att < 50000:
    att += 1
    z = random.choice(zh_uris)
    if z not in zh_labels: continue
    n = norm(zh_labels[z].get('trans', ''))
    if not n: continue
    pre = n.split()[0][:3] if n.split() else n[:3]
    cs = idx.get(pre, [])
    if not cs: continue
    e = random.choice(cs)
    if (z, e) not in gold: neg.append((z, e))

def l1_score(z, e):
    t = norm(zh_labels.get(z, {}).get('trans', ''))
    l = norm(en_labels.get(e, {}).get('label', ''))
    t1, t2 = set(t.split()), set(l.split())
    ns = len(t1 & t2) / max(len(t1 | t2), 1) if t1 and t2 else 0
    if t == l and t: ns = 1.0
    za, ea = zh_attrs.get(z, set()), en_attrs.get(e, set())
    as_ = len(za & ea) / max(len(za | ea), 1) if za and ea else 0
    zd, ed = len(zh_tris.get(z, [])), len(en_tris.get(e, []))
    ds = min(zd, ed) / max(zd, ed) if zd > 0 and ed > 0 else 0
    s = ns * 0.40 + as_ * 0.35 + ds * 0.15
    if as_ == 0 and (zd > 5 or ed > 5): s *= 0.7
    return s

dbp_pairs = [{'score': l1_score(z, e), 'label': 1} for z, e in pos]
dbp_pairs += [{'score': l1_score(z, e), 'label': 0} for z, e in neg]
print(f"  DBP15K: {len(dbp_pairs)} pairs ({sum(p['label'] for p in dbp_pairs)} positive)")

# DBP15K L2 accuracy = 100%
DBP_L2_ACC = 1.000

REJECT_TH = 0.3  # Fixed reject threshold

# ─── Compute threshold sweep ───
def threshold_sweep(pairs, l2_acc, merge_ths):
    """For each merge threshold, compute L1 accuracy, interception, pipeline accuracy."""
    results = []
    for th in merge_ths:
        tp = fp = tn = fn = 0
        escalated = 0
        for p in pairs:
            s = p['score']
            l = p['label']
            if s >= th:
                if l == 1: tp += 1
                else: fp += 1
            elif s < REJECT_TH:
                if l == 0: tn += 1
                else: fn += 1
            else:
                escalated += 1
        intercepted = len(pairs) - escalated
        l1_acc = (tp + tn) / max(intercepted, 1)
        interception_rate = intercepted / len(pairs)
        # Pipeline: L1 intercepts at l1_acc, L2 handles escalated at l2_acc
        pipeline = l1_acc * interception_rate + l2_acc * (1 - interception_rate)
        # L1 merge precision
        merge_prec = tp / max(tp + fp, 1)
        results.append({
            'threshold': th,
            'l1_accuracy': l1_acc,
            'interception': interception_rate,
            'pipeline': pipeline,
            'merge_precision': merge_prec,
            'escalation': 1 - interception_rate,
        })
    return results

thresholds = np.arange(0.35, 0.95, 0.05)

minec_sweep = threshold_sweep(minec_pairs, MINEC_L2_ACC, thresholds)
dbp_sweep = threshold_sweep(dbp_pairs, DBP_L2_ACC, thresholds)

# Also compute adaptive pipeline (replace L1 fixed accuracy with L1+feedback accuracy)
def adaptive_pipeline(sweep_data, l1_fb_acc, l2_acc):
    """Recompute pipeline with feedback L1 accuracy instead of fixed."""
    for r in sweep_data:
        r['pipeline_adaptive'] = l1_fb_acc * r['interception'] + l2_acc * (1 - r['interception'])

adaptive_pipeline(minec_sweep, MINEC_L1_FB_ACC, MINEC_L2_ACC)
# DBP15K already has adaptive (87.7%)
adaptive_pipeline(dbp_sweep, 0.877, DBP_L2_ACC)

# ─── Plot ───
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Pipeline accuracy vs threshold
ax1 = axes[0]
# Fixed L1 pipeline
ax1.plot([r['threshold'] for r in minec_sweep], [r['pipeline'] for r in minec_sweep],
         'o--', color='#2196F3', linewidth=1.5, markersize=5, alpha=0.6, label='MINEC fixed L1')
ax1.plot([r['threshold'] for r in dbp_sweep], [r['pipeline'] for r in dbp_sweep],
         's--', color='#FF5722', linewidth=1.5, markersize=5, alpha=0.6, label='DBP15K fixed L1')
# Adaptive L1 pipeline
ax1.plot([r['threshold'] for r in minec_sweep], [r['pipeline_adaptive'] for r in minec_sweep],
         'o-', color='#2196F3', linewidth=2.5, markersize=7, label='MINEC adaptive L1')
ax1.plot([r['threshold'] for r in dbp_sweep], [r['pipeline_adaptive'] for r in dbp_sweep],
         's-', color='#FF5722', linewidth=2.5, markersize=7, label='DBP15K adaptive L1')

# Mark current threshold (0.5)
ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Current (θ=0.5)')
# Find knee points
minec_best = max(minec_sweep, key=lambda r: r['pipeline'])
dbp_best = max(dbp_sweep, key=lambda r: r['pipeline'])
ax1.annotate(f"Best: {minec_best['pipeline']:.1%}\n(θ={minec_best['threshold']:.2f})",
             xy=(minec_best['threshold'], minec_best['pipeline']),
             xytext=(minec_best['threshold']+0.1, minec_best['pipeline']-0.03),
             fontsize=9, arrowprops=dict(arrowstyle='->', color='#2196F3'), color='#2196F3')
ax1.annotate(f"Best: {dbp_best['pipeline']:.1%}\n(θ={dbp_best['threshold']:.2f})",
             xy=(dbp_best['threshold'], dbp_best['pipeline']),
             xytext=(dbp_best['threshold']+0.1, dbp_best['pipeline']-0.05),
             fontsize=9, arrowprops=dict(arrowstyle='->', color='#FF5722'), color='#FF5722')

ax1.set_xlabel('Merge Threshold θ', fontsize=12)
ax1.set_ylabel('Pipeline Accuracy', fontsize=12)
ax1.set_title('(a) Pipeline Accuracy vs. Threshold', fontsize=13)
ax1.legend(fontsize=10)
ax1.set_ylim(0.75, 1.02)
ax1.grid(True, alpha=0.3)

# Panel 2: Interception rate vs threshold (cost side)
ax2 = axes[1]
ax2.plot([r['threshold'] for r in minec_sweep], [r['interception'] for r in minec_sweep],
         'o-', color='#2196F3', linewidth=2, markersize=6, label='MINEC')
ax2.plot([r['threshold'] for r in dbp_sweep], [r['interception'] for r in dbp_sweep],
         's-', color='#FF5722', linewidth=2, markersize=6, label='DBP15K')
ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Current (θ=0.5)')
ax2.set_xlabel('Merge Threshold θ', fontsize=12)
ax2.set_ylabel('L1 Interception Rate', fontsize=12)
ax2.set_title('(b) L1 Interception Rate vs. Threshold', fontsize=13)
ax2.legend(fontsize=10)
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

out_path = OUT / "threshold_sensitivity.pdf"
fig.savefig(out_path, bbox_inches='tight', dpi=300)
out_path_png = OUT / "threshold_sensitivity.png"
fig.savefig(out_path_png, bbox_inches='tight', dpi=150)
plt.close()

print(f"\nSaved to {out_path}")

# Print summary table
print(f"\n{'θ':>6} | {'MINEC Pipeline':>15} | {'MINEC Intercept':>16} | {'DBP15K Pipeline':>16} | {'DBP15K Intercept':>17}")
print("-" * 80)
for i, th in enumerate(thresholds):
    m = minec_sweep[i]
    d = dbp_sweep[i]
    marker = " ← current" if abs(th - 0.5) < 0.001 else ""
    print(f"{th:6.2f} | {m['pipeline']:15.1%} | {m['interception']:16.1%} | {d['pipeline']:16.1%} | {d['interception']:17.1%}{marker}")

# Save results JSON
sweep_data = {
    'thresholds': [round(t, 2) for t in thresholds],
    'minec': [{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in minec_sweep],
    'dbp15k': [{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in dbp_sweep],
    'minec_l2_accuracy': MINEC_L2_ACC,
    'dbp15k_l2_accuracy': DBP_L2_ACC,
    'reject_threshold': REJECT_TH,
}
with open(OUT / "threshold_sensitivity.json", 'w') as f:
    json.dump(sweep_data, f, indent=2)
print(f"\nData saved to {OUT / 'threshold_sensitivity.json'}")
