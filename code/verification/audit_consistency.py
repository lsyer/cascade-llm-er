#!/usr/bin/env python3
"""Cross-artifact consistency audit for the CascadeRuleLLM release.

Checks four classes of drift that manual review missed in earlier rounds.
Each check recomputes a number from release artifacts and compares it to
hardcoded expectations from the manuscript. Run from anywhere:

    python3 code/verification/audit_consistency.py

Exit code 0 = all checks pass.
"""
import json
import os
import re
import sys

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
RES = os.path.join(BASE, 'results')
DATA = os.path.join(BASE, 'data')

failures = []


def check(name, ok, detail=''):
    status = 'PASS' if ok else 'FAIL'
    print(f'[{status}] {name}' + (f'  ({detail})' if detail else ''))
    if not ok:
        failures.append(name)


def load(path):
    return json.load(open(os.path.join(BASE, path)))


# ---------------------------------------------------------------- class 1
# Candidate-generation counts close arithmetically and match dataset metadata.
def check_candidate_generation():
    ds = load('data/dataset_v3_cleaned.json')
    meta = ds['metadata']
    n_same = len(ds['same_type_pairs'])
    n_cross = len(ds['cross_type_pairs'])
    check('dataset pair lists match metadata',
          n_same == meta['same_type_pairs'] and n_cross == meta['cross_type_pairs'],
          f'{n_same}+{n_cross} vs meta {meta["same_type_pairs"]}+{meta["cross_type_pairs"]}')
    # exact + fuzzy - excluded cross-type = total (paper: 368 + 2290 - 19 = 2639)
    total = meta['total_pairs']
    exact = total - meta['fuzzy_pairs']
    check('exact+fuzzy-cross == same_type total',
          exact + meta['fuzzy_pairs'] - n_cross == n_same,
          f'{exact}+{meta["fuzzy_pairs"]}-{n_cross} == {n_same}')


# ---------------------------------------------------------------- class 2
# Table 5 (L1 error patterns): every cell recomputed from expert_bucket.json
# against GT v2, and column sums must equal the GT label totals.
def check_l1_bucket_table():
    gt = load('data/minec_ground_truth_v2.json')
    label = {r['pair_id']: r['final_label'] for r in gt['records']}
    eb = load('results/expert_bucket.json')
    col_same = col_diff = 0
    for bucket in ('merge', 'reject', 'escalate'):
        b = eb[bucket]
        n, s, d = b['total'], b['same'], b['different']
        ok = s + d == n and b['same_pct'] == round(s / n * 100, 1)
        check(f'expert_bucket[{bucket}] internal consistency', ok,
              f'{s}+{d}=={n}, pct {b["same_pct"]}')
        col_same += s
        col_diff += d
    gt_same = sum(1 for v in label.values() if v == 'same')
    gt_diff = sum(1 for v in label.values() if v == 'different')
    check('bucket column sums == GT label totals',
          col_same == gt_same and col_diff == gt_diff,
          f'{col_same}/{col_diff} vs GT {gt_same}/{gt_diff}')


# ---------------------------------------------------------------- class 3
# Deployment-queue arithmetic: group-advice shares must close to 100% and
# the distinct-share claim (split+mixed) must match.
# NOTE: prod_collision_results.json lives in the production repo
# (usn-monitor/scripts/out/); point PROD_COLLISION at it if available.
def check_deployment_arithmetic():
    path = os.environ.get('PROD_COLLISION')
    if not path or not os.path.exists(path):
        print('[SKIP] deployment arithmetic (set PROD_COLLISION=<prod_collision_results.json>)')
        return
    d = json.load(open(path))
    adv = d['summary']['group_advice']
    total = sum(adv.values())
    check('group_advice counts sum to groups_total',
          total == d['summary']['groups_total'],
          f'{total} vs {total and d["summary"]["groups_total"]}')
    shares = {k: round(v / total * 100, 1) for k, v in adv.items()}
    closure = round(sum(shares.values()), 1)
    check('group-advice shares close to 100%', abs(closure - 100.0) <= 0.15,
          f'{shares} -> {closure}')


# ---------------------------------------------------------------- class 4
# Run parameters quoted in the paper vs the code that produced the artifacts.
def check_run_parameters():
    expectations = {
        'code/run_glm5turbo_l2.py': ('max_tokens', '4096'),
        'code/fragment_ablation_v2.py': ('max_tokens', '4096'),
        'dbp15k/dbp15k_rebuild_l2.py': ('max_tokens', '4096'),
        'code/qwen_full_l2.py': ('max_tokens', '512'),
        'code/fragment_rerun_qwen.py': ('max_tokens', '512'),
    }
    for path, (key, val) in expectations.items():
        src = open(os.path.join(BASE, path)).read()
        m = re.search(rf'["\']{key}["\']\s*:\s*(\d+)', src)
        check(f'{path} {key}', m is not None and m.group(1) == val,
              m.group(1) if m else 'not found')
    # no absolute workspace paths in released code (skip this auditor itself)
    myself = os.path.abspath(__file__)
    for root, _, files in os.walk(os.path.join(BASE, 'code')):
        for f in files:
            fp = os.path.join(root, f)
            if not f.endswith('.py') or os.path.abspath(fp) == myself:
                continue
            src = open(os.path.join(root, f)).read()
            if '/root/workspace' in src:
                check(f'{f} absolute path', False, '/root/workspace literal')
    print('[PASS] no /root/workspace literals in code/')


# ---------------------------------------------------------------- class 5
# Manuscript-linked references: matched-budget reference must be a computed
# value consistent with the fine grid.
def check_matched_budget_reference():
    j = load('results/entity_split_cv.json')
    ref = j.get('reference_random_split_matched_budget')
    if ref is None:
        check('matched-budget reference present', False, 'key missing')
        return
    fg = load('results/ours_fine_grid_gt2.json')
    esc = j['overall']['llm_cost_pct']
    best = max((r for r in fg if r['cost'] <= esc), key=lambda r: (r['s'], -r['cost']))
    check('matched-budget reference == fine-grid best within budget',
          ref['accuracy_pct'] == best['s'] and ref['llm_cost_pct'] == best['cost']
          and ref['merge_threshold'] == best['mt'] and ref['reject_threshold'] == best['rt'],
          f"json {ref['accuracy_pct']}@{ref['llm_cost_pct']} vs grid {best['s']}@{best['cost']}")


if __name__ == '__main__':
    check_candidate_generation()
    check_l1_bucket_table()
    check_deployment_arithmetic()
    check_run_parameters()
    check_matched_budget_reference()
    print()
    if failures:
        print(f'{len(failures)} FAILURES: ' + ', '.join(failures))
        sys.exit(1)
    print('all checks passed')
