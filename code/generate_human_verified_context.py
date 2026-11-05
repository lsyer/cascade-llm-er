#!/usr/bin/env python3
"""Generate data/human_verified_pairs_context.json.

Joins the 599 uniquely human-verified pairs (342 human-decided GT labels
∪ 300 blind-annotated evaluation pairs, 43 overlap) with per-pair context:
names, vids, full attributes, and source-text fragments. Run from release/:

    python code/generate_human_verified_context.py

Inputs (all in data/): minec_ground_truth_v2.json,
minec_human_annotation_300.json, dataset_v3_cleaned.json, entity_fragments.json.
Output overwrites data/human_verified_pairs_context.json deterministically.
"""
import json
import os

D = os.path.join(os.path.dirname(__file__), '..', 'data')


def load(name):
    with open(os.path.join(D, name), encoding='utf-8') as f:
        return json.load(f)


def main():
    gt = load('minec_ground_truth_v2.json')['records']
    blind = load('minec_human_annotation_300.json')
    recs = blind['records'] if isinstance(blind, dict) else blind
    ds = load('dataset_v3_cleaned.json')
    pmap = {p['pair_id']: p for p in ds['same_type_pairs'] + ds['cross_type_pairs']}
    fr = load('entity_fragments.json')['fragments']

    human_dec = {e['pair_id']: e for e in gt
                 if e['final_label'] != 'unknown'
                 and e['label_source'] != 'glm52_claude_agree'}
    allp = set(human_dec) | {r['pair_id'] for r in recs}
    blind_by_pid = {r['pair_id']: r['human_label'] for r in recs}
    gt_by_pid = {e['pair_id']: e for e in gt}

    out, n_fr = [], 0
    for pid in sorted(allp, key=lambda x: int(x.split('_')[1])):
        p = pmap[pid]
        fa, fb = fr.get(p['vid_a'], []), fr.get(p['vid_b'], [])
        if fa or fb:
            n_fr += 1
        g = gt_by_pid[pid]
        out.append({
            'pair_id': pid, 'type': p['type_a'],
            'vid_a': p['vid_a'], 'vid_b': p['vid_b'],
            'name_a': p['name_a'], 'name_b': p['name_b'],
            'attrs_a': p.get('props_a'), 'attrs_b': p.get('props_b'),
            'fragments_a': fa, 'fragments_b': fb,
            'labels': {
                'gt_final': g['final_label'],
                'label_source': g['label_source'],
                'blind_300': blind_by_pid.get(pid),
            },
        })

    meta = {
        'description': 'Context export for all 599 uniquely human-verified '
                       'pairs: 342 human-decided GT labels (label_source != '
                       'glm52_claude_agree) union 300 blind-annotated '
                       'evaluation pairs (43 overlap). Attributes from '
                       'dataset_v3_cleaned.json; fragments from '
                       'entity_fragments.json (62.6% entity coverage — missing '
                       'fragments are structural, see paper '
                       'fragment-extraction section).',
        'n_pairs': len(out), 'n_with_fragments': n_fr,
        'fragment_coverage_pct': round(n_fr / len(out) * 100, 1),
    }
    path = os.path.join(D, 'human_verified_pairs_context.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'metadata': meta, 'pairs': out}, f,
                  ensure_ascii=False, indent=1)
    print(meta)


if __name__ == '__main__':
    main()
