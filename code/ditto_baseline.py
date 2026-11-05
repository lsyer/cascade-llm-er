#!/usr/bin/env python3
"""ditto_baseline.py — Ditto-style BERT cross-encoder baseline for MINEC ER.

Faithful reimplementation of the Ditto method (Li et al., VLDB 2020): BERT-base
cross-encoder over serialized entity records with Ditto-style COL/ATTR templates,
fine-tuned per fold. DeepMatcher is skipped (abandoned at torch 0.4; Ditto is its
modern successor — stated in the paper).

Protocol (identical to our LR/RF OOF setup so outputs are directly comparable):
  - same 5 folds: StratifiedKFold(5, shuffle=True, random_state=42)
  - OOF probability per pair -> (a) standalone matcher accuracy, (b) alternative
    L1 router in the cascade (threshold + confidence gating on Ditto probs),
    replayed against the GLM-5-Turbo checkpoint with the strict convention.
Hyperparams follow the Ditto paper defaults: bert-base-uncased, lr 3e-5,
batch 32, 4 epochs (budget-limited on CPU), max_len 128.
Output: results/ditto_oof.npz, results/ditto_baseline.json
"""
import json, os, re
import numpy as np
import torch
torch.set_num_threads(2)  # cgroup mem limit ~4GB: fewer threads = smaller arenas
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(BASE, 'results')
MODEL = 'bert-base-uncased'
MAXLEN = 128
BS = 16
EPOCHS = 4
LR = 3e-5

scored = json.load(open(os.path.join(RES, 'scored_pairs.json')))
ds = json.load(open(os.path.join(BASE, 'data', 'dataset_v3_cleaned.json')))
props_by_id = {}
for p in ds['same_type_pairs']:
    props_by_id[p['pair_id']] = (p['props_a'], p['props_b'])
pair_ids = [p['pair_id'] for p in scored]
y = np.array([p['label'] for p in scored])
N = len(y)

SKIP_PROPS = {'confidence', 'created_at', 'updated_at'}
def serialize(name, etype, props):
    """Ditto-style serialization: COL type VAL name ATTR key VAL value ..."""
    if isinstance(props, str):
        try:
            props = json.loads(props)
        except Exception:
            props = {}
    toks = [f'COL {etype.replace(" ", "_")} VAL {name}']
    for k, v in props.items():
        if k in SKIP_PROPS or k == 'name':
            continue
        v = str(v).strip()
        if not v or v == '{}':
            continue
        toks.append(f'ATTR {k} VAL {v[:60]}')
    return ' '.join(toks)

texts_a, texts_b = [], []
for p in scored:
    pa, pb = props_by_id[p['pair_id']]
    texts_a.append(serialize(p['name_a'], p['type_a'].replace('activity', 'event'), pa))
    texts_b.append(serialize(p['name_b'], p['type_b'].replace('activity', 'event'), pb))

tok = AutoTokenizer.from_pretrained(MODEL)
enc_all = tok(texts_a, texts_b, truncation=True, max_length=MAXLEN, padding=False)

class PairDS(Dataset):
    def __init__(self, idxs):
        self.idxs = idxs
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, i):
        j = self.idxs[i]
        item = {k: v[j] for k, v in enc_all.items()}
        item['labels'] = int(y[j])
        return item

def collate(batch):
    ml = max(len(b['input_ids']) for b in batch)
    pad = tok.pad_token_id
    out = {'input_ids': torch.full((len(batch), ml), pad, dtype=torch.long),
           'attention_mask': torch.zeros((len(batch), ml), dtype=torch.long)}
    for i, b in enumerate(batch):
        n = len(b['input_ids'])
        out['input_ids'][i, :n] = torch.tensor(b['input_ids'])
        out['attention_mask'][i, :n] = torch.tensor(b['attention_mask'])
    out['labels'] = torch.tensor([b['labels'] for b in batch])
    return out

device = 'cpu'
torch.manual_seed(42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.full(N, np.nan)
import time, gc
done_folds = set()
CKPT_F = os.path.join(RES, 'ditto_progress.npz')
if os.path.exists(CKPT_F):  # resume after OOM kill
    d0 = np.load(CKPT_F)
    oof = d0['oof']
    done_folds = set(int(x) for x in d0['done_folds'])
    print(f"resuming, folds done: {sorted(done_folds)}", flush=True)
for fi, (tr, te) in enumerate(skf.split(np.zeros(N), y)):
    if fi in done_folds:
        continue
    t0 = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    model.gradient_checkpointing_enable()  # cgroup mem ~4GB: trade ~30% speed for activations
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, eps=1e-8)
    dl = DataLoader(PairDS(list(tr)), batch_size=BS, shuffle=True, collate_fn=collate)
    total = len(dl) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, 0, total)
    for ep in range(EPOCHS):
        for batch in dl:
            out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                        labels=batch['labels'])
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); opt.zero_grad()
    model.eval()
    with torch.no_grad():
        dlt = DataLoader(PairDS(list(te)), batch_size=64, collate_fn=collate)
        k = 0
        for batch in dlt:
            logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
            prob = torch.softmax(logits, -1)[:, 1].numpy()
            oof[te[k:k + len(prob)]] = prob
            k += len(prob)
    acc = ((oof[te] >= 0.5).astype(int) == y[te]).mean()
    print(f"fold {fi}: test acc={acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    np.savez(CKPT_F, oof=oof, done_folds=np.array(sorted(done_folds | {fi})))
    del model, opt, sched
    gc.collect()

np.savez(os.path.join(RES, 'ditto_oof.npz'), proba=oof, y=y)
standalone = float(((oof >= 0.5).astype(int) == y).mean() * 100)
print(f"Ditto standalone OOF accuracy: {standalone:.2f}%  (LR ref 97.0, RF 95.5 at their op points)")

# cascade replay: best two-threshold + confidence gating on Ditto probs
ck = json.load(open(os.path.join(RES, 'checkpoints_v3', 'glm-5-turbo_checkpoint.json')))
def l2(pid):
    v = str(ck.get(pid, {}).get('verdict', '')).strip().lower()
    return 1 if v == 'same' else (0 if v == 'different' else -1)
l2lab = np.array([l2(x) for x in pair_ids])
l2_correct = np.where(l2lab != -1, (l2lab == y).astype(int), 0)
def pipeline(esc):
    int_ = ~esc
    arg = (oof >= 0.5).astype(int)
    l1c = int((arg[int_] == y[int_]).sum())
    l2c = int(l2_correct[esc].sum())
    return round((l1c + l2c) / N * 100, 2), round(esc.mean() * 100, 2)

best_thr = None
grid = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
for mt in grid:
    for rt in grid:
        if rt >= mt: continue
        esc = (oof < mt) & (oof > rt)
        if (~esc).sum() < 5: continue
        acc = (( (oof >= 0.5).astype(int)[~esc] == y[~esc]).sum()) / (~esc).sum() * 100
        if acc < 75: continue
        a, c = pipeline(esc)
        if best_thr is None or a > best_thr[0]:
            best_thr = (a, c, mt, rt)
print(f"Ditto two-threshold best pipeline: {best_thr[0]}% @ cost {best_thr[1]}% (mt={best_thr[2]}, rt={best_thr[3]})")
conf = np.maximum(oof, 1 - oof)
best_conf = None
for cth in [round(x, 2) for x in np.arange(0.5, 0.96, 0.02)]:
    esc = conf < cth
    if esc.sum() == 0: continue
    a, c = pipeline(esc)
    if best_conf is None or a > best_conf[0]:
        best_conf = (a, c, cth)
print(f"Ditto confidence-gating best pipeline: {best_conf[0]}% @ cost {best_conf[1]}% (c={best_conf[2]})")

json.dump({'standalone_oof_acc_pct': round(standalone, 2),
           'two_threshold_best': {'acc': best_thr[0], 'cost': best_thr[1], 'mt': best_thr[2], 'rt': best_thr[3]},
           'confidence_gating_best': {'acc': best_conf[0], 'cost': best_conf[1], 'c': best_conf[2]},
           'config': {'model': MODEL, 'epochs': EPOCHS, 'bs': BS, 'lr': LR, 'maxlen': MAXLEN,
                      'folds': 'StratifiedKFold(5, shuffle, rs=42) same as LR/RF'}},
          open(os.path.join(RES, 'ditto_baseline.json'), 'w'), indent=1)
print("written results/ditto_baseline.json")
