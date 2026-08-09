#!/usr/bin/env python3
"""Compute 5-fold CV OOF probabilities for LR and RF on MINEC scored pairs.
Outputs oof_probabilities.npz for the Pareto-frontier sweep."""
import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

BASE = os.path.join(os.path.dirname(__file__), '..')
DATA = os.path.join(BASE, 'results', 'scored_pairs.json')
OUT  = os.path.join(BASE, 'results', 'oof_probabilities.npz')

with open(DATA) as f:
    data = json.load(f)

FEATURE_NAMES = [
    'fld_aliases_conflict','fld_aliases_match','fld_category_conflict','fld_category_match',
    'fld_coordinates_conflict','fld_coordinates_match','fld_description_conflict','fld_description_match',
    'fld_equip_type_conflict','fld_equip_type_match','fld_event_type_conflict','fld_event_type_match',
    'fld_home_location_conflict','fld_home_location_match','fld_industry_conflict','fld_industry_match',
    'fld_latest_reported_at_conflict','fld_latest_reported_at_match','fld_loc_type_conflict','fld_loc_type_match',
    'fld_location_name_conflict','fld_location_name_match','fld_name_conflict','fld_name_match',
    'fld_occupation_conflict','fld_occupation_match','fld_org_name_conflict','fld_org_name_match',
    'fld_org_type_conflict','fld_org_type_match','fld_region_conflict','fld_region_match',
    'fld_start_date_conflict','fld_start_date_match','fld_state_conflict','fld_state_match',
    'n_effective_fields','name_containment','name_exact','name_similarity',
    'property_conflict_rate','property_overlap_rate',
]

X = np.array([[p['features_v2'].get(f, 0) for f in FEATURE_NAMES] for p in data])
y = np.array([p['label'] for p in data])
print(f"X shape: {X.shape}, y dist: {np.bincount(y)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr_proba_oof = np.zeros(len(y))
rf_proba_oof = np.zeros(len(y))
for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_scaled[train_idx], y[train_idx])
    lr_proba_oof[test_idx] = lr.predict_proba(X_scaled[test_idx])[:, 1]
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_scaled[train_idx], y[train_idx])
    rf_proba_oof[test_idx] = rf.predict_proba(X_scaled[test_idx])[:, 1]
    print(f"  fold {fold} done")

print(f"LR OOF proba range: [{lr_proba_oof.min():.3f}, {lr_proba_oof.max():.3f}]")
print(f"RF OOF proba range: [{rf_proba_oof.min():.3f}, {rf_proba_oof.max():.3f}]")
np.savez(OUT, lr_proba=lr_proba_oof, rf_proba=rf_proba_oof, y=y)
print(f"Saved: {OUT}")
