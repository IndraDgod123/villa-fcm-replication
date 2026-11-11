# -*- coding: utf-8 -*-
"""A2: deduplication key sensitivity. Re-run the full pipeline after stricter
de-duplication variants and compare with the published solution on the rows that
survive. Cluster identity is matched by the Hungarian method on the confusion matrix."""
import json, sys, re
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score as ARI
sys.path.insert(0, "scripts")
from pipeline import full_pipeline

master = pd.read_csv("data/master_data.csv")
_, _, _, _, _, u0, lab0, _, _ = full_pipeline(master)
master["_lab0"] = lab0

def strip_title(s):
    if not isinstance(s, str):
        return s
    return re.sub(r"^(Mr/s\.|Mr\.|Mrs\.|Ms\.)", "", s).strip().lower()

variants = {}
m = master.copy()
m["_name_norm"] = m["guest_name"].apply(strip_title)
# D1: same key as original but honorific-insensitive name
d1 = m.drop_duplicates(subset=["villa", "room_unit", "date_in", "date_out", "pax", "_name_norm"], keep="first")
# D2: honorific-insensitive name, pax ignored
d2 = m.drop_duplicates(subset=["villa", "room_unit", "date_in", "date_out", "_name_norm"], keep="first")
# D3: one booking per room unit per stay (only where room_unit is recorded)
has_room = m["room_unit"].notna()
d3 = pd.concat([m[has_room].drop_duplicates(subset=["villa", "room_unit", "date_in", "date_out"], keep="first"),
                m[~has_room]]).sort_index()
for name, d in {"D1_title_insensitive": d1, "D2_title_insensitive_no_pax": d2,
                "D3_room_and_dates": d3}.items():
    d = d.drop(columns=["_name_norm"]).copy()
    lab_ref = d["_lab0"].values
    feats, Xdf, pca, Xp, cntr, u, lab, it, fpc = full_pipeline(d.drop(columns=["_lab0"]).reset_index(drop=True))
    cm = pd.crosstab(lab_ref, lab).values
    r, c = linear_sum_assignment(-cm)
    agree = cm[r, c].sum() / cm.sum()
    # profile of the matched clusters
    mapping = dict(zip(c, r))
    lab_m = np.vectorize(mapping.get)(lab)
    prof = {}
    for k in range(3):
        sub = feats[lab_m == k]
        prof[f"C{k}"] = dict(
            size=int((lab_m == k).sum()), share=round(100 * (lab_m == k).mean(), 1),
            peak=round(100 * (sub.booking_season == "Peak").mean(), 1),
            asia=round(100 * (sub.nationality_region == "Asia").mean(), 1),
            intl_ota=round(100 * (sub.booking_platform_type == "International_OTA").mean(), 1),
            dmc=round(100 * (sub.booking_platform_type == "DMC").mean(), 1))
    variants[name] = dict(rows=int(len(d)), removed_vs_published=int(len(master) - len(d)),
                          pca_var=round(float(pca.explained_variance_ratio_.sum()), 4),
                          PC=round(float(fpc), 4), iterations=int(it),
                          ARI_vs_published=round(float(ARI(lab_ref, lab)), 4),
                          label_agreement=round(float(agree), 4), profiles=prof)
    print(name, variants[name]["rows"], variants[name]["ARI_vs_published"], variants[name]["label_agreement"])

json.dump(variants, open("out/a2_dedup_sensitivity.json", "w"), indent=2)
print(json.dumps(variants, indent=2))
