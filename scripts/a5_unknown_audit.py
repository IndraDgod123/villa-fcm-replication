# -*- coding: utf-8 -*-
"""A5: (i) sensitivity to 'Unknown'/'Other' categories; (ii) audit of a random sample of
100 residual room-and-date collision groups (names compared programmatically, never printed)."""
import json, sys, re
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score as ARI
sys.path.insert(0, "scripts")
from pipeline import full_pipeline, build_features
master = pd.read_csv("data/master_data.csv")
feats0, _, _, _, _, u0, lab0, _, _ = full_pipeline(master)
res = {}
# share of Unknown/Other per cluster
res["unknown_region_by_cluster"] = {int(k): round(100*(feats0.nationality_region[lab0==k]=="Unknown").mean(),1) for k in range(3)}
res["other_platform_by_cluster"] = {int(k): round(100*(feats0.booking_platform_type[lab0==k]=="Other").mean(),1) for k in range(3)}
res["unknown_region_overall"] = round(100*(feats0.nationality_region=="Unknown").mean(),1)
res["other_platform_overall"] = round(100*(feats0.booking_platform_type=="Other").mean(),1)
def compare(mask, name):
    d = master[mask].reset_index(drop=True)
    f, X, pca, Xp, c, u, lab, it, fpc = full_pipeline(d)
    ref = lab0[mask.values]
    res[name] = dict(rows=int(len(d)), ARI=round(float(ARI(ref, lab)),4), PC=round(float(fpc),4),
                     sizes=[int((lab==k).sum()) for k in range(3)])
unk = feats0.nationality_region.eq("Unknown") | feats0.booking_platform_type.eq("Other")
compare(~feats0.nationality_region.eq("Unknown"), "drop_unknown_region")
compare(~unk, "drop_unknown_region_and_other_platform")
compare(master.villa.ne("Bugis"), "drop_bugis_all_unknown_nationality")
# audit of 100 random K3 groups
m = master.copy()
k3 = ["villa", "room_unit", "date_in", "date_out"]
g = m[m.duplicated(subset=k3, keep=False)].copy()
g["_g"] = g.groupby(k3, dropna=False).ngroup()
rng = np.random.default_rng(42)
ids = rng.choice(g["_g"].unique(), size=100, replace=False)
def norm(s):
    s = "" if not isinstance(s, str) else s
    s = re.sub(r"^(Mr/s\.|Mr\.|Mrs\.|Ms\.)", "", s).strip().lower()
    return s
cls = {"same_name_honorific_only": 0, "same_name_exact_diff_pax": 0, "different_names": 0, "name_missing": 0}
same_src = 0
for i in ids:
    gg = g[g._g == i]
    raw = gg.guest_name.fillna("").astype(str).str.strip().str.lower()
    nn = gg.guest_name.apply(norm)
    same_src += int(gg.source_file.nunique() == 1)
    if (raw == "").any(): cls["name_missing"] += 1
    elif raw.nunique() == 1: cls["same_name_exact_diff_pax"] += 1
    elif nn.nunique() == 1: cls["same_name_honorific_only"] += 1
    else: cls["different_names"] += 1
res["k3_audit_sample100"] = dict(classes=cls, same_source_file=same_src)
json.dump(res, open("out/a5_unknown_audit.json", "w"), indent=2)
print(json.dumps(res, indent=2))
