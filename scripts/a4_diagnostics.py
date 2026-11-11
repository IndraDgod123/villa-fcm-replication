# -*- coding: utf-8 -*-
"""A4: extra diagnostics: distinct centres per k, k=7 profile, simple-rule surrogate of the FCM
partition, profile of the rebalanced solution, and 2020 split around the border closure."""
import json, sys
import numpy as np, pandas as pd
import skfuzzy as fuzz
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.tree import DecisionTreeClassifier, export_text
sys.path.insert(0, "scripts")
from pipeline import build_features, encode, run_fcm, validity

R = {}
master = pd.read_csv("data/master_data.csv")
f = build_features(master); X = encode(f).values.astype(float)
S = np.load("out/main_solution.npz"); lab = S["lab"]; Xp = S["Xp"]; f["cluster"] = lab

def profile(mask_lab, ff):
    out = {}
    for k in sorted(np.unique(mask_lab)):
        s = ff[mask_lab == k]
        out[int(k)] = dict(n=int(len(s)), peak=round(100*(s.booking_season=="Peak").mean(),1),
            asia=round(100*(s.nationality_region=="Asia").mean(),1),
            oceania=round(100*(s.nationality_region=="Oceania").mean(),1),
            domestic=round(100*(s.nationality_region=="Domestic").mean(),1),
            unknown=round(100*(s.nationality_region=="Unknown").mean(),1),
            dmc=round(100*(s.booking_platform_type=="DMC").mean(),1),
            ota=round(100*(s.booking_platform_type=="International_OTA").mean(),1),
            local_agent=round(100*(s.booking_platform_type=="Local_Agent").mean(),1),
            dream_suite=round(100*(s.villa=="Dream Suite").mean(),1),
            pax=round(s.pax.mean(),2), los=round(s.length_of_stay.mean(),2))
    return out

# distinct centres per k (best of 20)
dc = {}
for k in range(2, 8):
    best = None
    for s in range(20):
        c, u, jm, it, fpc = run_fcm(Xp, k=k, seed=s)
        if best is None or jm[-1] < best[0]:
            best = (jm[-1], c, u)
    c = best[1]
    d2 = ((c[:, None, :]-c[None, :, :])**2).sum(2)
    groups = []
    for i in range(k):
        for g in groups:
            if d2[i, g[0]] < 1e-6: g.append(i); break
        else:
            groups.append([i])
    dc[k] = dict(distinct_centres=len(groups), sizes=np.bincount(np.argmax(best[2], 0), minlength=k).tolist())
    if k == 7:
        l7 = np.argmax(best[2], 0)
        R["k7_profile"] = profile(l7, f)
        R["k7_ARI_vs_k3"] = round(ARI(lab, l7), 4)
R["distinct_centres"] = dc

# m = 1.3 profile
best = None
for s in range(10):
    c, u, jm, it, fpc = run_fcm(Xp, k=3, m=1.3, seed=s)
    if best is None or jm[-1] < best[0]: best = (jm[-1], c, u)
R["m13_profile"] = profile(np.argmax(best[2], 0), f)

# surrogate rules
asia = (f.nationality_region == "Asia").values; peak = (f.booking_season == "Peak").values
rule_ap = np.where(asia & peak, 1, np.where(asia & ~peak, 2, 0))
R["rule_asia_peak"] = dict(ARI=round(ARI(lab, rule_ap), 4), agreement=round(float((rule_ap == lab).mean()), 4))
nonota = (f.booking_platform_type != "International_OTA").values
rule_apo = np.where(asia & peak & nonota, 1, np.where(asia & ~peak & nonota, 2, 0))
R["rule_asia_peak_nonOTA"] = dict(ARI=round(ARI(lab, rule_apo), 4), agreement=round(float((rule_apo == lab).mean()), 4))
D = pd.get_dummies(f[["nationality_region", "booking_season", "booking_platform_type"]]).astype(int)
D["pax"] = f.pax; D["los"] = f.length_of_stay
for depth in [2, 3, 4]:
    t = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(D, lab)
    R[f"tree_depth{depth}_accuracy"] = round(float(t.score(D, lab)), 4)
    if depth == 3:
        R["tree_depth3_rules"] = export_text(t, feature_names=list(D.columns), max_depth=3)

# rebalanced solution profile (Dream Suite subsampled to 2248, one repeat)
ds_idx = np.where(f.villa.values == "Dream Suite")[0]; oth = np.where(f.villa.values != "Dream Suite")[0]
rr = np.random.default_rng(0); yrs = f.booking_year.values[ds_idx]; pick = []
for y in np.unique(yrs):
    pool = ds_idx[yrs == y]; take = int(round(2248*len(pool)/len(ds_idx)))
    pick.extend(rr.choice(pool, size=min(take, len(pool)), replace=False))
idx = np.concatenate([np.array(pick), oth])
pc = PCA(n_components=8, random_state=42).fit(X[idx]); c, u, *_ = run_fcm(pc.transform(X[idx]), seed=42)
R["rebalanced_profile_on_subsample"] = profile(np.argmax(u, 0), f.iloc[idx].reset_index(drop=True))

# 2020 around border closure (Jan-Mar vs Apr-Dec)
f20 = f[f.booking_year == 2020].copy(); f20["q"] = np.where(f20.booking_month <= 3, "Jan-Mar", "Apr-Dec")
R["y2020_split"] = {q: dict(n=int(len(g)), dream_suite=round(100*(g.villa=="Dream Suite").mean(),1),
                            cluster_pct=(g.cluster.value_counts(normalize=True).sort_index()*100).round(1).to_dict(),
                            domestic=round(100*(g.nationality_region=="Domestic").mean(),1),
                            asia=round(100*(g.nationality_region=="Asia").mean(),1))
                    for q, g in f20.groupby("q")}
# yearly composition restricted to Dream Suite (removes coverage confound)
dsf = f[f.villa == "Dream Suite"]
t = pd.crosstab(dsf.booking_year, dsf.cluster)
R["dream_suite_cluster_by_year"] = dict(counts=t.to_dict(), rowpct=(t.div(t.sum(1), axis=0)*100).round(1).to_dict())
kf = f[f.villa == "Kembali"]; t2 = pd.crosstab(kf.booking_year, kf.cluster)
R["kembali_cluster_by_year"] = dict(counts=t2.to_dict(), rowpct=(t2.div(t2.sum(1), axis=0)*100).round(1).to_dict())
json.dump(R, open("out/a4_diagnostics.json", "w"), indent=2, default=str)
print(json.dumps({k: v for k, v in R.items() if k != "tree_depth3_rules"}, indent=1, default=str))
print(R["tree_depth3_rules"])
