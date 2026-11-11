# -*- coding: utf-8 -*-
"""A3: cluster-number robustness, fuzzifier sensitivity, PCA sensitivity, effect sizes,
external validation, bootstrap and temporal stability, leave-one-villa-out, subsampling,
cluster x villa, yearly composition, baselines. Results -> out/a3_results.json"""
import json, sys, re, time
import numpy as np, pandas as pd
import skfuzzy as fuzz
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI, silhouette_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
sys.path.insert(0, "scripts")
from pipeline import build_features, encode, run_fcm, validity

T0 = time.time()
R = {}
master = pd.read_csv("data/master_data.csv")
feats = build_features(master)
Xdf = encode(feats)
X = Xdf.values.astype(float)
pca8 = PCA(n_components=8, random_state=42).fit(X)
Xp = pca8.transform(X)
cntr, u, jm, it, fpc = run_fcm(Xp)
lab = np.argmax(u, axis=0)
n = len(lab)
np.savez("out/main_solution.npz", u=u, lab=lab, Xp=Xp, cntr=cntr,
         evr=pca8.explained_variance_ratio_, components=pca8.components_)
Xdf.to_csv("out/feature_matrix_columns.csv", index=False, header=True, columns=[]) if False else None
R["main"] = dict(iterations=int(it), **{k: round(v, 4) for k, v in validity(Xp, cntr, u).items()},
                 sizes=np.bincount(lab).tolist())

def match(ref, new, k=3):
    cm = pd.crosstab(pd.Series(ref), pd.Series(new)).reindex(index=range(k), columns=range(k), fill_value=0).values
    r, c = linear_sum_assignment(-cm)
    mp = dict(zip(c, r))
    return np.array([mp.get(x, -1) for x in new])

def best_of_seeds(Z, k, m=2.0, seeds=range(20)):
    best = None
    for s in seeds:
        c_, u_, jm_, it_, fpc_ = run_fcm(Z, k=k, m=m, seed=s)
        obj = float(jm_[-1])
        if best is None or obj < best[0]:
            best = (obj, c_, u_, it_, s)
    return best

# ------------------------------------------------------------------ 1. k selection, 20 restarts per k
rows = []
for k in range(2, 8):
    objs, dmins, degenerate = [], [], 0
    best = None
    for s in range(20):
        c_, u_, jm_, it_, fpc_ = run_fcm(Xp, k=k, seed=s)
        v = validity(Xp, c_, u_)
        objs.append(float(jm_[-1])); dmins.append(v["min_centre_dist2"])
        if v["min_centre_dist2"] < 1e-6:
            degenerate += 1
        if best is None or jm_[-1] < best[0]:
            best = (float(jm_[-1]), v, s)
    v = best[1]
    rows.append(dict(k=k, best_seed=best[2], objective=round(best[0], 2),
                     PC=round(v["PC"], 4), PE=round(v["PE"], 4), XB=v["XB"], FS=round(v["FS"], 1),
                     Kwon=v["Kwon"], min_centre_dist2=v["min_centre_dist2"],
                     degenerate_runs=degenerate, objective_spread=round(max(objs) - min(objs), 3)))
    print("k", k, "done", round(time.time() - T0, 1))
R["k_selection"] = rows
pd.DataFrame(rows).to_csv("out/k_selection_multistart.csv", index=False)

# ------------------------------------------------------------------ 2. fuzzifier sensitivity (k=3)
mrows = []
for mval in [1.3, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]:
    obj, c_, u_, it_, s = best_of_seeds(Xp, 3, m=mval, seeds=range(10))
    v = validity(Xp, c_, u_, m=mval)
    l_ = np.argmax(u_, axis=0)
    mx = u_.max(axis=0)
    mrows.append(dict(m=mval, PC=round(v["PC"], 4), PC_norm=round((v["PC"] - 1 / 3) / (1 - 1 / 3), 4),
                      PE=round(v["PE"], 4), XB=round(v["XB"], 4), min_centre_dist2=round(v["min_centre_dist2"], 4),
                      ambiguous_pct=round(100 * (mx < 0.5).mean(), 1), strong_pct=round(100 * (mx >= 0.7).mean(), 1),
                      ARI_vs_m2=round(ARI(lab, l_), 4), sizes=sorted(np.bincount(l_).tolist(), reverse=True)))
R["m_sensitivity"] = mrows
pd.DataFrame(mrows).to_csv("out/m_sensitivity.csv", index=False)
print("m done", round(time.time() - T0, 1))

# ------------------------------------------------------------------ 3. PCA component sensitivity
prow = []
for p in [4, 5, 6, 7, 8, 10, 12, 17, 20]:
    if p == 20:
        Z = X; var = 1.0
    else:
        pc = PCA(n_components=p, random_state=42).fit(X); Z = pc.transform(X); var = pc.explained_variance_ratio_.sum()
    obj, c_, u_, it_, s = best_of_seeds(Z, 3, seeds=range(10))
    l_ = np.argmax(u_, axis=0)
    prow.append(dict(components=p if p != 20 else "none (20-D)", variance=round(float(var), 4),
                     ARI_vs_main=round(ARI(lab, l_), 4), NMI_vs_main=round(NMI(lab, l_), 4),
                     sizes=sorted(np.bincount(l_).tolist(), reverse=True)))
R["pca_sensitivity"] = prow
pd.DataFrame(prow).to_csv("out/pca_sensitivity.csv", index=False)
print("pca done", round(time.time() - T0, 1))

# ------------------------------------------------------------------ 4. effect sizes and tests
def cramers_v(a, b):
    ct = pd.crosstab(a, b)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    k = min(ct.shape) - 1
    return float(chi2), float(p), int(dof), float(np.sqrt(chi2 / (ct.values.sum() * k)))

def categorize_bill(text):
    if pd.isna(text):
        return "Unknown"
    s = str(text).strip().lower()
    if re.match(r"^book\s+\d?(o|0)bs?r?\s*[/\s]", s) or re.match(r"^book\s+[0-9]br/bd[vs]", s):
        return "Room_Type_Leak"
    if "honeymoon" in s:
        return "VIP_Honeymoon" if "vip" in s else "Honeymooner"
    if "anniversary" in s: return "Anniversary"
    if "birthday" in s: return "Birthday"
    if "vip" in s: return "VIP"
    inc = []
    if re.search(r"\b(bf|breakfast)\b", s): inc.append("BF")
    if "massage" in s: inc.append("Massage")
    if "airport" in s: inc.append("Airport")
    if ("lunch" in s) or ("dinner" in s): inc.append("Lunch_Dinner")
    if "food" in s and "disc" in s: inc.append("FB_Discount")
    if "drink" in s or "welcome" in s: inc.append("Welcome_Drink")
    if inc:
        return f"Inclusion_{inc[0]}" if len(inc) == 1 else "Inclusion_Multi"
    if "standard" in s: return "Standard"
    return "Other"

feats["cluster"] = lab
feats["bill_category"] = feats["bill_instruction"].apply(categorize_bill)
eff = []
for col, role in [("nationality_region", "input"), ("booking_season", "input"), ("booking_platform_type", "input"),
                  ("group_size_category", "derived from input (pax)"), ("length_category", "derived from input (length of stay)"),
                  ("villa", "not an input"), ("booking_year", "not an input")]:
    chi2, p, dof, v = cramers_v(feats[col], feats["cluster"])
    eff.append(dict(variable=col, role=role, test="Chi-square", statistic=round(chi2, 2), dof=dof,
                    p=p, effect="Cramer's V", effect_size=round(v, 3)))
for col in ["pax", "length_of_stay"]:
    groups = [feats.loc[feats.cluster == k, col].values for k in range(3)]
    F, pF = stats.f_oneway(*groups)
    eta2 = (F * 2) / (F * 2 + (n - 3))
    H, pH = stats.kruskal(*groups)
    eps2 = (H - 3 + 1) / (n - 3)
    eff.append(dict(variable=col, role="input", test="ANOVA", statistic=round(float(F), 2), dof="2, %d" % (n - 3),
                    p=float(pF), effect="eta squared", effect_size=round(float(eta2), 3)))
    eff.append(dict(variable=col, role="input", test="Kruskal-Wallis", statistic=round(float(H), 2), dof=2,
                    p=float(pH), effect="epsilon squared", effect_size=round(float(eps2), 3)))
# external: bill instruction (Dream Suite only, informative categories)
ds = feats[(feats.villa == "Dream Suite") & (~feats.bill_category.isin(["Room_Type_Leak", "Unknown"]))]
chi2, p, dof, v = cramers_v(ds["bill_category"], ds["cluster"])
eff.append(dict(variable="bill_instruction category (Dream Suite)", role="not an input", test="Chi-square",
                statistic=round(chi2, 2), dof=dof, p=p, effect="Cramer's V", effect_size=round(v, 3)))
R["effect_sizes"] = eff
pd.DataFrame(eff).to_csv("out/effect_sizes.csv", index=False)

bill_ct = pd.crosstab(ds["bill_category"], ds["cluster"])
bill_pct = (bill_ct / bill_ct.sum(axis=0) * 100).round(1)
R["bill_counts"] = dict(n_dream_suite_informative=int(len(ds)), counts=bill_ct.to_dict(), pct=bill_pct.to_dict(),
                        cluster_n=bill_ct.sum(axis=0).to_dict())
bill_ct.to_csv("out/bill_instruction_counts.csv")

# ------------------------------------------------------------------ 5. cluster x villa, year composition
cv = pd.crosstab(feats.villa, feats.cluster)
R["cluster_by_villa_counts"] = cv.to_dict()
R["cluster_by_villa_rowpct"] = (cv.div(cv.sum(axis=1), axis=0) * 100).round(1).to_dict()
cy = pd.crosstab(feats.booking_year, feats.cluster)
R["cluster_by_year_counts"] = cy.to_dict()
R["cluster_by_year_rowpct"] = (cy.div(cy.sum(axis=1), axis=0) * 100).round(1).to_dict()
R["year_totals"] = feats.booking_year.value_counts().sort_index().to_dict()
R["villa_by_year"] = pd.crosstab(feats.booking_year, feats.villa).to_dict()
for col in ["booking_platform_type", "nationality_region"]:
    t = pd.crosstab(feats.booking_year, feats[col])
    R[f"{col}_by_year_rowpct"] = (t.div(t.sum(axis=1), axis=0) * 100).round(1).to_dict()
cv.to_csv("out/cluster_by_villa.csv"); cy.to_csv("out/cluster_by_year.csv")

# ------------------------------------------------------------------ 6. centre distances and cluster heterogeneity
cd = np.sqrt(((cntr[:, None, :] - cntr[None, :, :]) ** 2).sum(axis=2))
R["centre_distances"] = np.round(cd, 4).tolist()
mx = u.max(axis=0)
het = {}
for k in range(3):
    sub = feats[feats.cluster == k]
    def H(col):
        pk = sub[col].value_counts(normalize=True).values
        return float(-(pk * np.log2(pk)).sum())
    het[f"C{k}"] = dict(channel_entropy_bits=round(H("booking_platform_type"), 3),
                        region_entropy_bits=round(H("nationality_region"), 3),
                        season_entropy_bits=round(H("booking_season"), 3),
                        mean_max_membership=round(float(mx[lab == k].mean()), 3),
                        strong=round(100 * (mx[lab == k] >= 0.7).mean(), 1),
                        moderate=round(100 * ((mx[lab == k] >= 0.5) & (mx[lab == k] < 0.7)).mean(), 1),
                        ambiguous=round(100 * (mx[lab == k] < 0.5).mean(), 1),
                        channel_mix=(sub.booking_platform_type.value_counts(normalize=True) * 100).round(1).to_dict())
R["cluster_heterogeneity"] = het
amb = feats[mx < 0.5]; strong = feats[mx >= 0.7]
R["ambiguous_vs_strong"] = dict(
    ambiguous_channel=(amb.booking_platform_type.value_counts(normalize=True) * 100).round(1).to_dict(),
    strong_channel=(strong.booking_platform_type.value_counts(normalize=True) * 100).round(1).to_dict(),
    ambiguous_season=(amb.booking_season.value_counts(normalize=True) * 100).round(1).to_dict(),
    strong_season=(strong.booking_season.value_counts(normalize=True) * 100).round(1).to_dict())

# ------------------------------------------------------------------ 7. bootstrap stability (Hennig-style Jaccard)
B = 100
rng = np.random.default_rng(2026)
aris, jac = [], {k: [] for k in range(3)}
for b in range(B):
    idx = rng.integers(0, n, n)
    pc_b = PCA(n_components=8, random_state=42).fit(X[idx])
    Zb = pc_b.transform(X[idx])
    c_b, u_b, jm_b, it_b, _ = run_fcm(Zb, seed=42)
    u_all, *_ = fuzz.cluster.cmeans_predict(pc_b.transform(X).T, c_b, 2.0, error=1e-5, maxiter=1000)
    l_all = match(lab, np.argmax(u_all, axis=0))
    aris.append(ARI(lab, l_all))
    for k in range(3):
        A = set(np.where(lab == k)[0]); Bk = set(np.where(l_all == k)[0])
        jac[k].append(len(A & Bk) / len(A | Bk) if A | Bk else 0)
R["bootstrap"] = dict(B=B, ARI_mean=round(float(np.mean(aris)), 4), ARI_sd=round(float(np.std(aris)), 4),
                      ARI_min=round(float(np.min(aris)), 4),
                      jaccard={f"C{k}": dict(mean=round(float(np.mean(v)), 3), sd=round(float(np.std(v)), 3),
                                             min=round(float(np.min(v)), 3)) for k, v in jac.items()})
print("bootstrap done", round(time.time() - T0, 1))

# ------------------------------------------------------------------ 8. temporal hold-out (train 2019-2023, test 2024)
yr = feats.booking_year.values
tr, te = yr <= 2023, yr == 2024
pc_t = PCA(n_components=8, random_state=42).fit(X[tr])
c_t, u_t, *_ = run_fcm(pc_t.transform(X[tr]), seed=42)
u_te, *_ = fuzz.cluster.cmeans_predict(pc_t.transform(X[te]).T, c_t, 2.0, error=1e-5, maxiter=1000)
# map the training-only clusters to the published labels using the training rows
u_tr, *_ = fuzz.cluster.cmeans_predict(pc_t.transform(X[tr]).T, c_t, 2.0, error=1e-5, maxiter=1000)
l_tr_raw = np.argmax(u_tr, axis=0)
cm = pd.crosstab(pd.Series(lab[tr]), pd.Series(l_tr_raw)).reindex(index=range(3), columns=range(3), fill_value=0).values
r_, c_ = linear_sum_assignment(-cm); mp = dict(zip(c_, r_))
l_te = np.array([mp[x] for x in np.argmax(u_te, axis=0)])
R["temporal_holdout"] = dict(train_rows=int(tr.sum()), test_rows=int(te.sum()),
                             ARI_test_vs_published=round(ARI(lab[te], l_te), 4),
                             agreement_test=round(float((lab[te] == l_te).mean()), 4),
                             test_shares_holdout_model=(pd.Series(l_te).value_counts(normalize=True).sort_index() * 100).round(1).tolist(),
                             test_shares_published=(pd.Series(lab[te]).value_counts(normalize=True).sort_index() * 100).round(1).tolist())
# dashboard consistency: stored centres re-predicting the training data
u_re, *_ = fuzz.cluster.cmeans_predict(Xp.T, cntr, 2.0, error=1e-5, maxiter=1000)
R["dashboard_repredict_agreement"] = round(float((np.argmax(u_re, axis=0) == lab).mean()), 4)

# ------------------------------------------------------------------ 9. leave-one-villa-out
lovo = []
for v in ["Dream Suite", "Kembali", "The Kumpi", "Bugis"]:
    keep = (feats.villa != v).values
    pc_v = PCA(n_components=8, random_state=42).fit(X[keep])
    c_v, u_v, *_ = run_fcm(pc_v.transform(X[keep]), seed=42)
    u_allv, *_ = fuzz.cluster.cmeans_predict(pc_v.transform(X).T, c_v, 2.0, error=1e-5, maxiter=1000)
    l_allv = match(lab, np.argmax(u_allv, axis=0))
    out_rows = ~keep
    lovo.append(dict(villa_left_out=v, train_rows=int(keep.sum()),
                     ARI_remaining=round(ARI(lab[keep], l_allv[keep]), 4),
                     agreement_remaining=round(float((lab[keep] == l_allv[keep]).mean()), 4),
                     agreement_left_out_rows=round(float((lab[out_rows] == l_allv[out_rows]).mean()), 4),
                     ARI_all=round(ARI(lab, l_allv), 4)))
R["leave_one_villa_out"] = lovo
print("lovo done", round(time.time() - T0, 1))

# ------------------------------------------------------------------ 10. balanced subsampling without synthetic data
sub_res = []
ds_idx = np.where(feats.villa.values == "Dream Suite")[0]
other_idx = np.where(feats.villa.values != "Dream Suite")[0]
for target in [2248, 5000]:
    a = []; ag = []
    for rep in range(30):
        rr = np.random.default_rng(rep)
        # stratified by booking year
        pick = []
        ds_years = feats.booking_year.values[ds_idx]
        for y in np.unique(ds_years):
            pool = ds_idx[ds_years == y]
            take = int(round(target * len(pool) / len(ds_idx)))
            pick.extend(rr.choice(pool, size=min(take, len(pool)), replace=False))
        idx = np.concatenate([np.array(pick), other_idx])
        pc_s = PCA(n_components=8, random_state=42).fit(X[idx])
        c_s, u_s, *_ = run_fcm(pc_s.transform(X[idx]), seed=42)
        u_alls, *_ = fuzz.cluster.cmeans_predict(pc_s.transform(X).T, c_s, 2.0, error=1e-5, maxiter=1000)
        l_alls = match(lab, np.argmax(u_alls, axis=0))
        a.append(ARI(lab, l_alls)); ag.append((lab == l_alls).mean())
    sub_res.append(dict(dream_suite_rows=target, total_rows=int(target + len(other_idx)), repeats=30,
                        ARI_mean=round(float(np.mean(a)), 4), ARI_sd=round(float(np.std(a)), 4),
                        ARI_min=round(float(np.min(a)), 4), agreement_mean=round(float(np.mean(ag)), 4)))
R["balanced_subsampling"] = sub_res
print("subsampling done", round(time.time() - T0, 1))

# ------------------------------------------------------------------ 11. baselines
base = {}
km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(Xp)
base["K-Means (PCA-8)"] = km.labels_
gm = GaussianMixture(n_components=3, covariance_type="full", random_state=42, n_init=5).fit(Xp)
base["GMM (PCA-8)"] = gm.predict(Xp)
try:
    from kmodes.kprototypes import KPrototypes
    raw = feats[["pax", "length_of_stay", "nationality_region", "booking_season", "booking_platform_type"]].copy()
    raw["pax"] = (raw.pax - raw.pax.min()) / (raw.pax.max() - raw.pax.min())
    raw["length_of_stay"] = (raw.length_of_stay - raw.length_of_stay.min()) / (raw.length_of_stay.max() - raw.length_of_stay.min())
    kp = KPrototypes(n_clusters=3, init="Cao", n_init=5, random_state=42, n_jobs=1)
    base["k-prototypes (mixed raw)"] = kp.fit_predict(raw.values, categorical=[2, 3, 4])
    R["kproto_gamma"] = float(kp.gamma)
except Exception as e:
    R["kproto_error"] = str(e)
try:
    from stepmix.stepmix import StepMix
    cat = feats[["nationality_region", "booking_season", "booking_platform_type"]].apply(lambda s: s.astype("category").cat.codes)
    num = feats[["pax", "length_of_stay"]].astype(float)
    Xl = pd.concat([cat, num], axis=1)
    desc = {"cat": {"model": "categorical", "n_columns": 3}, "num": {"model": "gaussian_diag", "n_columns": 2}}
    lca = StepMix(n_components=3, measurement=desc, n_init=5, random_state=42, verbose=0, progress_bar=0)
    lca.fit(Xl)
    base["Latent class model (mixed)"] = lca.predict(Xl)
except Exception as e:
    R["lca_error"] = str(e)
rule = np.where(feats.booking_platform_type.values == "DMC",
                np.where(feats.booking_season.values == "Peak", 1, 2), 0)
base["Rule-based (DMC x peak)"] = rule
rule4 = (feats.booking_platform_type.values == "DMC").astype(int) * 2 + (feats.booking_season.values == "Peak").astype(int)
base["Rule-based 4-cell (channel x season)"] = rule4

sil_idx = np.random.default_rng(7).choice(n, 8000, replace=False)
brow = []
for name, l_ in [("Fuzzy C-Means (PCA-8)", lab)] + list(base.items()):
    l_ = np.asarray(l_)
    try:
        sil = silhouette_score(Xp[sil_idx], l_[sil_idx]) if len(np.unique(l_)) > 1 else np.nan
    except Exception:
        sil = np.nan
    vv = cramers_v(feats.villa, l_)[3]
    vy = cramers_v(feats.booking_year, l_)[3]
    dsm = (feats.villa == "Dream Suite") & (~feats.bill_category.isin(["Room_Type_Leak", "Unknown"]))
    vb = cramers_v(feats.loc[dsm, "bill_category"], l_[dsm.values])[3]
    brow.append(dict(method=name, sizes=sorted(np.bincount(l_).tolist(), reverse=True),
                     ARI_vs_FCM=round(ARI(lab, l_), 4), NMI_vs_FCM=round(NMI(lab, l_), 4),
                     silhouette_PCA8=round(float(sil), 4),
                     V_villa=round(vv, 3), V_year=round(vy, 3), V_bill=round(vb, 3)))
R["baselines"] = brow
pd.DataFrame(brow).to_csv("out/baselines.csv", index=False)
R["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open("out/a3_results.json", "w"), indent=2, default=lambda o: o if not isinstance(o, (np.integer, np.floating)) else o.item())
print("ALL DONE", R["runtime_s"])
