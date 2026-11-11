# -*- coding: utf-8 -*-
"""A1-A2: missingness/Unknown by villa and residual-duplicate audit on the
de-duplicated master table. Guest names are never printed or saved."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from pipeline import build_features

master = pd.read_csv("data/master_data.csv")
feats = build_features(master)
res = {}

# ---- unit of analysis / basic facts ----
res["n_rows"] = int(len(master))
res["room_unit_missing"] = int(master["room_unit"].isna().sum())
res["room_unit_missing_by_villa"] = master.groupby("villa")["room_unit"].apply(lambda s: int(s.isna().sum())).to_dict()
res["guest_name_missing_by_villa"] = master.groupby("villa")["guest_name"].apply(lambda s: int(s.isna().sum())).to_dict()

# ---- missingness / Unknown by villa ----
rows = []
for v, g in master.groupby("villa"):
    fg = feats.loc[g.index]
    rows.append(dict(
        villa=v, n=len(g),
        nationality_missing=round(100 * g["nationality"].isna().mean(), 1),
        region_unknown=round(100 * (fg["nationality_region"] == "Unknown").mean(), 1),
        region_other=round(100 * (fg["nationality_region"] == "Other").mean(), 1),
        platform_missing=round(100 * g["booking_platform"].isna().mean(), 1),
        platform_other=round(100 * (fg["booking_platform_type"] == "Other").mean(), 1),
        pax_imputed=round(100 * g["pax"].isna().mean(), 1),
        los_imputed=round(100 * g["length_of_stay"].isna().mean(), 1),
    ))
miss = pd.DataFrame(rows)
tot = dict(villa="Total", n=len(master),
           nationality_missing=round(100 * master["nationality"].isna().mean(), 1),
           region_unknown=round(100 * (feats["nationality_region"] == "Unknown").mean(), 1),
           region_other=round(100 * (feats["nationality_region"] == "Other").mean(), 1),
           platform_missing=round(100 * master["booking_platform"].isna().mean(), 1),
           platform_other=round(100 * (feats["booking_platform_type"] == "Other").mean(), 1),
           pax_imputed=round(100 * master["pax"].isna().mean(), 1),
           los_imputed=round(100 * master["length_of_stay"].isna().mean(), 1))
miss = pd.concat([miss, pd.DataFrame([tot])], ignore_index=True)
miss.to_csv("out/missingness_by_villa.csv", index=False)
res["missingness_by_villa"] = miss.to_dict(orient="records")

# ---- residual duplicate audit under looser keys ----
m = master.copy()
m["date_in"] = pd.to_datetime(m["date_in"], errors="coerce")
m["date_out"] = pd.to_datetime(m["date_out"], errors="coerce")
keys = {
    "K0_original": ["villa", "room_unit", "date_in", "date_out", "pax", "guest_name"],
    "K1_no_name": ["villa", "room_unit", "date_in", "date_out", "pax"],
    "K2_no_pax": ["villa", "room_unit", "date_in", "date_out", "guest_name"],
    "K3_room_dates": ["villa", "room_unit", "date_in", "date_out"],
    "K4_no_room": ["villa", "date_in", "date_out", "pax", "guest_name"],
}
dup = {}
for name, k in keys.items():
    d = m.duplicated(subset=k, keep="first")
    dup[name] = dict(extra_rows_merged=int(d.sum()), pct=round(100 * d.mean(), 2),
                     by_villa=m[d].groupby("villa").size().to_dict())
res["residual_duplicates"] = dup

# characterise K3 collisions (same room unit, same dates) without exposing names
k3 = keys["K3_room_dates"]
grp = m[m.duplicated(subset=k3, keep=False)].copy()
grp["_g"] = grp.groupby(k3, dropna=False).ngroup()
summ = []
for gid, g in grp.groupby("_g"):
    names = g["guest_name"].fillna("").astype(str).str.strip().str.lower()
    plats = g["booking_platform"].fillna("").astype(str)
    summ.append(dict(size=len(g),
                     same_name=int(names.nunique() == 1),
                     same_platform=int(plats.nunique() == 1),
                     same_pax=int(g["pax"].nunique(dropna=False) == 1),
                     villa=g["villa"].iloc[0]))
summ = pd.DataFrame(summ)
res["k3_collision_groups"] = int(len(summ))
if len(summ):
    res["k3_collision_profile"] = dict(
        groups=int(len(summ)),
        rows=int(summ["size"].sum()),
        same_name_pct=round(100 * summ["same_name"].mean(), 1),
        same_platform_pct=round(100 * summ["same_platform"].mean(), 1),
        same_pax_pct=round(100 * summ["same_pax"].mean(), 1),
        by_villa=summ.groupby("villa").size().to_dict())

json.dump(res, open("out/a1_data_quality.json", "w"), indent=2, default=str)
print(json.dumps(res, indent=2, default=str))
