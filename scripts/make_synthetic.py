# -*- coding: utf-8 -*-
"""Create a synthetic booking table with the same columns as master_data.csv.
Joint (villa, year, month, nationality, channel) combinations are resampled from the real table;
dates, room units, guest numbers and stay lengths are re-drawn; names are artificial."""
import numpy as np, pandas as pd
rng = np.random.default_rng(2026)
m = pd.read_csv("data/master_data.csv")
d = pd.to_datetime(m["date_in"], errors="coerce")
base = pd.DataFrame({"villa": m.villa, "y": d.dt.year, "mo": d.dt.month, "nationality": m.nationality,
                     "booking_platform": m.booking_platform, "room_type": m.room_type, "bill_instruction": m.bill_instruction,
                     "pax": m.pax, "los": m.length_of_stay})
s = base.sample(len(base), replace=True, random_state=7).reset_index(drop=True)
day = rng.integers(1, 29, len(s))
din = pd.to_datetime(dict(year=s.y.fillna(2023).astype(int), month=s.mo.fillna(6).astype(int), day=day))
los = np.clip(np.round(s.los.fillna(3) + rng.integers(-1, 2, len(s))), 1, 30).astype(int)
pax = np.clip(np.round(s.pax.fillna(2) + rng.choice([0, 0, 0, 1, -1], len(s))), 1, 12)
pax = pd.Series(pax).where(~s.pax.isna(), np.nan)
syn = pd.DataFrame({"villa": s.villa, "room_unit": ["R" + str(x) for x in rng.integers(1, 30, len(s))],
    "guest_name": [f"Guest {i:05d}" for i in range(len(s))], "nationality": s.nationality, "pax": pax,
    "date_in": din.dt.strftime("%Y-%m-%d"), "date_out": (din + pd.to_timedelta(los, unit="D")).dt.strftime("%Y-%m-%d"),
    "length_of_stay": los, "booking_platform": s.booking_platform, "room_type": s.room_type, "bedrooms": np.nan,
    "has_breakfast": np.nan, "bill_instruction": s.bill_instruction, "source_file": "synthetic", "source_type": "synthetic"})
syn.to_csv("synthetic_master_data.csv", index=False); print(syn.shape)
