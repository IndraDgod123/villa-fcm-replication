# -*- coding: utf-8 -*-
"""Re-implementation of the thesis pipeline (build_features -> encode_scale -> PCA -> FCM),
ported in logic from the thesis source code."""
import os
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.decomposition import PCA
os.makedirs("out", exist_ok=True); os.makedirs("fig", exist_ok=True)

PLATFORM_TYPE_MAPPING = {
    "Agoda": "International_OTA", "Booking.com": "International_OTA",
    "Expedia": "International_OTA", "Airbnb": "International_OTA",
    "Ctrip": "International_OTA", "Hotelbeds": "International_OTA",
    "WebBeds": "International_OTA", "TBO Holidays": "International_OTA",
    "Traveloka": "Local_OTA", "Tiket.com": "Local_OTA",
    "U&I Holidays": "DMC", "Go-MMT": "DMC", "Goibibo": "DMC",
    "STS Travel": "DMC", "LH Travel": "DMC", "Travelling Compass": "DMC",
    "TravClan": "DMC", "Trip Mantra": "DMC", "Pick Your Trail": "DMC",
    "Boeing Holidays": "DMC", "Tokar Holidays": "DMC", "Roland Holidays": "DMC",
    "Holidays Booker": "DMC", "Apple Holidays": "DMC", "Vacatis World": "DMC",
    "Olympia Europe": "DMC", "Asian Trail": "DMC", "Red Apple": "DMC",
    "G2 Travel": "DMC", "Klik & Book": "DMC", "Ramayana Travel": "DMC",
    "One Above": "DMC", "Truly Asia Tour": "DMC", "Peak Season Holidays": "DMC",
    "Luxury Indo Travels": "DMC", "Travel Hangar": "DMC",
    "Travel Book Online": "DMC", "Swift Book": "DMC", "ICS Travel": "DMC",
    "1Trip": "DMC",
    "Pacto": "Local_Agent", "Bali Suci": "Local_Agent",
    "Indojaya Travel": "Local_Agent", "Exotic Bali": "Local_Agent",
    "Bali Sundaram": "Local_Agent", "Bali Lestari": "Local_Agent",
    "BLM Tour": "Local_Agent", "Indoplan": "Local_Agent",
    "CCI Travel": "Local_Agent", "Absolute Indonesia": "Local_Agent",
    "Nice Bali Tour": "Local_Agent", "Real Bali Holidays": "Local_Agent",
    "Restu Dewata": "Local_Agent", "Mimpi Tours": "Local_Agent",
    "Indo Pacific DMC": "Local_Agent", "Bali Trip Wisata": "Local_Agent",
    "Khrisna Dewata": "Local_Agent", "Bali Villa International": "Local_Agent",
    "Yorker Bali DMC": "Local_Agent", "DMC Asiana Pacific": "Local_Agent",
    "Bali Eyes Property": "Local_Agent", "Panorama Group": "Local_Agent",
    "Stay in Bali": "Local_Agent", "Kirana Bali": "Local_Agent",
    "Triauma": "Local_Agent", "KIP Tours": "Local_Agent",
    "Pandawa Lima Tour": "Local_Agent", "STAT Indonesia Tours": "Local_Agent",
    "Holidays Sketcher": "Local_Agent", "Golden Inara": "Local_Agent",
    "Alia Travel": "Local_Agent", "Aneka Kartika Tours": "Local_Agent",
    "DAsia Travels": "Local_Agent", "Nexus DMC": "Local_Agent",
    "Orchid DMC": "Local_Agent", "Swasti Bali Tours": "Local_Agent",
    "Bali Java": "Local_Agent", "TBB": "Local_Agent",
    "Walk-in": "Direct", "Direct": "Direct", "Agent (Other)": "Direct",
    "Group Note": "Direct", "Complimentary": "Direct",
    "OTA (Other)": "International_OTA",
}

NATIONALITY_REGION_MAPPING = {
    "Indonesia": "Domestic",
    "India": "Asia", "Malaysia": "Asia", "Singapore": "Asia",
    "China": "Asia", "Japan": "Asia", "Korea": "Asia",
    "Thailand": "Asia", "Philippines": "Asia", "Vietnam": "Asia",
    "Nepal": "Asia",
    "Australia": "Oceania", "New Zealand": "Oceania",
    "USA": "Americas", "Canada": "Americas",
    "UK": "Europe", "France": "Europe", "Germany": "Europe",
    "Italy": "Europe", "Spain": "Europe", "Russia": "Europe",
    "Netherlands": "Europe", "Switzerland": "Europe",
    "Ireland": "Europe", "Denmark": "Europe", "Sweden": "Europe",
    "Saudi Arabia": "Middle_East", "UAE": "Middle_East",
    "Middle East": "Middle_East",
    "Egypt": "Africa", "Tunisia": "Africa",
    "Ethiopia": "Africa", "South Africa": "Africa",
    "Unknown": "Unknown",
}

NUMERIC = ["pax", "length_of_stay"]
CATEG = ["nationality_region", "booking_season", "booking_platform_type"]


def build_features(df):
    df = df.copy()
    for col in ["nationality", "booking_platform", "room_type"]:
        df[col] = df[col].fillna("Unknown")
    for col in NUMERIC:
        df[col] = df.groupby("villa")[col].transform(lambda s: s.fillna(s.median()))
        df[col] = df[col].fillna(df[col].median())
    df["date_in"] = pd.to_datetime(df["date_in"], errors="coerce")
    df["booking_year"] = df["date_in"].dt.year
    df["booking_month"] = df["date_in"].dt.month

    def season(m):
        if pd.isna(m):
            return "Unknown"
        m = int(m)
        if m in (6, 7, 8, 12, 1):
            return "Peak"
        if m in (4, 5, 9, 10, 11):
            return "Shoulder"
        return "Low"
    df["booking_season"] = df["booking_month"].apply(season)

    def gcat(p):
        if pd.isna(p): return "Unknown"
        p = float(p)
        if p <= 1: return "Solo"
        if p <= 2: return "Couple"
        if p <= 4: return "Small_Group"
        return "Large_Group"
    df["group_size_category"] = df["pax"].apply(gcat)

    def lcat(l):
        if pd.isna(l): return "Unknown"
        l = float(l)
        if l <= 2: return "Short"
        if l <= 7: return "Medium"
        return "Long_Stay"
    df["length_category"] = df["length_of_stay"].apply(lcat)

    df["booking_platform_type"] = df["booking_platform"].map(PLATFORM_TYPE_MAPPING).fillna("Other")
    df["nationality_region"] = df["nationality"].map(NATIONALITY_REGION_MAPPING).fillna("Other")
    return df


def encode(df, categ=CATEG, numeric=NUMERIC):
    out = pd.DataFrame(index=df.index)
    for col in numeric:
        s = df[col].astype(float)
        rng = s.max() - s.min()
        out[f"{col}_scaled"] = 0.0 if rng == 0 else (s - s.min()) / rng
    for col in categ:
        d = pd.get_dummies(df[col], prefix=col).astype(int)
        out = pd.concat([out, d], axis=1)
    return out


def run_fcm(X, k=3, m=2.0, seed=42, error=1e-5, maxiter=1000):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X.T, c=k, m=m, error=error, maxiter=maxiter, seed=seed)
    return cntr, u, jm, p, fpc


def validity(X, cntr, u, m=2.0):
    """PC, PE, XB, FS (Fukuyama-Sugeno), Kwon, and minimum squared centre distance."""
    n = X.shape[0]
    c = cntr.shape[0]
    d2 = ((X[None, :, :] - cntr[:, None, :]) ** 2).sum(axis=2)  # c x n
    um = u ** m
    pc = float((u ** 2).sum() / n)
    uu = np.clip(u, 1e-12, 1)
    pe = float(-(u * np.log(uu)).sum() / n)
    cd2 = ((cntr[:, None, :] - cntr[None, :, :]) ** 2).sum(axis=2)
    iu = np.triu_indices(c, 1)
    dmin = float(cd2[iu].min())
    xb = float((um * d2).sum() / (n * dmin)) if dmin > 0 else np.inf
    vbar = X.mean(axis=0)
    cb2 = ((cntr - vbar) ** 2).sum(axis=1)  # c
    fs = float((um * (d2 - cb2[:, None])).sum())
    kwon = float(((u ** 2) * d2).sum() + cb2.mean()) / dmin if dmin > 0 else np.inf
    return dict(PC=pc, PE=pe, XB=xb, FS=fs, Kwon=kwon, min_centre_dist2=dmin)


def full_pipeline(master, n_pca=8, k=3, m=2.0, seed=42):
    feats = build_features(master)
    Xdf = encode(feats)
    pca = PCA(n_components=n_pca, random_state=seed)
    Xp = pca.fit_transform(Xdf.values.astype(float))
    cntr, u, jm, p, fpc = run_fcm(Xp, k=k, m=m, seed=seed)
    labels = np.argmax(u, axis=0)
    return feats, Xdf, pca, Xp, cntr, u, labels, p, fpc
