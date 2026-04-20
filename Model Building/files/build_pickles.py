"""
build_pickles.py
----------------
Run ONCE locally to fit both models and write linear_model.pkl /
nonlinear_model.pkl for hand-off to the UI team.

Usage:
    python build_pickles.py

Writes to the current directory:
    linear_model.pkl
    nonlinear_model.pkl
    MODEL_INPUTS.md            <- reference sheet for the UI team
    versions.txt               <- package versions used when fitting

This script mirrors the logic in Final_Models.qmd exactly (same
random_state, same Lasso-selected features, same LightGBM hyperparameters).
"""

from __future__ import annotations

import pickle
import re
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_absolute_error
import lightgbm as lgb

from price_predictors import LinearPricePredictor, NonlinearPricePredictor, _clean_name

RS = 2026
DEFAULT_CSV = "https://raw.githubusercontent.com/RMMcMurray/Math-4810_Capstone/main/data/final_adjusted.csv"

# Allow overriding the CSV path (local file OR alternate URL) via --csv argument.
CSV_PATH = DEFAULT_CSV
if "--csv" in sys.argv:
    CSV_PATH = sys.argv[sys.argv.index("--csv") + 1]


# ---------------------------------------------------------------------------
# 1. Load + shared cleaning   (copied verbatim from the notebook)
# ---------------------------------------------------------------------------
print(f"Loading data from: {CSV_PATH}")
raw = pd.read_csv(CSV_PATH)
df_shared = raw.copy()

for c in ["Land Value", "Total Value"]:
    if c in df_shared.columns:
        s = df_shared[c].astype(str).str.replace(r"[\$,\s]", "", regex=True)
        df_shared[c] = pd.to_numeric(s, errors="coerce").astype("float64")

DROP_COMMON = [
    "Sold Price", "Sold Date", "Concessions Amount", "Age", "Adj_Price",
    "ln_Price", "Time_1", "Time_2", "Time_3",
    "Count_DetGarage", "Count_ DetCarport", "Count_Barn",
    "Count_Guest_House", "Count_Pools", "Count_Recreational_Courts",
    "Count_Shed", "Count_Gazebo_Pavilion",
]
df_shared = df_shared.drop(columns=[c for c in DROP_COMMON if c in df_shared.columns])
df_shared = df_shared.reset_index(drop=True)
print(f"  Shared frame: {df_shared.shape[0]:,} rows x {df_shared.shape[1]} cols")


# ===========================================================================
# 2. LINEAR MODEL
# ===========================================================================
print("\n=== Fitting LINEAR model ===")
df_lin = df_shared.copy()
df_lin["ln_Trended_Price"] = np.log(df_lin["Trended_Price"])

df_lin["AsinH_Acreage"]                  = np.arcsinh(df_lin["Acreage"])
df_lin["AsinH_Land_Value"]               = np.arcsinh(df_lin["Land Value"])
df_lin["AsinH_Total_Value"]              = np.arcsinh(df_lin["Total Value"])
df_lin["AsinH_Tot_Bsmt"]                 = np.arcsinh(df_lin["Tot Bsmt"])
df_lin["AsinH_BsmtFin_Pct_Wghted"]       = np.arcsinh(df_lin["BsmtFinishPct_Weighted"])
df_lin["AsinH_Carport_Area"]             = np.arcsinh(df_lin["CarportArea"])
df_lin["AsinH_Garage_Area"]              = np.arcsinh(df_lin["GarageArea"])
df_lin["AsinH_SqFt_Det_Garage"]          = np.arcsinh(df_lin["SqFt_DetGarage"])
df_lin["AsinH_SqFt_Det_Carport"]         = np.arcsinh(df_lin["SqFt_DetCarport"])
df_lin["AsinH_SqFt_Barn"]                = np.arcsinh(df_lin["Sqft_Barn"])
df_lin["AsinH_SqFt_Guest_House"]         = np.arcsinh(df_lin["SqFt_Guest_House"])
df_lin["AsinH_SqFt_Pools"]               = np.arcsinh(df_lin["SqFt_Pools"])
df_lin["AsinH_SqFt_Recreational_Courts"] = np.arcsinh(df_lin["SqFt_Recreational_Courts"])
df_lin["AsinH_SqFt_Shed"]                = np.arcsinh(df_lin["SqFt_Shed"])
df_lin["AsinH_SqFt_Gazebo_Pavilion"]     = np.arcsinh(df_lin["SqFt_Gazebo_Pavilion"])

DROP_LIN = [
    "Trended_Price", "TotGLA", "Acreage", "Land Value", "Total Value", "Tot Bsmt",
    "BsmtFinishPct_Weighted", "CarportArea", "GarageArea",
    "SqFt_DetGarage", "SqFt_DetCarport", "Sqft_Barn", "SqFt_Guest_House",
    "SqFt_Pools", "SqFt_Recreational_Courts", "SqFt_Shed", "SqFt_Gazebo_Pavilion",
]
df_lin = df_lin.drop(columns=[c for c in DROP_LIN if c in df_lin.columns])

DUMMY_COLS = ["PropTypeDescription", "SpecificPropTypeDescription",
              "NbhdCode2", "Main_StyleDesc", "segment"]
df_lin_enc = df_lin.copy()
for col in DUMMY_COLS:
    if col in df_lin_enc.columns:
        d = pd.get_dummies(df_lin_enc[col], prefix=col, drop_first=True, dtype=int)
        df_lin_enc = pd.concat([df_lin_enc.drop(columns=col), d], axis=1)
df_lin_enc = df_lin_enc.fillna(0)

y_lin_all = df_lin_enc["ln_Trended_Price"].astype(float)
X_lin_all = (df_lin_enc.select_dtypes(include=[np.number])
             .drop(columns=["ln_Trended_Price"]).astype(float))

# Outlier drop
base = sm.OLS(y_lin_all, sm.add_constant(X_lin_all)).fit()
infl = OLSInfluence(base)
drop_mask = ((infl.cooks_distance[0] > 4.0 / len(y_lin_all))
             | (np.abs(infl.resid_studentized_internal) > 3.0))

X_keep = X_lin_all[~drop_mask].reset_index(drop=True)
y_keep = y_lin_all[~drop_mask].reset_index(drop=True)
seg_keep = df_lin.loc[~drop_mask, "segment"].reset_index(drop=True)

X_tr_lin, X_te_lin, y_tr_lin, y_te_lin = train_test_split(
    X_keep, y_keep, test_size=0.2, random_state=RS)
seg_te_lin = seg_keep.loc[X_te_lin.index].reset_index(drop=True)
seg_tr_lin = seg_keep.loc[X_tr_lin.index].reset_index(drop=True)
X_tr_lin = X_tr_lin.reset_index(drop=True)
X_te_lin = X_te_lin.reset_index(drop=True)
y_tr_lin = y_tr_lin.reset_index(drop=True)
y_te_lin = y_te_lin.reset_index(drop=True)

# Poly + seg-interaction feature build
POLY_COLS = ["ln_GLA", "Quality_Weighted", "EffYearBuilt_Weighted", "AsinH_Acreage"]
KEY_NUMERIC = ["ln_GLA", "Quality_Weighted", "EffYearBuilt_Weighted",
               "AsinH_Acreage", "AsinH_Land_Value", "AsinH_Total_Value"]
SEG_COLS = [c for c in X_tr_lin.columns if c.startswith("segment_")]

def build_poly_seg(X_in, ref_cols=None):
    X2 = X_in.copy()
    for c in POLY_COLS:
        if c in X2.columns:
            X2[f"{c}_sq"] = X2[c] ** 2
    for sc in SEG_COLS:
        for kn in KEY_NUMERIC:
            if kn in X2.columns:
                X2[f"{sc}_x_{kn}"] = X2[sc] * X2[kn]
    if ref_cols is not None:
        X2 = X2.reindex(columns=ref_cols, fill_value=0)
    return X2

X_tr_v4 = build_poly_seg(X_tr_lin)
X_te_v4 = build_poly_seg(X_te_lin, ref_cols=X_tr_v4.columns)

# Lasso select -> OLS refit
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(X_tr_v4)
lasso = LassoCV(alphas=np.logspace(-4, 0, 40), cv=5,
                max_iter=20000, random_state=RS).fit(Xtr_s, y_tr_lin)
sel_names = X_tr_v4.columns[np.where(lasso.coef_ != 0)[0]].tolist()

Xtr_sel = X_tr_v4[sel_names]
Xte_sel = X_te_v4[sel_names]
Xc_tr = sm.add_constant(Xtr_sel, has_constant="add")
Xc_te = (sm.add_constant(Xte_sel, has_constant="add")
         .reindex(columns=Xc_tr.columns, fill_value=0))

linear_model = sm.OLS(y_tr_lin, Xc_tr).fit()

yhat_tr = linear_model.predict(Xc_tr)
yhat_te = linear_model.predict(Xc_te)
r2_tr_lin = r2_score(y_tr_lin, yhat_tr)
r2_te_lin = r2_score(y_te_lin, yhat_te)
mae_te_lin = mean_absolute_error(y_te_lin, yhat_te)
dmae_te_lin = np.exp(y_te_lin.values).mean() * (np.exp(mae_te_lin) - 1)

print(f"  Lasso alpha        : {lasso.alpha_:.5f}")
print(f"  Features selected  : {len(sel_names)} / {X_tr_v4.shape[1]}")
print(f"  Train R^2          : {r2_tr_lin:.4f}")
print(f"  Test R^2           : {r2_te_lin:.4f}")
print(f"  Approx $ MAE       : ${dmae_te_lin:,.0f}")

# Segment defaults on the TRANSFORMED scale
LIN_SEG_DEFAULTS = {
    s: X_tr_lin.loc[seg_tr_lin == s].median().to_dict()
    for s in seg_keep.unique()
}
LIN_GLOBAL_DEFAULTS = X_tr_lin.median().to_dict()

# PI quantiles per segment
_res_global_lin = y_te_lin.values - linear_model.predict(Xc_te).values
_q_global_lin = tuple(np.quantile(_res_global_lin, [0.025, 0.975]))
LIN_PI_Q = {}
for s in seg_te_lin.unique():
    m = (seg_te_lin == s).values
    if m.sum() >= 20:
        r = y_te_lin[m].values - linear_model.predict(Xc_te[m]).values
        LIN_PI_Q[s] = tuple(np.quantile(r, [0.025, 0.975]))
    else:
        LIN_PI_Q[s] = _q_global_lin

LIN_TRANSFORMS = {
    "GLA":                      ("ln_GLA",                        "log"),
    "Acreage":                  ("AsinH_Acreage",                 "arcsinh"),
    "Land Value":               ("AsinH_Land_Value",              "arcsinh"),
    "Total Value":              ("AsinH_Total_Value",             "arcsinh"),
    "Tot Bsmt":                 ("AsinH_Tot_Bsmt",                "arcsinh"),
    "BsmtFinishPct_Weighted":   ("AsinH_BsmtFin_Pct_Wghted",      "arcsinh"),
    "CarportArea":              ("AsinH_Carport_Area",            "arcsinh"),
    "GarageArea":               ("AsinH_Garage_Area",             "arcsinh"),
    "SqFt_DetGarage":           ("AsinH_SqFt_Det_Garage",         "arcsinh"),
    "SqFt_DetCarport":          ("AsinH_SqFt_Det_Carport",        "arcsinh"),
    "Sqft_Barn":                ("AsinH_SqFt_Barn",               "arcsinh"),
    "SqFt_Guest_House":         ("AsinH_SqFt_Guest_House",        "arcsinh"),
    "SqFt_Pools":               ("AsinH_SqFt_Pools",              "arcsinh"),
    "SqFt_Recreational_Courts": ("AsinH_SqFt_Recreational_Courts","arcsinh"),
    "SqFt_Shed":                ("AsinH_SqFt_Shed",               "arcsinh"),
    "SqFt_Gazebo_Pavilion":     ("AsinH_SqFt_Gazebo_Pavilion",    "arcsinh"),
}
LIN_CATEGORICALS = {"PropTypeDescription", "SpecificPropTypeDescription",
                    "NbhdCode2", "Main_StyleDesc", "segment"}

linear_predictor = LinearPricePredictor(
    ols_result=linear_model,
    sel_names=sel_names,
    design_columns=list(Xc_tr.columns),
    base_columns=list(X_tr_lin.columns),
    v4_columns=list(X_tr_v4.columns),
    poly_cols=POLY_COLS,
    key_numeric=KEY_NUMERIC,
    seg_cols=SEG_COLS,
    lin_transforms=LIN_TRANSFORMS,
    lin_categoricals=LIN_CATEGORICALS,
    seg_defaults=LIN_SEG_DEFAULTS,
    global_defaults=LIN_GLOBAL_DEFAULTS,
    pi_quantiles=LIN_PI_Q,
    pi_global=_q_global_lin,
    metadata={
        "model_type": "Lasso-selected OLS on polynomial + segment-interaction features",
        "train_rows": int(X_tr_lin.shape[0]),
        "test_rows": int(X_te_lin.shape[0]),
        "test_r2": float(r2_te_lin),
        "test_dollar_mae": float(dmae_te_lin),
        "n_selected_features": int(len(sel_names)),
        "random_state": RS,
        "fitted_at": datetime.utcnow().isoformat() + "Z",
    },
)

with open("linear_model.pkl", "wb") as f:
    pickle.dump(linear_predictor, f, protocol=pickle.HIGHEST_PROTOCOL)
print("  -> linear_model.pkl")


# ===========================================================================
# 3. NON-LINEAR MODEL (LightGBM)
# ===========================================================================
print("\n=== Fitting NON-LINEAR model (LightGBM) ===")
df_nl = df_shared.copy()

if "TotGLA" in df_nl.columns:
    df_nl = df_nl.rename(columns={"TotGLA": "GLA"})

df_nl["ln_Trended_Price"] = np.log(df_nl["Trended_Price"])
df_nl = df_nl.drop(columns=["Trended_Price"])
if "ln_GLA" in df_nl.columns:
    df_nl = df_nl.drop(columns=["ln_GLA"])
df_nl = df_nl.reset_index(drop=True)

df_nl_enc = df_nl.copy()
for col in DUMMY_COLS:
    if col in df_nl_enc.columns:
        d = pd.get_dummies(df_nl_enc[col], prefix=col, drop_first=True, dtype=int)
        df_nl_enc = pd.concat([df_nl_enc.drop(columns=col), d], axis=1)
df_nl_enc = df_nl_enc.fillna(0)

y_nl_all = df_nl_enc["ln_Trended_Price"].astype(float)
X_nl_all = (df_nl_enc.select_dtypes(include=[np.number])
            .drop(columns=["ln_Trended_Price"]).astype(float))

# Clean column names for LightGBM
CLEAN_MAP, seen = {}, {}
for c in X_nl_all.columns:
    v = _clean_name(c)
    if v in seen:
        seen[v] += 1
        v = f"{v}_{seen[v]}"
    else:
        seen[v] = 0
    CLEAN_MAP[c] = v
X_nl_all = X_nl_all.rename(columns=CLEAN_MAP)
ORIG_OF = {v: k for k, v in CLEAN_MAP.items()}

# Outlier drop
base_nl = sm.OLS(y_nl_all, sm.add_constant(X_nl_all)).fit()
infl_nl = OLSInfluence(base_nl)
drop_nl = ((infl_nl.cooks_distance[0] > 4.0 / len(y_nl_all))
           | (np.abs(infl_nl.resid_studentized_internal) > 3.0))

X_nl_keep = X_nl_all[~drop_nl].reset_index(drop=True)
y_nl_keep = y_nl_all[~drop_nl].reset_index(drop=True)
seg_nl_keep = df_nl.loc[~drop_nl, "segment"].reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_nl_keep, y_nl_keep, test_size=0.2, random_state=RS)
seg_test = seg_nl_keep.loc[X_test.index].reset_index(drop=True)
X_train = X_train.reset_index(drop=True); X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True); y_test = y_test.reset_index(drop=True)

LGB_PARAMS = dict(
    num_leaves=45, learning_rate=0.1211, n_estimators=672,
    min_child_samples=13, subsample=0.9138, colsample_bytree=0.7861,
    reg_lambda=1.4174, random_state=RS, n_jobs=-1, verbose=-1,
)
nl_model = lgb.LGBMRegressor(**LGB_PARAMS).fit(X_train, y_train)

yhat_te_nl = nl_model.predict(X_test)
r2_te_nl = r2_score(y_test, yhat_te_nl)
mae_te_nl = mean_absolute_error(y_test, yhat_te_nl)
dmae_te_nl = np.exp(y_test.values).mean() * (np.exp(mae_te_nl) - 1)

print(f"  Test R^2          : {r2_te_nl:.4f}")
print(f"  Approx $ MAE      : ${dmae_te_nl:,.0f}")

# Segment defaults (raw NUMERIC feature names -> median)
NUM_FEATS = ["GLA", "EffYearBuilt_Weighted", "Acreage", "Tot Bsmt", "GarageArea",
             "CarportArea", "SqFt_DetGarage", "SqFt_DetCarport", "Sqft_Barn",
             "SqFt_Guest_House", "SqFt_Pools", "SqFt_Recreational_Courts",
             "SqFt_Shed", "SqFt_Gazebo_Pavilion", "FullBaths_Total",
             "HalfBaths_Total", "GarageCapacity",
             "Land Value", "Total Value", "Quality_Weighted",
             "BsmtFinishPct_Weighted"]
NUM_FEATS = [c for c in NUM_FEATS if c in df_nl.columns]

_df_nums = df_nl[NUM_FEATS].copy()
for c in NUM_FEATS:
    cl = _df_nums[c].astype(str).str.replace(r"[\$,\s]", "", regex=True)
    _df_nums[c] = pd.to_numeric(cl, errors="coerce").astype("float64")
_df_nums["segment"] = df_nl["segment"].values
NL_SEG_DEFAULTS = _df_nums.groupby("segment")[NUM_FEATS].median().to_dict(orient="index")

# PI quantiles per segment
_res_global_nl = y_test.values - nl_model.predict(X_test)
_q_global_nl = tuple(np.quantile(_res_global_nl, [0.025, 0.975]))
NL_PI_Q = {}
for s in seg_test.unique():
    m = (seg_test == s).values
    if m.sum() >= 20:
        r = y_test[m].values - nl_model.predict(X_test.loc[m])
        NL_PI_Q[s] = tuple(np.quantile(r, [0.025, 0.975]))
    else:
        NL_PI_Q[s] = _q_global_nl

NL_CATEGORICALS = {"PropTypeDescription", "SpecificPropTypeDescription",
                   "NbhdCode2", "Main_StyleDesc", "segment"}

nonlinear_predictor = NonlinearPricePredictor(
    lgb_model=nl_model,
    feature_columns=list(X_train.columns),
    clean_map=CLEAN_MAP,
    orig_of=ORIG_OF,
    num_feats=NUM_FEATS,
    nl_categoricals=NL_CATEGORICALS,
    seg_defaults=NL_SEG_DEFAULTS,
    pi_quantiles=NL_PI_Q,
    pi_global=_q_global_nl,
    metadata={
        "model_type": "LightGBM on raw features (tuned via RandomizedSearchCV, Model Building 5)",
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "test_r2": float(r2_te_nl),
        "test_dollar_mae": float(dmae_te_nl),
        "lgb_params": LGB_PARAMS,
        "random_state": RS,
        "fitted_at": datetime.utcnow().isoformat() + "Z",
    },
)

with open("nonlinear_model.pkl", "wb") as f:
    pickle.dump(nonlinear_predictor, f, protocol=pickle.HIGHEST_PROTOCOL)
print("  -> nonlinear_model.pkl")


# ===========================================================================
# 4. Sanity check - load the pickles back and run an example prediction
# ===========================================================================
print("\n=== Sanity check (reload + predict) ===")
with open("linear_model.pkl", "rb") as f:
    lin_reload = pickle.load(f)
with open("nonlinear_model.pkl", "rb") as f:
    nl_reload = pickle.load(f)

example = dict(
    segment="Residential",
    PropTypeDescription="Single Family Res",
    GLA=2400, EffYearBuilt_Weighted=2005, Acreage=0.25,
    **{"Tot Bsmt": 800}, GarageArea=500,
    FullBaths_Total=2, HalfBaths_Total=1, GarageCapacity=2,
    Quality_Weighted=4.0,
)
r_lin = lin_reload.predict(**example)
r_nl = nl_reload.predict(**example)
print(f"  Linear    : ${r_lin['pred']:>10,.0f}   "
      f"PI [${r_lin['pi_lo']:,.0f} - ${r_lin['pi_hi']:,.0f}]")
print(f"  LightGBM  : ${r_nl['pred']:>10,.0f}   "
      f"PI [${r_nl['pi_lo']:,.0f} - ${r_nl['pi_hi']:,.0f}]")


# ===========================================================================
# 5. Write versions.txt and MODEL_INPUTS.md
# ===========================================================================
import sklearn
import statsmodels

with open("versions.txt", "w") as f:
    f.write("# Package versions used to fit the pickled models.\n")
    f.write("# The UI environment should match these (or be close) to guarantee load.\n\n")
    f.write(f"python       {sys.version.split()[0]}\n")
    f.write(f"numpy        {np.__version__}\n")
    f.write(f"pandas       {pd.__version__}\n")
    f.write(f"scikit-learn {sklearn.__version__}\n")
    f.write(f"statsmodels  {statsmodels.__version__}\n")
    f.write(f"lightgbm     {lgb.__version__}\n")
print("  -> versions.txt")
print("\nNote: MODEL_INPUTS.md is hand-maintained (ship it alongside the pickles).")
print("\nDone.\n")
