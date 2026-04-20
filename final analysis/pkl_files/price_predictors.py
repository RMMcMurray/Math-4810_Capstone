"""
price_predictors.py
-------------------
Self-contained predictor classes that wrap the fitted Linear (Lasso->OLS)
and LightGBM models with ALL preprocessing baked in.

The pickles produced by build_pickles.py contain an instance of
LinearPricePredictor and NonlinearPricePredictor, respectively.

Two public methods on each class:

    .predict(**raw_features) -> dict
        Dollar prediction + 95% PI + segment used.

    .explain(**raw_features) -> dict
        Same prediction PLUS a ranked list of per-feature contributions
        to log-price (and dollars), for waterfall-plot UIs. See
        EXPLANATIONS.md for the full schema.

IMPORTANT: this module (price_predictors.py) must be importable in the
teammate's environment. Ship it alongside the .pkl files.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _clean_name(name: str) -> str:
    """Scrub a column name so LightGBM accepts it (mirrors the training script)."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_")


def _shap_to_dollar_delta(base_log: float, shap_log: float) -> float:
    """
    Convert a log-price SHAP contribution to an approximate dollar delta.

    Math: if base log-price is L and a feature's SHAP is s, the multiplicative
    price impact is exp(s) and the dollar delta (relative to the base) is
        exp(L) * (exp(s) - 1)
    This is an approximation -- SHAP values are additive on the log scale, so
    per-feature dollar-deltas will not sum exactly to the final dollar
    prediction, but they are the right scale for UI bars and the ordering is
    faithful.
    """
    return float(np.exp(base_log) * (np.exp(shap_log) - 1.0))


# ---------------------------------------------------------------------------
# Linear model wrapper  (Lasso -> OLS on poly + segment interactions)
# ---------------------------------------------------------------------------

class LinearPricePredictor:
    """
    Wraps a fitted statsmodels OLS result plus every piece of preprocessing
    needed to turn raw feature kwargs into a dollar prediction.
    """

    def __init__(
        self,
        *,
        ols_result,
        sel_names,
        design_columns,
        base_columns,
        v4_columns,
        poly_cols,
        key_numeric,
        seg_cols,
        lin_transforms,
        lin_categoricals,
        seg_defaults,
        global_defaults,
        pi_quantiles,
        pi_global,
        metadata,
    ):
        self.ols_result = ols_result
        self.sel_names = list(sel_names)
        self.design_columns = list(design_columns)
        self.base_columns = list(base_columns)
        self.v4_columns = list(v4_columns)
        self.poly_cols = list(poly_cols)
        self.key_numeric = list(key_numeric)
        self.seg_cols = list(seg_cols)
        self.lin_transforms = dict(lin_transforms)
        self.lin_categoricals = set(lin_categoricals)
        self.seg_defaults = {k: dict(v) for k, v in seg_defaults.items()}
        self.global_defaults = dict(global_defaults)
        self.pi_quantiles = {k: tuple(v) for k, v in pi_quantiles.items()}
        self.pi_global = tuple(pi_global)
        self.metadata = dict(metadata)

    # --- internal helpers -------------------------------------------------
    @staticmethod
    def _apply_transform(fn_name, value):
        if fn_name == "log":
            return float(np.log(value))
        if fn_name == "arcsinh":
            return float(np.arcsinh(value))
        raise ValueError(f"Unknown transform: {fn_name}")

    def _build_poly_seg_row(self, base_row: pd.Series) -> pd.Series:
        row = base_row.copy()
        for c in self.poly_cols:
            if c in row.index:
                row[f"{c}_sq"] = row[c] ** 2
        for sc in self.seg_cols:
            for kn in self.key_numeric:
                if kn in row.index:
                    row[f"{sc}_x_{kn}"] = row[sc] * row[kn]
        out = pd.Series(0.0, index=self.v4_columns)
        shared = row.index.intersection(out.index)
        out.loc[shared] = row.loc[shared].astype(float).values
        return out

    def _set_categorical_dummy(self, row: pd.Series, prefix: str, value):
        if value is None:
            return
        target = f"{prefix}_{value}"
        if target in row.index:
            row[target] = 1

    def _prepare_row(self, feats: dict):
        """Build the processed design row from raw feature kwargs."""
        feats = dict(feats)  # don't mutate caller
        segment = feats.pop("segment", "Residential")

        defaults = self.seg_defaults.get(segment, self.global_defaults)
        row = pd.Series(defaults, index=self.base_columns).fillna(0.0).astype(float)

        for c in row.index:
            if c.startswith("segment_"):
                row[c] = 0.0
        if f"segment_{segment}" in row.index:
            row[f"segment_{segment}"] = 1.0

        for k, v in feats.items():
            if k in self.lin_categoricals:
                self._set_categorical_dummy(row, k, v)
            elif k in self.lin_transforms:
                col, fn = self.lin_transforms[k]
                if col in row.index:
                    row[col] = self._apply_transform(fn, v)
            elif k in row.index:
                row[k] = float(v)

        v4_row = self._build_poly_seg_row(row)
        sel_row = v4_row.reindex(self.sel_names, fill_value=0.0)
        design_row = pd.Series(0.0, index=self.design_columns)
        design_row["const"] = 1.0
        for c in self.sel_names:
            if c in design_row.index:
                design_row[c] = sel_row[c]
        return segment, design_row

    # --- public API --------------------------------------------------------
    def predict(self, **feats) -> dict:
        """
        Predict a dollar price.

        See MODEL_INPUTS.md for the full list of accepted keyword arguments.

        Returns a dict:
          pred          float  dollar point estimate
          pi_lo, pi_hi  float  95% prediction interval (dollars)
          log_pred      float  prediction on the log scale
          segment_used  str
          n_features    int    number of Lasso-selected features
        """
        segment, design_row = self._prepare_row(feats)
        log_pred = float(self.ols_result.predict(design_row.to_frame().T).iloc[0])
        q_lo, q_hi = self.pi_quantiles.get(segment, self.pi_global)
        return {
            "pred": float(np.exp(log_pred)),
            "pi_lo": float(np.exp(log_pred + q_lo)),
            "pi_hi": float(np.exp(log_pred + q_hi)),
            "log_pred": log_pred,
            "segment_used": segment,
            "n_features": len(self.sel_names),
        }

    def explain(self, top_n: int = 15, **feats) -> dict:
        """
        Predict AND return per-feature contributions for a waterfall plot.

        For the linear model, each contribution is (coefficient * feature_value)
        on the log scale -- an exact decomposition of the prediction around the
        intercept. These sum exactly to log_pred - base_value.

        See EXPLANATIONS.md for the full schema.
        """
        segment, design_row = self._prepare_row(feats)
        log_pred = float(self.ols_result.predict(design_row.to_frame().T).iloc[0])
        q_lo, q_hi = self.pi_quantiles.get(segment, self.pi_global)

        params = self.ols_result.params
        base_value = float(params.get("const", 0.0))

        contribs = []
        for feat_name in self.sel_names:
            if feat_name not in params.index:
                continue
            coef = float(params[feat_name])
            value = float(design_row.get(feat_name, 0.0))
            shap_log = coef * value
            if shap_log == 0.0:
                continue
            contribs.append({
                "feature": feat_name,
                "value": value,
                "coefficient": coef,
                "shap_log": shap_log,
                "shap_dollars": _shap_to_dollar_delta(base_value, shap_log),
            })

        contribs.sort(key=lambda d: -abs(d["shap_log"]))
        if top_n is not None:
            contribs = contribs[:top_n]

        return {
            "pred": float(np.exp(log_pred)),
            "pi_lo": float(np.exp(log_pred + q_lo)),
            "pi_hi": float(np.exp(log_pred + q_hi)),
            "log_pred": log_pred,
            "segment_used": segment,
            "explanation": {
                "method": "linear_coefficients",
                "base_value": base_value,
                "base_value_dollars": float(np.exp(base_value)),
                "contributions": contribs,
            },
        }


# ---------------------------------------------------------------------------
# Non-linear model wrapper  (LightGBM on raw untransformed features)
# ---------------------------------------------------------------------------

class NonlinearPricePredictor:
    """
    Wraps a fitted LightGBM regressor plus preprocessing, with SHAP-based
    explanations for waterfall UIs.
    """

    def __init__(
        self,
        *,
        lgb_model,
        feature_columns,
        clean_map,
        orig_of,
        num_feats,
        nl_categoricals,
        seg_defaults,
        pi_quantiles,
        pi_global,
        metadata,
    ):
        self.lgb_model = lgb_model
        self.feature_columns = list(feature_columns)
        self.clean_map = dict(clean_map)
        self.orig_of = dict(orig_of)
        self.num_feats = list(num_feats)
        self.nl_categoricals = set(nl_categoricals)
        self.seg_defaults = {k: dict(v) for k, v in seg_defaults.items()}
        self.pi_quantiles = {k: tuple(v) for k, v in pi_quantiles.items()}
        self.pi_global = tuple(pi_global)
        self.metadata = dict(metadata)
        self._explainer = None  # lazy-init on first .explain() call

    def _set_categorical_dummy(self, row: pd.Series, prefix: str, value):
        if value is None:
            return
        target = _clean_name(f"{prefix}_{value}")
        if target in row.index:
            row[target] = 1
            return
        for c in row.index:
            if c.startswith(prefix + "_"):
                tail = c[len(prefix) + 1:]
                if _clean_name(tail).lower() == _clean_name(str(value)).lower():
                    row[c] = 1
                    return

    def _prepare_row(self, feats: dict):
        """Returns (segment, X_row) where X_row is a 1-row DataFrame."""
        feats = dict(feats)
        segment = feats.pop("segment", "Residential")

        row = pd.Series(0.0, index=self.feature_columns)
        for raw_name, val in self.seg_defaults.get(segment, {}).items():
            cl = _clean_name(raw_name)
            if cl in row.index:
                row[cl] = val

        seg_col = f"segment_{segment}"
        if seg_col in row.index:
            row[seg_col] = 1

        for k, v in feats.items():
            if k in self.nl_categoricals:
                self._set_categorical_dummy(row, k, v)
            else:
                cl = _clean_name(k)
                if cl in row.index:
                    row[cl] = float(v)

        X_row = row.to_frame().T[self.feature_columns].astype(float)
        return segment, X_row

    def _get_explainer(self):
        """Build the SHAP TreeExplainer on first call, cache after."""
        if self._explainer is None:
            import shap  # lazy import so .predict() doesn't require shap
            self._explainer = shap.TreeExplainer(self.lgb_model)
        return self._explainer

    def predict(self, **feats) -> dict:
        """
        Predict a dollar price.

        Returns a dict:
          pred          float  dollar point estimate
          pi_lo, pi_hi  float  95% prediction interval (dollars)
          log_pred      float  prediction on the log scale
          segment_used  str
        """
        segment, X_row = self._prepare_row(feats)
        log_pred = float(self.lgb_model.predict(X_row)[0])
        q_lo, q_hi = self.pi_quantiles.get(segment, self.pi_global)
        return {
            "pred": float(np.exp(log_pred)),
            "pi_lo": float(np.exp(log_pred + q_lo)),
            "pi_hi": float(np.exp(log_pred + q_hi)),
            "log_pred": log_pred,
            "segment_used": segment,
        }

    def explain(self, top_n: int = 15, **feats) -> dict:
        """
        Predict AND return SHAP-based per-feature contributions.

        Requires `shap` in the teammate's environment:
            pip install shap

        See EXPLANATIONS.md for the full schema.
        """
        segment, X_row = self._prepare_row(feats)
        log_pred = float(self.lgb_model.predict(X_row)[0])
        q_lo, q_hi = self.pi_quantiles.get(segment, self.pi_global)

        explainer = self._get_explainer()
        shap_vals = explainer.shap_values(X_row)[0]
        base_value = (float(explainer.expected_value)
                      if np.isscalar(explainer.expected_value)
                      else float(explainer.expected_value[0]))

        contribs = []
        for i, col in enumerate(self.feature_columns):
            s = float(shap_vals[i])
            if s == 0.0:
                continue
            contribs.append({
                "feature": self.orig_of.get(col, col),
                "feature_internal": col,
                "value": float(X_row.iloc[0, i]),
                "shap_log": s,
                "shap_dollars": _shap_to_dollar_delta(base_value, s),
            })

        contribs.sort(key=lambda d: -abs(d["shap_log"]))
        if top_n is not None:
            contribs = contribs[:top_n]

        return {
            "pred": float(np.exp(log_pred)),
            "pi_lo": float(np.exp(log_pred + q_lo)),
            "pi_hi": float(np.exp(log_pred + q_hi)),
            "log_pred": log_pred,
            "segment_used": segment,
            "explanation": {
                "method": "shap_tree",
                "base_value": base_value,
                "base_value_dollars": float(np.exp(base_value)),
                "contributions": contribs,
            },
        }
