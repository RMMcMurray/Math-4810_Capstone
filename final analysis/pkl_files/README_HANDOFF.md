# Price Models — Hand-off Bundle

This package contains two fully-wrapped price prediction models ready for
UI integration. Each pickle loads as an object with a single `.predict(**raw_features)`
method — no preprocessing required on the UI side.

## What's in the box

| File                  | Purpose                                                          |
| :-------------------- | :--------------------------------------------------------------- |
| `linear_model.pkl`    | Lasso→OLS linear model (Test R² ≈ 0.88, $ MAE ≈ $35k)            |
| `nonlinear_model.pkl` | LightGBM non-linear model (Test R² ≈ 0.95, $ MAE ≈ $18k)         |
| `price_predictors.py` | **Required** — class definitions the pickles depend on           |
| `MODEL_INPUTS.md`     | Full reference of every input the models accept                  |
| `EXPLANATIONS.md`     | How to use `.explain()` and build waterfall plots from its output |
| `versions.txt`        | Package versions the pickles were fit with                       |

## Two methods, same inputs

Each pickle exposes two methods that take the **same raw feature kwargs**:

- **`.predict(**features)`** — fast dollar prediction + 95% PI.
- **`.explain(**features)`** — same prediction plus a ranked list of per-feature
  contributions for building waterfall plots. Read `EXPLANATIONS.md` before
  using this one.

## Minimum working example

```python
import pickle

with open("linear_model.pkl", "rb") as f:
    linear_model = pickle.load(f)

with open("nonlinear_model.pkl", "rb") as f:
    nonlinear_model = pickle.load(f)

# Both models take the SAME raw feature kwargs
features = dict(
    segment="Residential",
    PropTypeDescription="Single Family Res",
    GLA=2400,
    Acreage=0.25,
    EffYearBuilt_Weighted=2005,
    Quality_Weighted=4.0,
    FullBaths_Total=2,
    HalfBaths_Total=1,
    GarageCapacity=2,
    GarageArea=500,
    **{"Tot Bsmt": 800},  # kwargs with spaces need unpacking syntax
)

# Dollar prediction
result = linear_model.predict(**features)
print(f"${result['pred']:,.0f}")
print(f"95% PI: ${result['pi_lo']:,.0f} - ${result['pi_hi']:,.0f}")

# Same prediction plus a waterfall-ready explanation
explained = nonlinear_model.explain(top_n=10, **features)
for c in explained["explanation"]["contributions"]:
    print(f"  {c['feature']:<30s}  Δ$ {c['shap_dollars']:+,.0f}")
```

## Installation

```bash
pip install numpy pandas scikit-learn statsmodels lightgbm
# Optional: only needed for nonlinear_model.explain() (SHAP waterfalls)
pip install shap
```

Match major versions to `versions.txt` if you can — pickles are sensitive
to library upgrades (especially statsmodels and lightgbm).

## Important — shipping the class module

`pickle.load` needs to import `LinearPricePredictor` and
`NonlinearPricePredictor` from `price_predictors.py`. **Keep that file
alongside the pickles** (either on the Python path, or in the same
directory as your loading code).

## Read next

- `MODEL_INPUTS.md` — the complete list of variables the models accept, with
  types, units, and notes on the three categorical segments.
- `EXPLANATIONS.md` — the schema for `.explain()` output, how to build
  waterfall plots from it, and the important math note on log vs dollar
  additivity.

## Questions or issues

Ping me (Melanie) and I'll re-run `build_pickles.py` to produce fresh
artifacts. The build script is deterministic — same seed (`random_state=2026`),
same Lasso-selected features, same LightGBM hyperparameters as `Final_Models.qmd`.
