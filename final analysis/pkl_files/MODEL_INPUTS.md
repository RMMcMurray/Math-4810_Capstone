# Model Inputs — Reference Sheet

Both `linear_model.pkl` and `nonlinear_model.pkl` accept the **exact same** raw
keyword arguments. Any feature you omit falls back to the training-set
**median for the requested `segment`**, so you can call `.predict(...)` with
as few or as many features as you have.

```python
import pickle
with open("linear_model.pkl", "rb") as f:
    linear_model = pickle.load(f)

result = linear_model.predict(
    segment="Residential",
    GLA=2400,
    Acreage=0.25,
)
# result -> {"pred": 412345.0, "pi_lo": 285000.0, "pi_hi": 598000.0,
#            "log_pred": 12.93, "segment_used": "Residential", "n_features": 87}
```

> **Note on loading:** `price_predictors.py` must be importable in your
> environment. Ship it alongside the `.pkl` files — `pickle.load` reads the
> class definitions from there.

---

## Return value

Every `.predict(**features)` call returns a `dict`:

| Key            | Type   | Meaning                                          |
| :------------- | :----- | :----------------------------------------------- |
| `pred`         | float  | Point estimate in **dollars**                    |
| `pi_lo`        | float  | Lower bound of the 95% prediction interval ($)   |
| `pi_hi`        | float  | Upper bound of the 95% prediction interval ($)   |
| `log_pred`     | float  | Prediction on the log scale (for debugging)      |
| `segment_used` | str    | Segment the model priced into                    |
| `n_features`   | int    | (linear only) Lasso-selected features in the fit |

---

## Categorical features (strings)

Pass the raw label exactly as it appears in the training data. Unknown values
are silently ignored and the model falls back to the segment baseline.

| Kwarg                          | Example values                                        |
| :----------------------------- | :---------------------------------------------------- |
| `segment`                      | `"Residential"`, `"Land Sales"`, `"Multi-Family"`     |
| `PropTypeDescription`          | `"Single Family Res"`, `"Condominium"`, ...           |
| `SpecificPropTypeDescription`  | e.g. `"Detached"`, `"Townhouse"`, ...                 |
| `NbhdCode2`                    | neighbourhood code string                             |
| `Main_StyleDesc`               | e.g. `"Rambler/Ranch"`, `"2 Story"`, ...              |

> For the full list of valid values per field, read `model.metadata` or
> inspect the training CSV — the model only recognises categories it saw at
> fit time.

---

## Numeric features

All numbers are raw, untransformed values. The linear model applies log /
arcsinh transforms internally; the LightGBM model uses them as-is. You do
**not** do any transformation yourself.

### Area / size (sqft)

| Kwarg                         | Notes                                          |
| :---------------------------- | :--------------------------------------------- |
| `GLA`                         | Gross living area, sqft                        |
| `'Tot Bsmt'`                  | Total basement area, sqft (**key has a space** — use `**{'Tot Bsmt': 800}`) |
| `GarageArea`                  | Attached garage, sqft                          |
| `CarportArea`                 | Attached carport, sqft                         |
| `SqFt_DetGarage`              | Detached garage, sqft                          |
| `SqFt_DetCarport`             | Detached carport, sqft                         |
| `Sqft_Barn`                   | Barn, sqft (note the lowercase `q`)            |
| `SqFt_Guest_House`            | Guest house, sqft                              |
| `SqFt_Pools`                  | Pools, sqft                                    |
| `SqFt_Recreational_Courts`    | Courts, sqft                                   |
| `SqFt_Shed`                   | Shed, sqft                                     |
| `SqFt_Gazebo_Pavilion`        | Gazebo / pavilion, sqft                        |

### Land / valuation

| Kwarg            | Notes                                          |
| :--------------- | :--------------------------------------------- |
| `Acreage`        | Lot size in acres                              |
| `'Land Value'`   | Assessor land value, dollars (**key has a space**) |
| `'Total Value'`  | Assessor total value, dollars (**key has a space**) |

### Condition / age / quality

| Kwarg                      | Notes                                        |
| :------------------------- | :------------------------------------------- |
| `EffYearBuilt_Weighted`    | Effective year built (e.g. `2005`)           |
| `Quality_Weighted`         | Quality rating (roughly 1-6 scale)           |
| `BsmtFinishPct_Weighted`   | Fraction of basement finished, 0-1           |

### Counts

| Kwarg               | Notes                                                |
| :------------------ | :--------------------------------------------------- |
| `FullBaths_Total`   | Integer count                                        |
| `HalfBaths_Total`   | Integer count                                        |
| `GarageCapacity`    | Number of bay spaces                                 |
| `KitchenCount`      | Integer count (nonlinear model only)                 |

---

## Kwargs with spaces

Python won't let you write `predict(Tot Bsmt=800)`. Three columns have
spaces in the training data — call them via kwargs unpacking:

```python
model.predict(
    segment="Residential",
    GLA=2400,
    **{"Tot Bsmt": 800, "Land Value": 125000, "Total Value": 480000},
)
```

---

## Segments

The three segments have meaningfully different baselines and prediction
intervals:

| Segment         | PI width    | Typical use case                   |
| :-------------- | :---------- | :--------------------------------- |
| `Residential`   | Tightest    | Single-family homes, condos        |
| `Multi-Family`  | Wider       | Duplex, triplex, small apartments  |
| `Land Sales`    | Widest      | Vacant / undeveloped lots          |

If no `segment` is passed, the model defaults to `"Residential"`.

---

## Metadata

Each pickle exposes a `.metadata` dict you can surface in an "About this
model" page:

```python
>>> linear_model.metadata
{"model_type": "Lasso-selected OLS ...",
 "train_rows": 3421, "test_rows": 856,
 "test_r2": 0.8812, "test_dollar_mae": 34877.3,
 "n_selected_features": 87,
 "random_state": 2026,
 "fitted_at": "2026-04-20T21:44:10Z"}
```

---

## Environment

See `versions.txt` — your environment should have the same major versions of
`statsmodels`, `lightgbm`, and `scikit-learn` that the models were fit with,
or `pickle.load` may fail or warn. `numpy` and `pandas` compatibility is
usually looser but matching is safest.
