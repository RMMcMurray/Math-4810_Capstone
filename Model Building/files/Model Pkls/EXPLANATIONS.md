# Explanations — Waterfall Data Reference

Both models expose an `.explain(**features)` method in addition to
`.predict(**features)`. This document covers what the explanation data
looks like, what each field means, and how to turn it into a waterfall
plot (or any "why did the model say this?" UI element).

## When to call which method

| Call            | Use when…                                                                  |
| :-------------- | :------------------------------------------------------------------------- |
| `.predict(...)` | You just need the dollar number (search results, bulk pricing, API calls). |
| `.explain(...)` | You're showing one property to a user and want to explain the quote.       |

`.explain()` is slower — maybe 50–200 ms per call vs single-digit ms for
`.predict()` — because it runs SHAP (non-linear model) or sums coefficient
contributions (linear model). Don't call it in batch loops; call it once
when the user clicks "explain this estimate."

---

## The return shape

`.explain()` returns everything `.predict()` does, plus one extra key:
`explanation`.

```python
{
  "pred": 412345.0,
  "pi_lo": 285000.0,
  "pi_hi": 598000.0,
  "log_pred": 12.93,
  "segment_used": "Residential",
  "n_features": 87,                    # linear only
  "explanation": {
    "method": "shap_tree",             # "shap_tree" or "linear_coefficients"
    "base_value": 12.67,               # log-price intercept / expected value
    "base_value_dollars": 318200.0,    # exp(base_value) for UI convenience
    "contributions": [
      {
        "feature": "GLA",              # human-readable name
        "value": 2400.0,               # the value the model saw
        "shap_log": 0.2847,            # contribution on log-price scale
        "shap_dollars": 102540.0       # approximate dollar delta
      },
      {
        "feature": "Total Value",
        "value": 480000.0,
        "shap_log": 0.1533,
        "shap_dollars": 52760.0
      },
      ...
    ]
  }
}
```

### Field-by-field

| Field                               | Type    | What it is                                                             |
| :---------------------------------- | :------ | :--------------------------------------------------------------------- |
| `explanation.method`                | str     | `"shap_tree"` (LightGBM) or `"linear_coefficients"` (linear).          |
| `explanation.base_value`            | float   | Model's log-price baseline before any feature pushes.                  |
| `explanation.base_value_dollars`    | float   | `exp(base_value)` — convenient starting point for the waterfall.       |
| `explanation.contributions`         | list    | Ranked by `abs(shap_log)` descending. Length = `top_n` (default 15).   |
| `contributions[i].feature`          | str     | Human-readable feature name (what to show on the bar label).           |
| `contributions[i].value`            | float   | The feature's value for this prediction.                               |
| `contributions[i].shap_log`         | float   | Push on log-price. **Positive = raises price, negative = lowers.**     |
| `contributions[i].shap_dollars`     | float   | Approximate dollar equivalent (see "important math note" below).       |
| `contributions[i].coefficient`      | float   | (linear only) The OLS coefficient used — `shap_log = coefficient × value`. |
| `contributions[i].feature_internal` | str     | (LightGBM only) The cleaned column name (usually not shown in UI).     |

---

## Controlling the list length

```python
model.explain(top_n=10, segment="Residential", GLA=2400)   # top 10
model.explain(top_n=None, segment="Residential", GLA=2400) # every feature
```

Default is 15 — matches the `max_display=15` used in the notebook's waterfalls.

---

## Important math note: log vs dollars

SHAP contributions are additive on the **log scale**, not the dollar
scale. Two consequences matter for the UI:

**1. On the log scale, the math is exact and clean.** For both models:

```
log_pred ≈ base_value + sum(contribution.shap_log for all contribs)
```

(Exact for the linear model when `top_n=None`. Approximately exact for
LightGBM — SHAP values sum to the prediction up to floating-point noise.)

**2. On the dollar scale, bars don't sum to the final price.** The
`shap_dollars` field is computed as `base_dollars × (exp(shap_log) - 1)`.
That's the dollar delta *relative to the base*, which is what you want
for a waterfall bar — but if you naively add them up, you'll overshoot
or undershoot the final prediction a bit because exponentials aren't
additive.

**Bottom line for the UI:** use `shap_dollars` to decide bar *sizes* and
*direction* (up or down). Use `pred` (from `.predict()`/`.explain()`
directly) as the final total displayed in the UI. Don't compute the
total by summing bars.

If the UI wants the bars to add up perfectly (some designers care,
others don't), use the log scale: start the x-axis at `base_value` in
log space, show each bar's width as `shap_log`, and label the axis ticks
in dollars via `exp()`. The notebook's existing waterfalls do it this
way.

---

## Example — building a waterfall in pseudo-code

```python
result = model.explain(segment="Residential", GLA=2400, Acreage=0.25)
expl = result["explanation"]

# x-axis starts here
x = expl["base_value_dollars"]
bars = []

for c in expl["contributions"]:
    bar = {
        "label": f"{c['feature']} = {c['value']:,.0f}",
        "start": x,
        "end": x + c["shap_dollars"],
        "color": "green" if c["shap_log"] > 0 else "red",
    }
    bars.append(bar)
    x = bar["end"]

# Final total (use the authoritative pred, not x)
final_total = result["pred"]
```

That's it — the `bars` list drives whatever chart library the UI team
prefers (D3, Plotly waterfall, Chart.js, Recharts, custom SVG).

---

## Two models, same API — but subtly different explanations

Both models expose `.explain(**feats)` with the same return shape, but
the **meaning** of a contribution differs:

- **Linear model** — `shap_log = coefficient × feature_value`. Clean,
  interpretable, and every bar maps to one engineered feature (which may
  be a polynomial term like `ln_GLA_sq` or an interaction term like
  `segment_Residential_x_ln_GLA`). Some feature names will look
  mathematical; that's expected for a Lasso+OLS model with poly/interaction
  features.

- **LightGBM** — `shap_log` is the TreeSHAP contribution, which captures
  non-linear effects and feature interactions in a principled way.
  Feature names are the original human-readable names (`GLA`, `Total Value`,
  `Quality_Weighted`, etc.), and segment-level behavior shows up naturally
  in the contributions rather than needing explicit interaction terms.

For most UIs, the LightGBM explanation is the one you'll want to show —
it's easier to read and more accurate (R² ≈ 0.95 vs 0.88). The linear
model's explanation is useful when you need the "every coefficient has
a p-value" story for a technical audience.

---

## Dependency note

`.explain()` on the **LightGBM** model requires the `shap` package:

```bash
pip install shap
```

`.predict()` does **not** require shap. So if the UI environment can't
install shap for any reason, `.predict()` still works fine — you just
lose the explanation feature on the non-linear model. The linear model's
`.explain()` has no shap dependency at all (it's just arithmetic on the
OLS coefficients).

---

## Performance ballpark

| Call                              | Typical time per call |
| :-------------------------------- | :-------------------- |
| `linear.predict(...)`             | ~2–5 ms               |
| `linear.explain(...)`             | ~3–6 ms               |
| `nonlinear.predict(...)`          | ~2–5 ms               |
| `nonlinear.explain(...)` (1st)    | ~300–500 ms (SHAP explainer builds on first call) |
| `nonlinear.explain(...)` (after)  | ~50–150 ms            |

The SHAP explainer is built lazily and cached on the first `.explain()`
call, so the first one is slow and the rest are fast. If the UI does
server-side rendering, calling `.explain()` once at startup to warm up
the cache is a reasonable optimization.
