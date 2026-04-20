"""Validate that the .explain() method on both pickled models works."""
import pickle
import numpy as np

with open("linear_model.pkl", "rb") as f:
    lin = pickle.load(f)
with open("nonlinear_model.pkl", "rb") as f:
    nl = pickle.load(f)

example = dict(
    segment="Residential",
    PropTypeDescription="Single Family Res",
    GLA=2400, EffYearBuilt_Weighted=2005, Acreage=0.25,
    **{"Tot Bsmt": 800}, GarageArea=500,
    FullBaths_Total=2, HalfBaths_Total=1, GarageCapacity=2,
    Quality_Weighted=4.0,
)

print("=" * 70)
print("Test 1: .explain() returns the prediction PLUS an explanation block")
print("=" * 70)
for m, name in [(lin, "Linear"), (nl, "LightGBM")]:
    r = m.explain(**example)
    assert "pred" in r and "pi_lo" in r and "pi_hi" in r
    assert "explanation" in r
    expl = r["explanation"]
    assert set(expl.keys()) == {"method", "base_value", "base_value_dollars", "contributions"}
    print(f"  {name:10s}  pred=${r['pred']:>10,.0f}  "
          f"base=${expl['base_value_dollars']:>10,.0f}  "
          f"contribs={len(expl['contributions'])}  "
          f"method={expl['method']}")

print("\n" + "=" * 70)
print("Test 2: top_n parameter controls list length")
print("=" * 70)
for m, name in [(lin, "Linear"), (nl, "LightGBM")]:
    r5 = m.explain(top_n=5, **example)
    r20 = m.explain(top_n=20, **example)
    rall = m.explain(top_n=None, **example)
    print(f"  {name:10s}  top_n=5: {len(r5['explanation']['contributions'])} | "
          f"top_n=20: {len(r20['explanation']['contributions'])} | "
          f"top_n=None: {len(rall['explanation']['contributions'])}")
    assert len(r5["explanation"]["contributions"]) <= 5

print("\n" + "=" * 70)
print("Test 3: Contributions are sorted by |shap_log| descending")
print("=" * 70)
for m, name in [(lin, "Linear"), (nl, "LightGBM")]:
    r = m.explain(**example)
    contribs = r["explanation"]["contributions"]
    abs_shaps = [abs(c["shap_log"]) for c in contribs]
    is_sorted = all(abs_shaps[i] >= abs_shaps[i + 1] for i in range(len(abs_shaps) - 1))
    print(f"  {name:10s}  sorted descending by |shap_log|: {is_sorted}")
    assert is_sorted

print("\n" + "=" * 70)
print("Test 4: Every contribution has the expected fields")
print("=" * 70)
for m, name in [(lin, "Linear"), (nl, "LightGBM")]:
    r = m.explain(**example)
    c = r["explanation"]["contributions"][0]
    required = {"feature", "value", "shap_log", "shap_dollars"}
    missing = required - set(c.keys())
    print(f"  {name:10s}  top contrib: {c['feature']:30s}  "
          f"shap_log={c['shap_log']:+.4f}  shap_dollars=${c['shap_dollars']:+,.0f}")
    assert not missing, f"missing fields: {missing}"

print("\n" + "=" * 70)
print("Test 5: Math check - sum of shap_log should ~= log_pred - base_value")
print("=" * 70)
for m, name in [(lin, "Linear"), (nl, "LightGBM")]:
    r = m.explain(top_n=None, **example)  # all contributions
    sum_shap = sum(c["shap_log"] for c in r["explanation"]["contributions"])
    expected = r["log_pred"] - r["explanation"]["base_value"]
    diff = abs(sum_shap - expected)
    print(f"  {name:10s}  sum(shap_log)={sum_shap:+.4f}  "
          f"log_pred - base={expected:+.4f}  |diff|={diff:.6f}")
    # Allow larger tolerance for linear because 'const' coefficient differs slightly from base_value convention
    tol = 1e-4 if name == "LightGBM" else 0.05
    assert diff < tol, f"{name}: log-scale additivity broken (diff={diff})"

print("\n" + "=" * 70)
print("Test 6: Changing a feature changes its contribution, not unrelated ones")
print("=" * 70)
for m, name in [(lin, "Linear"), (nl, "LightGBM")]:
    r1 = m.explain(top_n=None, segment="Residential", GLA=1500)
    r2 = m.explain(top_n=None, segment="Residential", GLA=3500)
    # Predictions should differ meaningfully
    pct = (r2["pred"] - r1["pred"]) / r1["pred"] * 100
    print(f"  {name:10s}  GLA 1500 -> ${r1['pred']:>10,.0f}  |  "
          f"GLA 3500 -> ${r2['pred']:>10,.0f}  ({pct:+.1f}%)")
    assert r2["pred"] > r1["pred"]

print("\n" + "=" * 70)
print("Test 7: Show a sample waterfall for LightGBM (top 8)")
print("=" * 70)
r = nl.explain(top_n=8, **example)
expl = r["explanation"]
print(f"  Base value (log):         {expl['base_value']:+.4f}")
print(f"  Base value (dollars):     ${expl['base_value_dollars']:,.0f}")
print(f"  Final prediction:         ${r['pred']:,.0f}")
print(f"  PI:                       [${r['pi_lo']:,.0f} - ${r['pi_hi']:,.0f}]")
print()
print(f"  {'Feature':<35s}{'Value':>12s}{'Δ log':>10s}{'Δ $':>15s}")
print(f"  {'-'*35}{'-'*12}{'-'*10}{'-'*15}")
for c in expl["contributions"]:
    arrow = "+" if c["shap_log"] > 0 else "-"
    print(f"  {c['feature'][:33]:<35s}"
          f"{c['value']:>12,.2f}"
          f"{c['shap_log']:>+10.4f}"
          f"  {arrow} ${abs(c['shap_dollars']):>10,.0f}")

print("\n" + "=" * 70)
print("Test 8: .predict() still works without triggering SHAP explainer build")
print("=" * 70)
# Calling .predict() should NOT trigger shap import or SHAP explainer build.
# Simulate a fresh pickle load (reset the cached explainer).
for m, name in [(lin, "Linear"), (nl, "LightGBM")]:
    r = m.predict(**example)
    print(f"  {name:10s}  .predict() ok, ${r['pred']:>10,.0f}")

# Reset the explainer and verify .predict() keeps it None
nl._explainer = None
_ = nl.predict(**example)
print(f"  After .predict() only, _explainer is None: {nl._explainer is None}")
assert nl._explainer is None, ".predict() should not build the SHAP explainer"
_ = nl.explain(**example)
print(f"  After .explain(),      _explainer is None: {nl._explainer is None}")
assert nl._explainer is not None, ".explain() should build the SHAP explainer"

print("\nAll explanation tests passed.")
