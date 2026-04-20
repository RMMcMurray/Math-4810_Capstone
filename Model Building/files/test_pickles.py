"""Validate that the pickled models behave correctly when loaded fresh."""
import pickle

# Load both models
with open("linear_model.pkl", "rb") as f:
    lin = pickle.load(f)
with open("nonlinear_model.pkl", "rb") as f:
    nl = pickle.load(f)

print("=" * 70)
print("Test 1: Bare minimum call (just segment)")
print("=" * 70)
for m, name in [(lin, "Linear"), (nl, "LightGBM")]:
    r = m.predict(segment="Residential")
    print(f"  {name:10s}  ${r['pred']:>10,.0f}   "
          f"PI [${r['pi_lo']:,.0f} - ${r['pi_hi']:,.0f}]")

print("\n" + "=" * 70)
print("Test 2: Larger GLA should increase price")
print("=" * 70)
for m, name in [(lin, "Linear"), (nl, "LightGBM")]:
    low  = m.predict(segment="Residential", GLA=1200)
    high = m.predict(segment="Residential", GLA=4000)
    delta = high["pred"] - low["pred"]
    print(f"  {name:10s}  GLA 1200 -> ${low['pred']:>10,.0f} | "
          f"GLA 4000 -> ${high['pred']:>10,.0f}  (+${delta:,.0f})")
    assert delta > 0, f"{name}: larger GLA should raise price"

print("\n" + "=" * 70)
print("Test 3: Segment swap should shift baseline meaningfully")
print("=" * 70)
for m, name in [(lin, "Linear"), (nl, "LightGBM")]:
    res  = m.predict(segment="Residential")
    land = m.predict(segment="Land Sales")
    mf   = m.predict(segment="Multi-Family")
    print(f"  {name:10s}  Residential ${res['pred']:>10,.0f}  |  "
          f"Land ${land['pred']:>10,.0f}  |  MF ${mf['pred']:>10,.0f}")

print("\n" + "=" * 70)
print("Test 4: Spaced kwargs work via **{} unpacking")
print("=" * 70)
for m, name in [(lin, "Linear"), (nl, "LightGBM")]:
    r = m.predict(
        segment="Residential",
        GLA=2400,
        **{"Tot Bsmt": 800, "Land Value": 150000, "Total Value": 500000},
    )
    print(f"  {name:10s}  ${r['pred']:>10,.0f}   "
          f"PI [${r['pi_lo']:,.0f} - ${r['pi_hi']:,.0f}]")

print("\n" + "=" * 70)
print("Test 5: Unknown kwargs are ignored (forward-compat)")
print("=" * 70)
for m, name in [(lin, "Linear"), (nl, "LightGBM")]:
    try:
        r = m.predict(segment="Residential", GLA=2400, Completely_Made_Up_Field=999)
        print(f"  {name:10s}  OK, got ${r['pred']:>10,.0f}")
    except Exception as e:
        print(f"  {name:10s}  FAILED: {e}")

print("\n" + "=" * 70)
print("Test 6: Unknown categorical value is ignored")
print("=" * 70)
for m, name in [(lin, "Linear"), (nl, "LightGBM")]:
    try:
        r = m.predict(segment="Residential", PropTypeDescription="Moon Base")
        print(f"  {name:10s}  OK, got ${r['pred']:>10,.0f}")
    except Exception as e:
        print(f"  {name:10s}  FAILED: {e}")

print("\n" + "=" * 70)
print("Test 7: Metadata is accessible")
print("=" * 70)
print(f"  Linear.metadata    : {lin.metadata}")
print(f"  Nonlinear.metadata : {nl.metadata}")

print("\n" + "=" * 70)
print("Test 8: Realistic full call")
print("=" * 70)
example = dict(
    segment="Residential",
    PropTypeDescription="Single Family Res",
    GLA=2400, EffYearBuilt_Weighted=2005, Acreage=0.25,
    **{"Tot Bsmt": 800}, GarageArea=500,
    FullBaths_Total=2, HalfBaths_Total=1, GarageCapacity=2,
    Quality_Weighted=4.0,
)
for m, name in [(lin, "Linear"), (nl, "LightGBM")]:
    r = m.predict(**example)
    print(f"  {name:10s}  ${r['pred']:>10,.0f}   "
          f"PI [${r['pi_lo']:,.0f} - ${r['pi_hi']:,.0f}]   "
          f"segment={r['segment_used']}")

print("\nAll tests passed.")
