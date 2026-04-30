import json
p = json.load(open("ttm_k50_bitbrains.json"))
b = p["granite_ttm_zero_shot"]
print(f"{'horizon':8s} {'n':>6s} {'r2_mean':>9s} {'naive_sub':>10s} {'delta_pp':>10s}")
for hz in ["10min","30min","60min","120min"]:
    h = b[hz]
    delta = (h["r2_mean"] - h["r2_naive_subsample"]) * 100
    print(f"{hz:8s} {h['n_points']:>6d} {h['r2_mean']:>9.4f} {h['r2_naive_subsample']:>10.4f} {delta:>+10.2f}")
