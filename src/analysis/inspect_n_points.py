import json
print(f"{'dataset':10s} {'horizon':8s} {'TTM_n':>8s} {'TimesFM_n':>10s} {'match?':>8s}")
pairs = [
    ("Alibaba",   "ttm_k20_alibaba.json",   "timesfm_k20_alibaba.json"),
    ("Bitbrains", "ttm_k50_bitbrains.json", "timesfm_k50_bitbrains.json"),
    ("ByteDance", "ttm_k50_bytedance.json", "timesfm_k50_bytedance.json"),
]
for ds, ttm_f, tfm_f in pairs:
    ttm = json.load(open(ttm_f))["granite_ttm_zero_shot"]
    tfm = json.load(open(tfm_f))["timesfm_zero_shot"]
    for hz in ["10min","30min","60min","120min"]:
        n_ttm = ttm[hz]["n_points"]
        n_tfm = tfm[hz]["n_points"]
        match = "OK" if n_ttm == n_tfm else "DIFF"
        print(f"{ds:10s} {hz:8s} {n_ttm:>8d} {n_tfm:>10d} {match:>8s}")
