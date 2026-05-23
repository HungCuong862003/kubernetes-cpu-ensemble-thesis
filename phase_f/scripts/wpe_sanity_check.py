# phase_f/scripts/wpe_sanity_check.py
# Sanity check: WPE vs ACF@24h, CV, Hurst correlations per dataset.
# Run separately because compute_wpe.py crashed at this step due to
# Bitbrains int vs str ID-type mismatch. WPE CSVs already saved.

import pandas as pd

# ---- Load WPE results ----
bb_wpe = pd.read_csv("phase_f/data/wpe_bitbrains.csv")
byt_wpe = pd.read_csv("phase_f/data/wpe_bytedance.csv")
ali_wpe = pd.read_csv("phase_f/data/wpe_alibaba.csv")

# ---- Load predictability stats ----
bb_omega = pd.read_csv("results/bitbrains/omega_bitbrains.csv")
byt_stats = pd.read_csv("results/bytedance/bytedance_per_instance_stats.csv")
ali_omega = pd.read_csv("results/alibaba/omega_alibaba.csv")

print("=" * 60)
print("ID type inspection (debug)")
print("=" * 60)
print(f"BB  WPE container_id: {bb_wpe['container_id'].head(2).tolist()}  dtype={bb_wpe['container_id'].dtype}")
print(f"BB  omega vm_id:      {bb_omega['vm_id'].head(2).tolist()}  dtype={bb_omega['vm_id'].dtype}")
print(f"BYT WPE container_id: {byt_wpe['container_id'].head(2).tolist()}  dtype={byt_wpe['container_id'].dtype}")
print(f"BYT stats columns:    {byt_stats.columns.tolist()}")
print(f"BYT stats instance_id: {byt_stats['instance_id'].head(2).tolist()}  dtype={byt_stats['instance_id'].dtype}")
print(f"ALI WPE container_id: {ali_wpe['container_id'].head(2).tolist()}  dtype={ali_wpe['container_id'].dtype}")
print(f"ALI omega container_id: {ali_omega['container_id'].head(2).tolist()}  dtype={ali_omega['container_id'].dtype}")
print()

# ============================================================
# Bitbrains — strip "bb_" prefix from WPE side to match int vm_id
# ============================================================
print("=" * 60)
print("Bitbrains")
print("=" * 60)
bb_wpe["vm_id"] = bb_wpe["container_id"].str.replace("bb_", "", regex=False).astype(int)
bb_merged = bb_wpe.merge(
    bb_omega[["vm_id", "acf_24h", "cv", "hurst"]],
    on="vm_id", how="inner"
)
print(f"  merged: {len(bb_merged)} VMs")
print(f"  WPE vs ACF@24h: r = {bb_merged['wpe_m4_t1'].corr(bb_merged['acf_24h']):.4f}")
print(f"  WPE vs CV:      r = {bb_merged['wpe_m4_t1'].corr(bb_merged['cv']):.4f}")
print(f"  WPE vs Hurst:   r = {bb_merged['wpe_m4_t1'].corr(bb_merged['hurst']):.4f}")

# ============================================================
# ByteDance — likely instance_id == container_id format
# ============================================================
print()
print("=" * 60)
print("ByteDance")
print("=" * 60)
byt_stats["instance_id"] = "bd_" + byt_stats["instance_id"]
byt_merged = byt_wpe.merge(
    byt_stats.rename(columns={"instance_id": "container_id"})[["container_id", "acf_24h", "hurst"]],
    on="container_id", how="inner"
)
print(f"  merged: {len(byt_merged)} instances")
print(f"  WPE vs ACF@24h: r = {byt_merged['wpe_m4_t1'].corr(byt_merged['acf_24h']):.4f}")
print(f"  WPE vs Hurst:   r = {byt_merged['wpe_m4_t1'].corr(byt_merged['hurst']):.4f}")

# ============================================================
# Alibaba — both sides should be string container_id
# ============================================================
print()
print("=" * 60)
print("Alibaba")
print("=" * 60)
ali_merged = ali_wpe.merge(
    ali_omega[["container_id", "acf_24h", "cv", "hurst"]],
    on="container_id", how="inner"
)
print(f"  merged: {len(ali_merged)} containers")
print(f"  WPE vs ACF@24h: r = {ali_merged['wpe_m4_t1'].corr(ali_merged['acf_24h']):.4f}")
print(f"  WPE vs CV:      r = {ali_merged['wpe_m4_t1'].corr(ali_merged['cv']):.4f}")
print(f"  WPE vs Hurst:   r = {ali_merged['wpe_m4_t1'].corr(ali_merged['hurst']):.4f}")

print()
print("=" * 60)
print("Cross-dataset WPE summary")
print("=" * 60)
for name, df in [("Bitbrains", bb_merged), ("ByteDance", byt_merged), ("Alibaba", ali_merged)]:
    print(f"  {name:10s}  n={len(df):4d}  "
          f"WPE median={df['wpe_m4_t1'].median():.4f}  "
          f"ACF@24h median={df['acf_24h'].median():.4f}")

print("\nDone.")

# ============================================================
# Spearman robustness — Bitbrains Pearson +0.87 looks high;
# narrow ACF@24h range may inflate Pearson. Spearman is rank-based
# and more robust to range issues.
# ============================================================
print()
print("=" * 60)
print("Spearman (rank-based) correlations")
print("=" * 60)
for name, df, has_cv in [
    ("Bitbrains", bb_merged, True),
    ("ByteDance", byt_merged, False),
    ("Alibaba", ali_merged, True),
]:
    rho_acf = df["wpe_m4_t1"].corr(df["acf_24h"], method="spearman")
    rho_hurst = df["wpe_m4_t1"].corr(df["hurst"], method="spearman")
    line = f"  {name:10s}  WPE-ACF: rho={rho_acf:+.4f}  WPE-Hurst: rho={rho_hurst:+.4f}"
    if has_cv:
        rho_cv = df["wpe_m4_t1"].corr(df["cv"], method="spearman")
        line += f"  WPE-CV: rho={rho_cv:+.4f}"
    print(line)
