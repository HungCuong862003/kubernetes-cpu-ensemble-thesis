import csv
rows = [r for r in csv.DictReader(open("bcf_out_5model/bcf_pairs.csv")) if r["model"]=="Granite-TTM"]
print(f"{'dataset':10s} {'horizon':8s} {'pred':>5s} {'wins':>5s} {'delta_pp':>10s}  agree?")
for r in rows:
    p = int(r["predicate"]); w = int(r["ml_wins"])
    agree = "OK" if p == w else "MISS"
    print(f"{r['dataset']:10s} {r['horizon']:8s} {p:>5d} {w:>5d} {float(r['delta_pp']):>10.3f}  {agree}")
