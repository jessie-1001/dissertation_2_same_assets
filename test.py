import pandas as pd
res = pd.read_csv("garch_copula_all_results.csv", parse_dates=["Date"]).set_index("Date")

# 1. 每天 VaR 序列排序应满足:  |VaR_Clayton| >= |VaR_Gaussian|
print((res["Clayton_VaR"].abs() >= res["Gaussian_VaR"].abs()).mean())
# 若接近 1.0 → 条件基本成立

# 2. ES 应绝对不小于 VaR
for fam in ["Gaussian","StudentT","Clayton","Gumbel"]:
    pct = (res[f"{fam}_ES"] <= res[f"{fam}_VaR"]).mean()
    print(f"{fam}: ES<=VaR ratio = {pct:.3f}  (理想=1.0)")

# 3. 检查权重是否落在 30‑70% 区间
print(res[["Gaussian_wSPX","StudentT_wSPX"]].describe().loc[["min","max"]])