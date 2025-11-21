import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ===== 1. Läs JSON-filerna =====
with open("bond.json", "r", encoding="utf-8") as f:
    bond_raw = json.load(f)

with open("stock.json", "r", encoding="utf-8") as f:
    stock_raw = json.load(f)

# ===== 2. Plocka ut dataSerie =====
bond_df = pd.DataFrame(bond_raw["dataSerie"])
stock_df = pd.DataFrame(stock_raw["dataSerie"])

# ===== 3. Tidsstämpel till datum =====
bond_df["date"] = pd.to_datetime(bond_df["x"], unit="ms")
stock_df["date"] = pd.to_datetime(stock_df["x"], unit="ms")

bond_df = bond_df.set_index("date").sort_index()
stock_df = stock_df.set_index("date").sort_index()

# ===== 4. Månadsavkastningar =====
bond_ret = bond_df["y"].resample("M").last().pct_change().dropna()
stock_ret = stock_df["y"].resample("M").last().pct_change().dropna()

# ===== 5. Matcha tidsperioder =====
df = pd.concat([bond_ret, stock_ret], axis=1).dropna()
df.columns = ["bond", "stock"]

# ===== 6. Statistik (årsvis) =====
r_bond = df["bond"].mean() * 12
r_stock = df["stock"].mean() * 12
vol_bond = df["bond"].std() * np.sqrt(12)
vol_stock = df["stock"].std() * np.sqrt(12)
corr = df["bond"].corr(df["stock"])

print("Bond annual return:", r_bond)
print("Stock annual return:", r_stock)
print("Bond volatility:", vol_bond)
print("Stock volatility:", vol_stock)
print("Correlation:", corr)

# ===== 7. Effektiv front (endast två tillgångar) =====
w = np.linspace(0, 1, 500)   # andel aktier
r_port = w * r_stock + (1 - w) * r_bond
vol_port = np.sqrt(
    w**2 * vol_stock**2 +
    (1 - w)**2 * vol_bond**2 +
    2 * w * (1 - w) * corr * vol_stock * vol_bond
)

# ===== 8. Minsta varians-portfölj =====
min_var_idx = np.argmin(vol_port)
w_min_var = w[min_var_idx]
r_min_var = r_port[min_var_idx]
vol_min_var = vol_port[min_var_idx]

print(f"Minsta varians-portfölj: {w_min_var*100:.1f}% aktier, "
      f"{(1-w_min_var)*100:.1f}% ränta")

# ===== 9. Sharpe-ratio-linje (Capital Allocation Line) =====
rf = 0.02  # antagen riskfri ränta, 2 %. Ändra om du vill.

sharpe = (r_port - rf) / vol_port
tan_idx = np.argmax(sharpe)
w_tan = w[tan_idx]
r_tan = r_port[tan_idx]
vol_tan = vol_port[tan_idx]

print(f"Tangensportfölj (max Sharpe): {w_tan*100:.1f}% aktier, "
      f"{(1-w_tan)*100:.1f}% ränta, Sharpe={sharpe[tan_idx]:.2f}")

# ===== 10. Plot =====
fig, ax = plt.subplots(figsize=(8, 6))

# Effektiv front
ax.plot(vol_port, r_port, label="Effektiv front")

# Punkter för specifika vikter
weights_to_mark = [0.0, 0.25, 0.5, 0.75, 1.0]
for w_mark in weights_to_mark:
    r_m = w_mark * r_stock + (1 - w_mark) * r_bond
    v_m = np.sqrt(
        w_mark**2 * vol_stock**2 +
        (1 - w_mark)**2 * vol_bond**2 +
        2 * w_mark * (1 - w_mark) * corr * vol_stock * vol_bond
    )
    ax.scatter(v_m, r_m, color="red")
    ax.annotate(
        f"{int(w_mark*100)}% aktier\n{int((1-w_mark)*100)}% ränta",
        (v_m, r_m),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8
    )

# Minsta varians-portfölj
ax.scatter(vol_min_var, r_min_var, color="green", s=80, zorder=5, label="Minsta varians")
ax.annotate(
    "Minsta\nvarians",
    (vol_min_var, r_min_var),
    textcoords="offset points",
    xytext=(5, -15),
    fontsize=8,
    color="green"
)

# Tangensportfölj (max Sharpe) + Sharpe-linje
ax.scatter(vol_tan, r_tan, color="purple", s=80, zorder=5, label="Max Sharpe")

# Sharpe-linje från riskfri punkt (0, rf) till tangeringsportföljen
ax.plot([0, vol_tan], [rf, r_tan], linestyle="--", color="purple", label="Sharpe-linje")

# Riskfri punkt (frivilligt)
ax.scatter(0, rf, color="black", s=40, zorder=5)
ax.annotate("Riskfri\nränta", (0, rf), textcoords="offset points",
            xytext=(5, 5), fontsize=8)

# Procentformat på axlarna
ax.xaxis.set_major_formatter(PercentFormatter(1.0))
ax.yaxis.set_major_formatter(PercentFormatter(1.0))

ax.set_xlabel("Volatilitet (risk)")
ax.set_ylabel("Förväntad årsavkastning")
ax.set_title("Effektiv front: OMX Stockholm vs Räntefond")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()
