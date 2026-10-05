"""
Abalone Characteristics Analysis - full EDA script
Reproduces every table and chart used in the report.

Usage: put abalone.data (from UCI) in the same folder, then run:
    python abalone_eda_full.py
Charts are saved in the 'figures' folder; tables are printed to the console.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------
# 0. Load data (file has no header, so we supply the names)
# ---------------------------------------------------------
COLUMNS = ["Sex", "Length", "Diameter", "Height", "Whole_weight",
           "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]
NUM = COLUMNS[1:]
df = pd.read_csv("abalone.data", names=COLUMNS)
os.makedirs("figures", exist_ok=True)

# Sea colour palette
NAVY, OCEAN, TEAL, AQUA, CORAL, SEAFOAM = "#0B3C5D", "#1D6FA3", "#1FA5A5", "#8FD3D6", "#E07A5F", "#5FC9B8"
colors = {"M": OCEAN, "F": CORAL, "I": SEAFOAM}
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": NAVY, "axes.labelcolor": NAVY, "xtick.color": NAVY,
    "ytick.color": NAVY, "axes.titlecolor": NAVY, "axes.facecolor": "#FBFEFE", "axes.grid": True,
    "grid.color": "#CFE6E8", "grid.linestyle": "--", "axes.axisbelow": True, "figure.autolayout": True})


def save(name):
    plt.savefig(f"figures/{name}.png", dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------
# 5. Data quality assessment
# ---------------------------------------------------------
print("--- Shape:", df.shape)
print("--- Data types:\n", df.dtypes)
print("--- Missing values:\n", df.isnull().sum())
print("--- Duplicate rows:", df.duplicated().sum())
print("--- Sex categories:", df["Sex"].unique())
print("--- Records with Height == 0:\n", df[df["Height"] == 0])
print("--- Records with Height > 0.4:\n", df[df["Height"] > 0.4])
print("--- Shell heavier than whole:", (df["Shell_weight"] > df["Whole_weight"]).sum())
print("--- Shucked heavier than whole:", (df["Shucked_weight"] > df["Whole_weight"]).sum())
parts = df["Shucked_weight"] + df["Viscera_weight"] + df["Shell_weight"]
print("--- Parts heavier than whole:", (parts > df["Whole_weight"]).sum())

# ---------------------------------------------------------
# 6. Descriptive statistics
# ---------------------------------------------------------
desc = df[NUM].describe().T
desc["median"] = df[NUM].median()
desc["skew"] = df[NUM].skew()
desc["cv_%"] = desc["std"] / desc["mean"] * 100
print("\n--- Descriptive statistics:\n", desc.round(4))
print("\n--- Sex counts:\n", df["Sex"].value_counts())
print((df["Sex"].value_counts(normalize=True) * 100).round(2))

# ---------------------------------------------------------
# 7. Univariate analysis (outliers by 1.5 x IQR rule)
# ---------------------------------------------------------
print("\n--- Outliers (IQR rule):")
for c in NUM:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    low = (df[c] < q1 - 1.5 * iqr).sum()
    high = (df[c] > q3 + 1.5 * iqr).sum()
    print(f"{c:15s} low={low:4d} high={high:4d} share={(low + high) / len(df) * 100:.2f}%")

fig, axs = plt.subplots(3, 3, figsize=(8.2, 7.4))
axs = axs.ravel()
for i, c in enumerate(NUM):
    ax = axs[i]
    bins = np.arange(0.5, 30.5, 1) if c == "Rings" else 30
    ax.hist(df[c], bins=bins, color=OCEAN if i % 2 == 0 else TEAL, edgecolor="white")
    ax.axvline(df[c].mean(), color=CORAL, lw=1.8, label=f"mean {df[c].mean():.3f}")
    ax.axvline(df[c].median(), color=NAVY, lw=1.5, ls="--", label=f"median {df[c].median():.3f}")
    ax.set_title(c.replace("_", " "), fontsize=10, weight="bold")
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_ylabel("Frequency", fontsize=8)
sc = df["Sex"].value_counts().reindex(["M", "F", "I"])
axs[8].bar(sc.index, sc.values, color=[colors[s] for s in sc.index], edgecolor="white")
for i, v in enumerate(sc.values):
    axs[8].text(i, v + 20, str(v), ha="center", fontsize=8, color=NAVY)
axs[8].set_title("Sex (M / F / I)", fontsize=10, weight="bold")
axs[8].set_ylim(0, 1750)
fig.suptitle("Distributions of All Variables", fontsize=13, weight="bold", color=NAVY)
save("hist_grid")

fig, axs = plt.subplots(2, 4, figsize=(8.2, 4.8))
axs = axs.ravel()
for i, c in enumerate(NUM):
    bp = axs[i].boxplot(df[c], patch_artist=True, widths=0.55,
                        flierprops=dict(marker="o", markersize=2.5, markerfacecolor=CORAL, markeredgecolor=CORAL, alpha=0.6),
                        medianprops=dict(color=NAVY, lw=1.8))
    bp["boxes"][0].set(facecolor=AQUA, edgecolor=NAVY)
    axs[i].set_title(c.replace("_", " "), fontsize=9, weight="bold")
    axs[i].set_xticks([])
fig.suptitle("Boxplots Showing Spread and Outliers", fontsize=12, weight="bold", color=NAVY)
save("box_grid")

# ---------------------------------------------------------
# 9. Grouping and aggregation  +  Q1 and Q4 charts
# ---------------------------------------------------------
order = ["M", "F", "I"]
print("\n--- Mean of every variable by Sex:\n", df.groupby("Sex")[NUM].mean().reindex(["F", "M", "I"]).round(4))
print("\n--- Median of every variable by Sex:\n", df.groupby("Sex")[NUM].median().reindex(["F", "M", "I"]).round(4))
print("\n--- Rings by Sex:\n", df.groupby("Sex")["Rings"].agg(["count", "mean", "median", "std", "min", "max"]).round(3))

width = 0.25
weight_cols = ["Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight"]
med = df.groupby("Sex")[weight_cols].median().reindex(order)
fig, ax = plt.subplots(figsize=(8, 4.6))
x = np.arange(len(weight_cols))
for i, s in enumerate(order):
    ax.bar(x + (i - 1) * width, med.loc[s], width, label=s, color=colors[s], edgecolor="white")
ax.set_title("Q1: Median Weight Distribution by Tissue Type and Sex", weight="bold")
ax.set_xticks(x); ax.set_xticklabels(weight_cols)
ax.set_xlabel("Weight Metric"); ax.set_ylabel("Median Weight (grams)"); ax.legend(title="Sex Cohort")
save("q1")

size_cols = ["Length", "Diameter", "Height"]
sm = df.groupby("Sex")[size_cols].median().reindex(order)
fig, ax = plt.subplots(figsize=(8, 4.6))
x = np.arange(len(size_cols))
for i, s in enumerate(order):
    ax.bar(x + (i - 1) * width, sm.loc[s], width, label=s, color=colors[s], edgecolor="white")
ax.set_title("Q4: Morphometric Dimensions by Sex Cohort (Median Values)", weight="bold")
ax.set_xticks(x); ax.set_xticklabels(size_cols)
ax.set_xlabel("Shell Dimension"); ax.set_ylabel("Measurement (scaled units)"); ax.legend(title="Sex Cohort")
save("q4")

# Age groups
df["AgeGroup"] = pd.cut(df["Rings"], [0, 7, 10, 15, 30],
                        labels=["Young\n(1-7 rings)", "Adult\n(8-10)", "Mature\n(11-15)", "Old\n(16+)"])
print("\n--- By age group:\n", df.groupby("AgeGroup", observed=True)[["Length", "Whole_weight", "Shell_weight"]].agg(["count", "mean"]).round(3))
ct = pd.crosstab(df["AgeGroup"], df["Sex"])[["M", "F", "I"]]
print("\n--- Sex counts per age group:\n", ct)
pct = ct.div(ct.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(7.4, 4.4))
bottom = np.zeros(4)
for s in order:
    ax.bar(range(4), pct[s], bottom=bottom, color=colors[s], edgecolor="white", label=s)
    for i, v in enumerate(pct[s]):
        if v > 6:
            ax.text(i, bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=9, weight="bold")
    bottom += pct[s].values
ax.set_xticks(range(4)); ax.set_xticklabels(pct.index)
ax.set_ylabel("Share of abalone (%)"); ax.set_ylim(0, 100)
ax.set_title("Sex Composition Across Age Groups", weight="bold")
ax.legend(title="Sex", loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
save("agegroup_sex")

# ---------------------------------------------------------
# 8. Bivariate analysis  (Q2, Q3 and rings by sex)
# ---------------------------------------------------------
pearson_wt = df["Length"].corr(df["Whole_weight"])
spearman_wt = df["Length"].rank().corr(df["Whole_weight"].rank())
print(f"\n--- Length vs Whole weight: Pearson {pearson_wt:.4f}, Spearman {spearman_wt:.4f}")
fig, ax = plt.subplots(figsize=(8, 4.8))
for s in order:
    sub = df[df["Sex"] == s]
    ax.scatter(sub["Length"], sub["Whole_weight"], s=18, alpha=0.4, color=colors[s], label=s, edgecolors="none")
curve = np.poly1d(np.polyfit(df["Length"], df["Whole_weight"], 3))
xv = np.linspace(df["Length"].min(), df["Length"].max(), 200)
ax.plot(xv, curve(xv), color=NAVY, ls="--", lw=2.2, label="Cubic polynomial fit")
ax.set_title(f"Q2: Length vs. Whole Weight (Pearson r={pearson_wt:.2f}, Spearman \u03c1={spearman_wt:.2f})", weight="bold")
ax.set_xlabel("Length (scaled units)"); ax.set_ylabel("Whole Weight (grams)"); ax.legend()
save("q2")

pearson_r = df["Length"].corr(df["Rings"])
spearman_r = df["Length"].rank().corr(df["Rings"].rank())
print(f"--- Length vs Rings: Pearson {pearson_r:.4f}, Spearman {spearman_r:.4f}")
fig, ax = plt.subplots(figsize=(8, 4.8))
for s in order:
    sub = df[df["Sex"] == s]
    ax.scatter(sub["Rings"], sub["Length"], s=18, alpha=0.35, color=colors[s], label=s, edgecolors="none")
ring_means = df.groupby("Rings")["Length"].mean()
ax.plot(ring_means.index, ring_means.values, color=NAVY, lw=2.6, marker="o", ms=4, label="Mean length per ring count")
ax.set_title(f"Q3: Shell Length vs. Rings (Pearson r={pearson_r:.2f}, Spearman \u03c1={spearman_r:.2f})", weight="bold")
ax.set_xlabel("Rings (Count; Age = Rings + 1.5 years)"); ax.set_ylabel("Length (scaled units)"); ax.legend(loc="lower right")
save("q3")

fig, ax = plt.subplots(figsize=(7.2, 4.2))
bp = ax.boxplot([df[df["Sex"] == s]["Rings"] for s in order], patch_artist=True, widths=0.5,
                medianprops=dict(color=NAVY, lw=2),
                flierprops=dict(marker="o", markersize=3, alpha=0.5, markeredgecolor=CORAL, markerfacecolor=CORAL))
for b, s in zip(bp["boxes"], order):
    b.set(facecolor=colors[s], alpha=0.85, edgecolor=NAVY)
ax.set_xticklabels(["Male (M)", "Female (F)", "Infant (I)"])
ax.set_ylabel("Rings (age indicator)"); ax.set_title("Rings Distribution by Sex", weight="bold")
save("rings_by_sex")

# ---------------------------------------------------------
# 10. Correlation analysis
# ---------------------------------------------------------
corr = df[NUM].corr()
print("\n--- Correlation matrix:\n", corr.round(3))
print("\n--- Correlation with Rings:\n", corr["Rings"].sort_values(ascending=False).round(3))
heat = LinearSegmentedColormap.from_list("seaheat", ["#F7FCFD", AQUA, TEAL, OCEAN, NAVY])
fig, ax = plt.subplots(figsize=(7.4, 6.2))
ax.grid(False)
im = ax.imshow(corr.values, cmap=heat, vmin=0.4, vmax=1)
ax.set_xticks(range(8)); ax.set_yticks(range(8))
labels = [c.replace("_", "\n") for c in NUM]
ax.set_xticklabels(labels, fontsize=8); ax.set_yticklabels(labels, fontsize=8)
for i in range(8):
    for j in range(8):
        v = corr.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, weight="bold",
                color="white" if v > 0.78 else NAVY)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("Pearson correlation", color=NAVY)
ax.set_title("Correlation Heatmap of Numerical Variables", weight="bold", pad=12)
save("heatmap")

print("\n[Completed] Tables printed above; 9 charts saved in the 'figures' folder.")
