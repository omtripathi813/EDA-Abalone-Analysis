import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set visual styling defaults
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})

# ---------------------------------------------------------
# 0. Data Ingestion & Schema Definition
# ---------------------------------------------------------
COLUMNS = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole_weight",
    "Shucked_weight",
    "Viscera_weight",
    "Shell_weight",
    "Rings",
]

df = pd.read_csv("abalone.data", names=COLUMNS)

# Palette mapping
colors = {'M': '#1f77b4', 'F': '#ff7f0e', 'I': '#2ca02c'}

# ---------------------------------------------------------
# CHART 1: Weight Variance Across Sex Cohorts (Q1)
# Grouped Bar Chart of Median Weights
# ---------------------------------------------------------
weight_cols = ["Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight"]
weight_summary = df.groupby("Sex")[weight_cols].agg(["mean", "median"]).reindex(["M", "F", "I"])

print("--- 1. Weight Breakdown by Sex (Mean & Median) ---")
formatted_wt = weight_summary.copy()
formatted_wt.columns = [f"{col}_{stat}" for col, stat in formatted_wt.columns]
print(formatted_wt.round(4))

# Compute medians for plotting
medians_df = df.groupby("Sex")[weight_cols].median().reindex(["M", "F", "I"])

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(weight_cols))
width = 0.25

for i, sex in enumerate(['M', 'F', 'I']):
    ax.bar(x + (i - 1) * width, medians_df.loc[sex], width, label=sex, color=colors[sex])

ax.set_title("Q1: Median Weight Distribution by Tissue Type and Sex", fontsize=13, weight="bold")
ax.set_xlabel("Weight Metric")
ax.set_ylabel("Median Weight (grams)")
ax.set_xticks(x)
ax.set_xticklabels(weight_cols)
ax.legend(title="Sex Cohort")
ax.grid(True, linestyle="--", alpha=0.6)
plt.savefig("1_weight_variation_by_sex.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Saved: 1_weight_variation_by_sex.png")

# ---------------------------------------------------------
# CHART 2: Length vs. Whole Weight (Q2)
# Scatter Plot with Cubic Polynomial Fit (No Scipy Required)
# ---------------------------------------------------------
pearson_wt = df["Length"].corr(df["Whole_weight"])
# Spearman rank correlation calculated natively via pandas
spearman_wt = df["Length"].rank().corr(df["Whole_weight"].rank())

print("\n--- 2. Length vs Whole Weight Correlation ---")
print(f"Pearson r : {pearson_wt:.4f}")
print(f"Spearman ρ: {spearman_wt:.4f}")

plt.figure(figsize=(8, 5))
for sex in ['M', 'F', 'I']:
    sub = df[df['Sex'] == sex]
    plt.scatter(sub["Length"], sub["Whole_weight"], label=sex, color=colors[sex], alpha=0.4, s=25)

# Polynomial regression line (cubic curve)
poly_coeffs = np.polyfit(df["Length"], df["Whole_weight"], deg=3)
poly_curve = np.poly1d(poly_coeffs)
x_vals = np.linspace(df["Length"].min(), df["Length"].max(), 200)
plt.plot(x_vals, poly_curve(x_vals), color='black', linestyle='--', linewidth=2, label="Cubic Fit (W ∝ L³)")

plt.title(f"Q2: Length vs. Whole Weight (Pearson r={pearson_wt:.2f}, Spearman ρ={spearman_wt:.2f})", fontsize=13, weight="bold")
plt.xlabel("Length (mm)")
plt.ylabel("Whole Weight (grams)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("2_length_vs_whole_weight.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Saved: 2_length_vs_whole_weight.png")

# ---------------------------------------------------------
# CHART 3: Length vs. Rings / Age (Q3)
# Scatter Plot with Age Trajectory
# ---------------------------------------------------------
pearson_rings = df["Length"].corr(df["Rings"])
spearman_rings = df["Length"].rank().corr(df["Rings"].rank())

print("\n--- 3. Length vs Rings Correlation ---")
print(f"Pearson r : {pearson_rings:.4f}")
print(f"Spearman ρ: {spearman_rings:.4f}")

plt.figure(figsize=(8, 5))
for sex in ['M', 'F', 'I']:
    sub = df[df['Sex'] == sex]
    plt.scatter(sub["Rings"], sub["Length"], label=sex, color=colors[sex], alpha=0.35, s=25)

# Binned mean trendline across rings
ring_means = df.groupby("Rings")["Length"].mean()
plt.plot(ring_means.index, ring_means.values, color="crimson", linewidth=2.5, marker="o", markersize=4, label="Empirical Mean Growth")

plt.title(f"Q3: Shell Length vs. Rings (Pearson r={pearson_rings:.2f}, Spearman ρ={spearman_rings:.2f})", fontsize=13, weight="bold")
plt.xlabel("Rings (Count; Age = Rings + 1.5 years)")
plt.ylabel("Length (mm)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("3_length_vs_rings_age.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Saved: 3_length_vs_rings_age.png")

# ---------------------------------------------------------
# CHART 4: Morphometric Dimensions by Sex (Q4)
# Grouped Bar Chart of Dimensions
# ---------------------------------------------------------
size_cols = ["Length", "Diameter", "Height"]
size_summary = df.groupby("Sex")[size_cols].agg(["mean", "median", "std"]).reindex(["M", "F", "I"])

print("\n--- 4. Dimensional Metrics by Sex ---")
formatted_size = size_summary.copy()
formatted_size.columns = [f"{col}_{stat}" for col, stat in formatted_size.columns]
print(formatted_size.round(4))

size_medians = df.groupby("Sex")[size_cols].median().reindex(["M", "F", "I"])

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(size_cols))

for i, sex in enumerate(['M', 'F', 'I']):
    ax.bar(x + (i - 1) * width, size_medians.loc[sex], width, label=sex, color=colors[sex])

ax.set_title("Q4: Morphometric Dimensions by Sex Cohort (Median Values)", fontsize=13, weight="bold")
ax.set_xlabel("Shell Dimension")
ax.set_ylabel("Measurement (mm)")
ax.set_xticks(x)
ax.set_xticklabels(size_cols)
ax.legend(title="Sex Cohort")
ax.grid(True, linestyle="--", alpha=0.6)
plt.savefig("4_size_dimensions_by_sex.png", dpi=300, bbox_inches="tight")
plt.close()
print("✓ Saved: 4_size_dimensions_by_sex.png")

print("\n[Completed] All 4 analysis charts created and saved cleanly.")