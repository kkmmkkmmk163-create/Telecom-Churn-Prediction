"""
Generate all figures for the Telecom Churn Prediction project.
Reads data from ../data/data.csv and saves plots to ../figures/.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11

DATA_PATH = "../data/data.csv"
FIG_DIR = "../figures"

# ── Load & clean ──────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
# Churn as 0/1 for rate calculations
df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)

# ══════════════════════════════════════════════════════════════
# 1. Churn Distribution
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))
counts = df["Churn"].value_counts().sort_index()
bars = ax.bar(["No (retained)", "Yes (churned)"], counts.values,
              color=["#4C72B0", "#DD8452"], edgecolor="white", width=0.55)
for bar, val in zip(bars, counts.values):
    pct = val / counts.sum() * 100
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
            f"{val:,}\n({pct:.2f}%)", ha="center", va="bottom", fontsize=11)
ax.set_xlabel("Churn")
ax.set_ylabel("Count")
ax.set_title("Churn Distribution (n = 7,043)")
ax.set_ylim(0, counts.max() * 1.18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/churn_distribution.png", bbox_inches="tight")
plt.close()
print("✓ churn_distribution.png")

# ══════════════════════════════════════════════════════════════
# 2. Churn by Contract Type (stacked bar)
# ══════════════════════════════════════════════════════════════
contract_churn = pd.crosstab(df["Contract"], df["Churn"], normalize="index")
order = ["Month-to-month", "One year", "Two year"]
contract_churn = contract_churn.reindex(order)

fig, ax = plt.subplots(figsize=(8, 5))
contract_churn.plot(kind="bar", stacked=True, ax=ax,
                    color=["#4C72B0", "#DD8452"], edgecolor="white", width=0.6)
ax.set_xlabel("Contract Type")
ax.set_ylabel("Proportion")
ax.set_title("Churn Rate by Contract Type")
ax.legend(["No Churn", "Churn"], loc="upper right")
ax.set_xticklabels(order, rotation=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# annotate churn %
for i, idx in enumerate(contract_churn.index):
    churn_pct = contract_churn.loc[idx, "Yes"] * 100
    ax.text(i, churn_pct / 2 + (100 - churn_pct), f"{churn_pct:.1f}%",
            ha="center", va="center", color="white", fontweight="bold", fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/churn_by_contract.png", bbox_inches="tight")
plt.close()
print("✓ churn_by_contract.png")

# ══════════════════════════════════════════════════════════════
# 3. Churn by Internet Service
# ══════════════════════════════════════════════════════════════
internet_churn = df.groupby("InternetService")["Churn_bin"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(internet_churn.index, internet_churn.values * 100,
              color=["#DD8452", "#55A868", "#4C72B0"], edgecolor="white", width=0.55)
for bar, val in zip(bars, internet_churn.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val*100:.1f}%", ha="center", va="bottom", fontsize=11)
ax.set_xlabel("Internet Service")
ax.set_ylabel("Churn Rate (%)")
ax.set_title("Churn Rate by Internet Service Type")
ax.set_xticklabels(internet_churn.index, rotation=0)
ax.set_ylim(0, internet_churn.max() * 100 * 1.18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/churn_by_internet_service.png", bbox_inches="tight")
plt.close()
print("✓ churn_by_internet_service.png")

# ══════════════════════════════════════════════════════════════
# 4. Tenure vs Churn (line + binned bar)
# ══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# left: continuous line
tenure_churn = df.groupby("tenure")["Churn_bin"].mean()
ax1.plot(tenure_churn.index, tenure_churn.values * 100,
         color="#4C72B0", linewidth=1.5)
ax1.fill_between(tenure_churn.index, tenure_churn.values * 100,
                 alpha=0.15, color="#4C72B0")
ax1.set_xlabel("Tenure (Months)")
ax1.set_ylabel("Churn Rate (%)")
ax1.set_title("Churn Rate by Tenure (continuous)")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# right: binned bar
bins = [0, 12, 24, 36, 48, 60, 72]
labels = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)
tenure_group = df.groupby("tenure_group", observed=True)["Churn_bin"].mean()
bars = ax2.bar(labels, tenure_group.values * 100,
               color="#4C72B0", edgecolor="white", width=0.6)
for bar, val in zip(bars, tenure_group.values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{val*100:.1f}%", ha="center", va="bottom", fontsize=10)
ax2.set_xlabel("Tenure Group (Months)")
ax2.set_ylabel("Churn Rate (%)")
ax2.set_title("Churn Rate by Tenure Group")
ax2.set_ylim(0, tenure_group.max() * 100 * 1.2)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/tenure_vs_churn.png", bbox_inches="tight")
plt.close()
print("✓ tenure_vs_churn.png")

# ══════════════════════════════════════════════════════════════
# 5. Model Comparison (grouped bar)
# ══════════════════════════════════════════════════════════════
models = ["Logistic\nRegression", "Decision\nTree", "Random\nForest"]
accuracy  = [0.7743, 0.7509, 0.7466]
precision = [0.5622, 0.5273, 0.5164]
recall    = [0.6765, 0.5936, 0.7166]
f1        = [0.6141, 0.5585, 0.6002]
roc_auc   = [0.8409, np.nan, 0.8190]

x = np.arange(len(models))
width = 0.16
fig, ax = plt.subplots(figsize=(11, 6))
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
metrics = [("Accuracy", accuracy), ("Precision", precision),
           ("Recall", recall), ("F1", f1), ("ROC-AUC", roc_auc)]
for i, (name, vals) in enumerate(metrics):
    bars = ax.bar(x + (i - 2) * width, vals, width, label=name,
                  color=colors[i], edgecolor="white")
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, rotation=0)

ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.set_title("Final Model Comparison on Test Set")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0.45, 0.92)
ax.legend(loc="upper right", ncol=5, fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.axhline(y=0.7346, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.text(2.45, 0.74, "Baseline (majority class)", fontsize=8, color="gray", ha="right")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/model_comparison.png", bbox_inches="tight")
plt.close()
print("✓ model_comparison.png")

# ══════════════════════════════════════════════════════════════
# 6. Confusion Matrix (final Logistic Regression)
# ══════════════════════════════════════════════════════════════
cm_lr = np.array([[838, 197], [121, 253]])
fig, ax = plt.subplots(figsize=(6.5, 5.5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_lr,
                              display_labels=["No Churn", "Churn"])
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Logistic Regression — Confusion Matrix\n(threshold = 0.39)")
# add percentages
total = cm_lr.sum()
for i in range(2):
    for j in range(2):
        pct = cm_lr[i, j] / total * 100
        ax.text(j, i + 0.18, f"({pct:.1f}%)", ha="center", va="center",
                fontsize=10, color="gray")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/confusion_matrix.png", bbox_inches="tight")
plt.close()
print("✓ confusion_matrix.png")

# ══════════════════════════════════════════════════════════════
# 7. Learning Curve (Logistic Regression, ROC-AUC)
# ══════════════════════════════════════════════════════════════
train_sizes = [450, 1030, 1609, 2189, 2768, 3348, 3927, 4507]
train_auc   = [0.8889, 0.8681, 0.8637, 0.8616, 0.8612, 0.8567, 0.8541, 0.8496]
val_auc     = [0.8293, 0.8408, 0.8429, 0.8443, 0.8447, 0.8449, 0.8454, 0.8457]

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(train_sizes, train_auc, "o-", color="#4C72B0", label="Train AUC", linewidth=2)
ax.plot(train_sizes, val_auc, "s-", color="#DD8452", label="Validation AUC (CV=5)", linewidth=2)
ax.fill_between(train_sizes, train_auc, val_auc, alpha=0.12, color="gray")
ax.set_xlabel("Training Set Size")
ax.set_ylabel("ROC-AUC")
ax.set_title("Learning Curve — Logistic Regression")
ax.legend(loc="center right")
ax.set_ylim(0.82, 0.90)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
# annotate gap
ax.annotate("", xy=(4507, 0.8496), xytext=(4507, 0.8457),
            arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2))
ax.text(4300, 0.8476, "gap→0", fontsize=9, color="gray", ha="right")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/learning_curve.png", bbox_inches="tight")
plt.close()
print("✓ learning_curve.png")

print("\n✅ All 7 figures generated successfully.")
