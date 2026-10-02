import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Styling for academic reports
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 1. Log-Scaled Histogram (Simulated matching data)
# Replace 'df["char_length"]' with your actual dataframe column
sns.histplot(df["char_length"], log_scale=True, kde=True, ax=axes[0], color="#2b5c8f")
axes[0].axvline(332, color="#d9534f", linestyle="--", linewidth=2, label="Median: 332")
axes[0].axvline(1098, color="#5cb85c", linestyle="-", linewidth=2, label="Mean: 1098")
axes[0].set_title("Log-Scaled Character Length Distribution", fontsize=14, weight="bold")
axes[0].set_xlabel("Character Length (Log Scale)", fontsize=12)
axes[0].set_ylabel("Count", fontsize=12)
axes[0].legend(fontsize=11)

# 2. Horizontal Box Plot
sns.boxplot(x=df["char_length"], ax=axes[1], color="#4a90e2", fliersize=3)
axes[1].set_xscale("log") # Keep scales consistent
axes[1].set_title("Biographical Block Percentile Spread", fontsize=14, weight="bold")
axes[1].set_xlabel("Character Length (Log Scale)", fontsize=12)

# Explicitly marking your calculated percentiles on the box plot
percentiles = {68: 'P10', 103: 'P25', 332: 'Med', 2240: 'P75', 3000: 'P90'}
for val, label in percentiles.items():
    axes[1].text(val, -0.4, f"{label}\n({val})", ha='center', va='bottom', 
                 color='#333333', fontsize=9, weight='bold')

plt.tight_layout()
plt.savefig("eda_text_distribution.png", dpi=300)
plt.show()