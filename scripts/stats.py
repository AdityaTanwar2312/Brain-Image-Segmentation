from requirements import *
from clustering import *

df = pd.read_csv("segmentation_clusters.csv")

print("Data Loaded. Columns:", df.columns)

if "Cluster" not in df.columns:
    raise ValueError("'Cluster' column not found! Ensure clustering was performed.")

feature_cols = ["Volume", "Mean Intensity", "Std Intensity", "PCA1", "PCA2"]
anova_results = {}

# Perform ANOVA & Kruskal-Wallis Test
for feature in feature_cols:
    groups = [df[df["Cluster"] == cluster][feature] for cluster in df["Cluster"].unique()]
    
    if all(len(g) > 1 for g in groups):
        f_stat, p_value_anova = stats.f_oneway(*groups)  # ANOVA
        h_stat, p_value_kruskal = stats.kruskal(*groups)  # Kruskal-Wallis
    else:
        p_value_anova = np.nan
        p_value_kruskal = np.nan

    anova_results[feature] = {
        "ANOVA F-statistic": f_stat if not np.isnan(p_value_anova) else "N/A",
        "ANOVA p-value": p_value_anova,
        "Kruskal-Wallis H-statistic": h_stat if not np.isnan(p_value_kruskal) else "N/A",
        "Kruskal-Wallis p-value": p_value_kruskal
    }

anova_df = pd.DataFrame.from_dict(anova_results, orient="index")

anova_df.to_csv("anova_kruskal_results.csv", index=True)
print("\n📊 ANOVA & Kruskal-Wallis results saved to 'anova_kruskal_results.csv'.")

# Post-hoc Tukey's HSD test (only for significant ANOVA results)
for feature in feature_cols:
    if anova_results[feature]["ANOVA p-value"] < 0.05:
        print(f"\n📌 Performing Tukey's HSD test for {feature} (p < 0.05)...")
        tukey = pairwise_tukeyhsd(df[feature], df["Cluster"], alpha=0.05)
        print(tukey)

        with open(f"tukey_{feature}.txt", "w") as f:
            f.write(str(tukey))
        print(f"✅ Tukey's HSD results saved for {feature}.")


# %%
df = pd.read_csv("segmentation_clusters.csv")

sns.set_style("whitegrid")

metrics = ["Volume", "Mean Intensity", "Std Intensity"]
pca_components = ["PCA1", "PCA2"]

# Boxplots for Volume, Mean Intensity, and Std Intensity across clusters
plt.figure(figsize=(15, 5))
for i, metric in enumerate(metrics, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x="Cluster", y=metric, data=df, palette="Set2")
    plt.title(f"{metric} Across Clusters")
plt.tight_layout()
plt.savefig("boxplots.png")
plt.show()

# Swarm Plots for better distribution visualization
plt.figure(figsize=(15, 5))
for i, metric in enumerate(metrics, 1):
    plt.subplot(1, 3, i)
    sns.swarmplot(x="Cluster", y=metric, data=df, palette="husl", size=4)
    plt.title(f"{metric} Distribution in Clusters")
plt.tight_layout()
plt.savefig("swarm_plots.png")
plt.show()

# Scatter Plot for PCA1 vs PCA2 (Color-coded by Cluster)
plt.figure(figsize=(8, 6))
sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=df, palette="tab10", s=60, edgecolor="black")
plt.title("PCA1 vs PCA2 (Cluster Representation)")
plt.legend(title="Cluster")
plt.savefig("pca_scatter.png")
plt.show()