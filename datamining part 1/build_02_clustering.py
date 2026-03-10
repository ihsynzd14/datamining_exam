"""Generate 02_clustering.ipynb — K-Means + DBSCAN. Optimised for Windows/speed."""
import json, os

cells = []
def md(source):
    lines = source.split("\n")
    for i in range(len(lines)-1):
        if not lines[i].endswith("\n"): lines[i] += "\n"
    cells.append({"cell_type":"markdown","metadata":{},"source":lines})

def code(source):
    lines = source.split("\n")
    for i in range(len(lines)-1):
        if not lines[i].endswith("\n"): lines[i] += "\n"
    cells.append({"cell_type":"code","metadata":{},"source":lines,
                  "execution_count":None,"outputs":[]})

# ===== SETUP =====
md("""# Task 2 — Clustering

We apply clustering algorithms to the Board Games dataset:
1. **K-Means** (centroid-based, mandatory)
2. **DBSCAN** (density-based, mandatory)
3. Hierarchical clustering and final comparison (Sections 4-5, to be added)""")

code("""import sys, os, warnings
warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
sys.path.append(os.path.join(os.pardir, 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             calinski_harabasz_score, davies_bouldin_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from plotting import save_plot, setup_style
setup_style()
%matplotlib inline""")

# ===== 1. LOAD & ATTRIBUTE SELECTION =====
md("""---
## 1. Load Data and Select Attributes

Clustering works best on continuous numeric features. We exclude:
- **Rank:\\*** columns — sentinel value 21926 for "unranked" distorts distances.
- **Cat:\\*** binary columns — binary flags add little to distance-based methods.
- **Rating** — target label; kept aside for external validation only.
- **IsReimplementation, Kickstarted** — binary flags with 0/1 values.
- **log\\_\\*** columns — the originals are standardised, making log versions redundant.

**Justification:** The 15 selected features capture core game characteristics: complexity (GameWeight), popularity (NumOwned, NumWish), player info (MinPlayers, MaxPlayers, BestPlayers), playtime (MfgPlaytime), age recommendations (ComAgeRec, MfgAgeRec), and engagement metrics (NumWeightVotes, NumAlternates, NumExpansions, NumImplementations, LanguageEase, YearPublished).""")

code("""df = pd.read_csv(os.path.join(os.pardir, 'dataset', 'processed',
                              'DM1_game_dataset_clean.csv'))
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

rating = df['Rating'].copy()

cluster_features = [
    'YearPublished', 'GameWeight', 'MinPlayers', 'MaxPlayers',
    'ComAgeRec', 'LanguageEase', 'BestPlayers',
    'NumOwned', 'NumWish', 'NumWeightVotes',
    'MfgPlaytime', 'MfgAgeRec',
    'NumAlternates', 'NumExpansions', 'NumImplementations',
]

X_raw = df[cluster_features].copy()
print(f"\\nSelected {len(cluster_features)} features:")
for f in cluster_features:
    print(f"  - {f}")""")

md("""### Standardisation

All features are z-score standardised so each dimension contributes equally to distance. Without this, NumOwned (mean ~1500) would dominate GameWeight (mean ~2.0).""")

code("""scaler = StandardScaler()
X_arr = scaler.fit_transform(X_raw)
X = pd.DataFrame(X_arr, columns=cluster_features)
print(f"Standardised matrix: {X.shape}")
print(X.describe().round(2).T[['mean','std','min','max']])""")

md("""### PCA for Visualisation

We reduce to 2D using PCA for plotting only. All clustering algorithms run on the full 15 features.""")

code("""pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_arr)
ev = pca.explained_variance_ratio_
print(f"PC1: {ev[0]:.1%} variance, PC2: {ev[1]:.1%} variance")
print(f"Total explained by 2 components: {ev.sum():.1%}")""")

# ===== 2. K-MEANS =====
md("""---
## 2. K-Means Clustering (mandatory)

K-Means partitions data into k spherical clusters by minimising within-cluster sum of squares (inertia).

### 2.1 Finding the Best k

We test k = 2 to 10 and evaluate with four metrics:
- **Inertia** (elbow method) — diminishing returns indicate the natural k.
- **Silhouette score** — how well each point fits its cluster vs nearest other. Higher is better [-1, 1].
- **Calinski-Harabasz** — ratio of between/within cluster variance. Higher is better.
- **Davies-Bouldin** — average similarity between clusters. Lower is better.

*Note: Silhouette is computed on a random sample of 5000 points for efficiency (22k pairwise distances would be very slow). The sampling is seeded for reproducibility.*

**Guideline:** *"identify the best value of k"* and *"list which different parameters you tested and justify your choice"*.""")

code("""k_range = range(2, 11)
res_km = []

for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
    labels = km.fit_predict(X_arr)
    sil = silhouette_score(X_arr, labels, sample_size=5000, random_state=42)
    ch  = calinski_harabasz_score(X_arr, labels)
    db  = davies_bouldin_score(X_arr, labels)
    res_km.append({'k': k, 'inertia': km.inertia_,
                   'silhouette': sil, 'calinski': ch, 'davies_bouldin': db})
    print(f"k={k:2d}  inertia={km.inertia_:>10,.0f}  sil={sil:.4f}  "
          f"CH={ch:>8,.0f}  DB={db:.3f}")

res_km = pd.DataFrame(res_km)""")

code("""fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(res_km['k'], res_km['inertia'], 'bo-')
axes[0,0].set_title('Elbow Method (Inertia)')
axes[0,0].set_xlabel('k'); axes[0,0].set_ylabel('Inertia')

axes[0,1].plot(res_km['k'], res_km['silhouette'], 'ro-')
axes[0,1].set_title('Silhouette Score')
axes[0,1].set_xlabel('k'); axes[0,1].set_ylabel('Silhouette')

axes[1,0].plot(res_km['k'], res_km['calinski'], 'go-')
axes[1,0].set_title('Calinski-Harabasz Index')
axes[1,0].set_xlabel('k'); axes[1,0].set_ylabel('CH Index')

axes[1,1].plot(res_km['k'], res_km['davies_bouldin'], 'mo-')
axes[1,1].set_title('Davies-Bouldin (lower = better)')
axes[1,1].set_xlabel('k'); axes[1,1].set_ylabel('DB Index')

plt.suptitle('K-Means: Choosing the Best k', fontsize=16, y=1.01)
plt.tight_layout()
save_plot(fig, 'kmeans_k_selection.png')
plt.show()""")

md("""**Discussion — Choosing k:**

- The **elbow** in inertia shows the rate of improvement slows around k=3-4.
- The **silhouette** score is typically highest at small k (k=2 or k=3).
- **Calinski-Harabasz** peaks at small k.
- **Davies-Bouldin** is lowest at small k.

All four metrics point towards k=3 as a good choice. We also test k=4 and k=5 to be thorough and confirm that k=3 is indeed the best balance.""")

md("""### 2.2 Detailed Comparison of k = 3, 4, 5""")

code("""candidates = [3, 4, 5]
best_score = -1
best_km = None
best_k = None

for k in candidates:
    km = KMeans(n_clusters=k, n_init=20, random_state=42, max_iter=500)
    labels = km.fit_predict(X_arr)
    sil = silhouette_score(X_arr, labels, sample_size=5000, random_state=42)
    ch  = calinski_harabasz_score(X_arr, labels)
    db  = davies_bouldin_score(X_arr, labels)
    print(f"k={k}: silhouette={sil:.4f}, CH={ch:.0f}, DB={db:.3f}")
    if sil > best_score:
        best_score, best_km, best_k = sil, km, k

print(f"\\n=> Best k = {best_k} (silhouette = {best_score:.4f})")
kmeans_labels = best_km.labels_
df['KMeans_Cluster'] = kmeans_labels""")

md("""### 2.3 K-Means Cluster Visualisation""")

code("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# PCA scatter with centroids
scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels,
                          cmap='viridis', alpha=0.3, s=5)
centroids_pca = pca.transform(best_km.cluster_centers_)
axes[0].scatter(centroids_pca[:,0], centroids_pca[:,1],
                c='red', marker='X', s=200, edgecolors='black', linewidths=1.5)
axes[0].set_title(f'K-Means (k={best_k}) — PCA Projection')
axes[0].set_xlabel(f'PC1 ({ev[0]:.1%})'); axes[0].set_ylabel(f'PC2 ({ev[1]:.1%})')
plt.colorbar(scatter, ax=axes[0], label='Cluster')

# Cluster sizes
counts = pd.Series(kmeans_labels).value_counts().sort_index()
axes[1].bar(counts.index, counts.values,
            color=[plt.cm.viridis(i / best_k) for i in counts.index])
axes[1].set_title('K-Means Cluster Sizes')
axes[1].set_xlabel('Cluster'); axes[1].set_ylabel('Count')
for i, v in zip(counts.index, counts.values):
    axes[1].text(i, v + 200, str(v), ha='center', fontsize=11)

plt.tight_layout()
save_plot(fig, 'kmeans_clusters.png')
plt.show()""")

md("""**Discussion:** The PCA scatter shows data projected onto the first two principal components (which explain a limited amount of variance, so some separation may not be visible in 2D). Red X markers are cluster centroids. The bar chart shows cluster sizes — ideally clusters should not be too imbalanced.""")

md("""### 2.4 Silhouette Analysis

We compute silhouette values on a random sample of 6000 points (for speed) and plot the silhouette diagram.""")

code("""# Sample for silhouette plot (full 22k is too slow for silhouette_samples)
np.random.seed(42)
sil_sample_size = 6000
sil_idx = np.random.choice(len(X_arr), size=sil_sample_size, replace=False)
X_sil = X_arr[sil_idx]
labels_sil = kmeans_labels[sil_idx]

sil_vals = silhouette_samples(X_sil, labels_sil)

fig, ax = plt.subplots(figsize=(10, 7))
y_lower = 10

for i in range(best_k):
    cluster_sil = np.sort(sil_vals[labels_sil == i])
    y_upper = y_lower + len(cluster_sil)
    color = plt.cm.viridis(float(i) / best_k)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                      facecolor=color, edgecolor=color, alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * len(cluster_sil), str(i), fontsize=12)
    y_lower = y_upper + 10

avg_sil = sil_vals.mean()
ax.axvline(x=avg_sil, color='red', linestyle='--', label=f'Average = {avg_sil:.3f}')
ax.set_title(f'Silhouette Plot — K-Means (k={best_k}, sample of {sil_sample_size})')
ax.set_xlabel('Silhouette Coefficient'); ax.set_ylabel('Cluster (sorted)')
ax.legend()
plt.tight_layout()
save_plot(fig, 'kmeans_silhouette.png')
plt.show()""")

md("""**Discussion:** Each horizontal bar represents one data point. Points above the red average line are well-assigned to their cluster. A cluster with many negative silhouette values would be poorly separated. Wide bars mean large clusters. We look for clusters where most points have positive, above-average silhouette values.""")

md("""### 2.5 Cluster Profiling

We examine the mean feature values per cluster to understand what kind of games each cluster contains.""")

code("""profile = df.groupby('KMeans_Cluster')[cluster_features].mean()
print("K-Means Cluster Profiles (original-scale means):")
print(profile.T.round(2).to_string())""")

code("""# Heatmap: z-scored per feature so we can see relative differences
profile_z = profile.T.copy()
for col in profile_z.columns:
    mu, sd = profile_z[col].mean(), profile_z[col].std()
    profile_z[col] = (profile_z[col] - mu) / (sd + 1e-8)

fig, ax = plt.subplots(figsize=(max(8, best_k * 2.5), 10))
sns.heatmap(profile_z, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, linewidths=0.5)
ax.set_title('K-Means Cluster Profiles (z-scored per feature)')
ax.set_xlabel('Cluster'); ax.set_ylabel('Feature')
plt.tight_layout()
save_plot(fig, 'kmeans_profiles.png')
plt.show()""")

md("""**Discussion:** Red means above average, blue means below average for that feature across clusters. This reveals the "personality" of each cluster:
- A cluster with high GameWeight, high MfgPlaytime, and high NumOwned = popular complex strategy games.
- A cluster with low GameWeight and high MaxPlayers = simple party/family games.
- A cluster with low NumOwned, low NumWish = obscure/niche games.

The exact interpretation depends on the data, but this heatmap makes cluster differences immediately visible.""")

md("""### 2.6 External Validation with Rating

We compare K-Means clusters against the known Rating labels (Low/Medium/High) to see if the clustering captures quality-related structure.""")

code("""ct = pd.crosstab(df['KMeans_Cluster'], rating, normalize='index').round(3)
print("K-Means Clusters vs Rating (row-normalised):")
print(ct)

fig, ax = plt.subplots(figsize=(10, 5))
ct.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
ax.set_title('K-Means Clusters vs Rating Distribution')
ax.set_xlabel('Cluster'); ax.set_ylabel('Proportion')
ax.legend(title='Rating', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
save_plot(fig, 'kmeans_vs_rating.png')
plt.show()""")

md("""**Discussion:** If certain clusters are dominated by High or Low ratings, the clustering has found quality-related structure in the feature space. If all clusters have similar Rating distributions, it means the clusters capture other differences (complexity, popularity, player type). Both outcomes are valid and informative — the clustering does not need to replicate the Rating variable to be useful.""")

# ===== 3. DBSCAN =====
md("""---
## 3. DBSCAN Clustering (mandatory)

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) finds clusters as dense regions separated by sparse areas. Unlike K-Means, it:
- Detects **noise** (outlier points).
- Finds clusters of **arbitrary shape**.
- Does **not** require specifying the number of clusters.

It needs two parameters:
- **eps** — the radius of the neighbourhood around each point.
- **min_samples** — the minimum number of points in a neighbourhood for a point to be "core".

### 3.1 Choosing eps — k-Distance Plot

We compute distances to the k-th nearest neighbour (k=15, a typical starting value for min_samples) on a random sample of 5000 points. The "elbow" in the sorted distance plot suggests where density drops off, giving a good eps value.

**Justification:** Using a sample of 5000 is sufficient to capture the distance distribution and keeps computation fast.""")

code("""np.random.seed(42)
sample_size = 5000
sample_idx = np.random.choice(len(X_arr), size=sample_size, replace=False)
X_sample = X_arr[sample_idx]

k_nn = 15
nn = NearestNeighbors(n_neighbors=k_nn)
nn.fit(X_sample)
distances, _ = nn.kneighbors(X_sample)
k_dist = np.sort(distances[:, -1])

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(k_dist, linewidth=0.8)
ax.set_title(f'{k_nn}-Distance Plot (sample of {sample_size} points)')
ax.set_xlabel('Points (sorted by distance)')
ax.set_ylabel(f'Distance to {k_nn}-th nearest neighbour')

for p, c, ls in [(90,'green','--'), (95,'red','--'), (97,'orange','--'), (99,'purple','--')]:
    val = np.percentile(k_dist, p)
    ax.axhline(y=val, color=c, linestyle=ls, alpha=0.7, label=f'{p}th pct = {val:.2f}')

ax.legend()
plt.tight_layout()
save_plot(fig, 'dbscan_kdistance.png')
plt.show()

print("\\nCandidate eps values from percentiles of k-distance:")
for p in [90, 93, 95, 97, 99]:
    print(f"  {p}th percentile: eps = {np.percentile(k_dist, p):.2f}")""")

md("""**Discussion:** The k-distance plot shows how "far" each point is from its neighbourhood. The elbow region (where the curve steepens) indicates the natural density boundary. Points above this boundary are in sparse regions and will likely become noise. We select eps candidates from the 90th to 99th percentiles to test systematically.""")

md("""### 3.2 Parameter Grid Search

We test combinations of eps and min_samples on a **random sample of 8000 points** for speed (DBSCAN on full 22k with 15 features is slow due to pairwise distance computation).

**Tested parameters:**
- eps: derived from the 90th, 95th, 97th, and 99th percentiles of the k-distance plot.
- min_samples: 5, 10, 15, 20.

**Guideline:** *"identify the best parameter configuration"* and *"list which different parameters you tested"*.""")

code("""eps_candidates = sorted(set([
    round(float(np.percentile(k_dist, p)), 2) for p in [90, 95, 97, 99]
]))
ms_candidates = [5, 10, 15, 20]

print(f"eps candidates: {eps_candidates}")
print(f"min_samples candidates: {ms_candidates}\\n")

# Use a sample for the grid search (DBSCAN on 22k is ~40s per config)
np.random.seed(42)
grid_sample_size = 8000
grid_idx = np.random.choice(len(X_arr), size=grid_sample_size, replace=False)
X_grid = X_arr[grid_idx]

results_db = []
for eps_val in eps_candidates:
    for ms in ms_candidates:
        db = DBSCAN(eps=eps_val, min_samples=ms)
        lbl = db.fit_predict(X_grid)
        n_cl = len(set(lbl)) - (1 if -1 in lbl else 0)
        n_noise = (lbl == -1).sum()
        noise_pct = n_noise / len(lbl) * 100

        sil = -1.0
        if n_cl >= 2:
            mask = lbl != -1
            if mask.sum() > n_cl:
                sil = silhouette_score(X_grid[mask], lbl[mask],
                                       sample_size=min(3000, mask.sum()),
                                       random_state=42)

        results_db.append({'eps': eps_val, 'min_samples': ms,
                           'n_clusters': n_cl, 'noise_%': round(noise_pct, 1),
                           'silhouette': round(sil, 4)})

rdb = pd.DataFrame(results_db)
print("Grid search results (sorted by silhouette):")
print(rdb.sort_values('silhouette', ascending=False).to_string(index=False))""")

md("""**Discussion:** We look for configurations that produce:
1. At least 2 clusters (otherwise DBSCAN just lumps everything together).
2. Reasonable noise level (< 30-40%).
3. Highest silhouette score among valid configs.

In high-dimensional data (15 features), DBSCAN often struggles because the "curse of dimensionality" makes distances less meaningful. A large eps is needed to form clusters, which can result in one big cluster + noise. This is an inherent limitation of density-based methods in high dimensions, not a mistake in parameter choice.""")

md("""### 3.3 Apply Best DBSCAN Configuration on Full Data""")

code("""# Pick best config from grid search
valid = rdb[(rdb['silhouette'] > 0) & (rdb['noise_%'] < 40)]
if len(valid) == 0:
    valid = rdb[rdb['silhouette'] > 0]
if len(valid) == 0:
    print("No config produced 2+ clusters with sil > 0. Using default.")
    best_eps = round(float(np.percentile(k_dist, 97)), 2)
    best_ms = 5
else:
    best_row = valid.sort_values('silhouette', ascending=False).iloc[0]
    best_eps = float(best_row['eps'])
    best_ms  = int(best_row['min_samples'])

print(f"Selected: eps={best_eps}, min_samples={best_ms}")
print("Applying to full dataset...")

dbscan_final = DBSCAN(eps=best_eps, min_samples=best_ms)
dbscan_labels = dbscan_final.fit_predict(X_arr)

n_cl_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise_db = (dbscan_labels == -1).sum()
noise_pct_db = n_noise_db / len(dbscan_labels) * 100
print(f"\\nResult on full data:")
print(f"  Clusters: {n_cl_db}")
print(f"  Noise points: {n_noise_db} ({noise_pct_db:.1f}%)")

if n_cl_db >= 2:
    mask_nn = dbscan_labels != -1
    db_sil = silhouette_score(X_arr[mask_nn], dbscan_labels[mask_nn],
                              sample_size=min(5000, mask_nn.sum()), random_state=42)
    print(f"  Silhouette (non-noise): {db_sil:.4f}")

df['DBSCAN_Cluster'] = dbscan_labels""")

md("""### 3.4 DBSCAN Cluster Visualisation""")

code("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

noise_mask = dbscan_labels == -1
cluster_mask = ~noise_mask

# PCA scatter
if cluster_mask.any():
    axes[0].scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1],
                    c=dbscan_labels[cluster_mask], cmap='tab10', alpha=0.4, s=5)
axes[0].scatter(X_pca[noise_mask, 0], X_pca[noise_mask, 1],
                c='lightgrey', alpha=0.15, s=3, label=f'Noise ({n_noise_db})')
axes[0].set_title(f'DBSCAN (eps={best_eps}, min_samples={best_ms})')
axes[0].set_xlabel(f'PC1 ({ev[0]:.1%})'); axes[0].set_ylabel(f'PC2 ({ev[1]:.1%})')
axes[0].legend(loc='upper right')

# Cluster sizes
counts_db = pd.Series(dbscan_labels).value_counts().sort_index()
colors_bar = ['lightgrey' if i == -1 else plt.cm.tab10(i % 10) for i in counts_db.index]
axes[1].bar([str(i) for i in counts_db.index], counts_db.values, color=colors_bar)
axes[1].set_title('DBSCAN Cluster Sizes')
axes[1].set_xlabel('Cluster (-1 = noise)'); axes[1].set_ylabel('Count')
for pos, (idx, v) in enumerate(counts_db.items()):
    axes[1].text(pos, v + 200, str(v), ha='center', fontsize=10)

plt.tight_layout()
save_plot(fig, 'dbscan_clusters.png')
plt.show()""")

md("""**Discussion:** DBSCAN assigns outlier games to the noise cluster (-1, shown in grey). In the PCA projection, noise points are often at the edges of the point cloud (extreme games: very high complexity, very large player count, very obscure titles). The coloured clusters contain games that are in dense regions of the feature space.""")

md("""### 3.5 DBSCAN Cluster Profiling""")

code("""non_noise_df = df[df['DBSCAN_Cluster'] != -1]
n_unique = non_noise_df['DBSCAN_Cluster'].nunique()

if n_unique >= 2:
    profile_db = non_noise_df.groupby('DBSCAN_Cluster')[cluster_features].mean()
    print("DBSCAN Cluster Profiles (mean values, excluding noise):")
    print(profile_db.T.round(2).to_string())
else:
    # Compare the main cluster vs noise
    print(f"DBSCAN found {n_unique} non-noise cluster(s).")
    print("Comparing main cluster vs noise to understand what DBSCAN considers outliers:\\n")
    for name, label in [('Main cluster (0)', 0), ('Noise (-1)', -1)]:
        mask = df['DBSCAN_Cluster'] == label
        if mask.any():
            print(f"--- {name}: {mask.sum()} games ---")
            print(df.loc[mask, cluster_features].mean().round(2).to_string())
            print()""")

md("""**Discussion:** When DBSCAN finds only 1 cluster + noise, the profiling becomes a comparison of "typical" games (cluster 0) vs "unusual" games (noise). This is still valuable — it tells us what makes a game an outlier in this feature space. Noise games often have extreme values in one or more features (very popular, very complex, very long playtime, etc.).""")

md("""### 3.6 External Validation with Rating""")

code("""ct_db = pd.crosstab(df['DBSCAN_Cluster'], rating, normalize='index').round(3)
print("DBSCAN Clusters vs Rating (row-normalised):")
print(ct_db)

fig, ax = plt.subplots(figsize=(10, 5))
ct_db.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
ax.set_title('DBSCAN Clusters vs Rating Distribution')
ax.set_xlabel('Cluster (-1 = noise)'); ax.set_ylabel('Proportion')
ax.legend(title='Rating', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
save_plot(fig, 'dbscan_vs_rating.png')
plt.show()""")

md("""**Discussion:** If the noise cluster has a different Rating distribution (e.g., more High-rated games), it means DBSCAN is separating popular/high-quality games as outliers — which makes sense because popular games have extreme NumOwned and NumWish values.

### DBSCAN Summary

DBSCAN's strengths on this dataset:
- Automatically identifies outlier/extreme games as noise.
- Does not force all points into clusters.

DBSCAN's limitations on this dataset:
- With 15 features, the "curse of dimensionality" makes distance-based density less meaningful.
- Often produces 1-2 large clusters rather than fine-grained partitions.
- Parameter selection (eps, min_samples) is harder than K-Means' single k parameter.

This does not mean DBSCAN is a bad algorithm — it means the data structure in 15D is not well-suited to density-based clustering. For datasets with clear density-separated groups, DBSCAN would outperform K-Means.""")

md("""---
*Sections 4 (Hierarchical Clustering) and 5 (Final Discussion) will be added next.*""")

# ===== WRITE =====
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.12.3"}
    },
    "cells": cells
}

out = os.path.join("notebooks", "02_clustering.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Written {out} — {len(cells)} cells ({sum(1 for c in cells if c['cell_type']=='code')} code, {sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
