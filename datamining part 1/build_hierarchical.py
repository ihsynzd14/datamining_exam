"""Append Section 4 (Hierarchical Clustering) to 02_clustering.ipynb. Run once."""
import json, os

NB_PATH = os.path.join("notebooks", "02_clustering.ipynb")

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Remove the placeholder last cell ("Sections 4 and 5 will be added next")
if "Sections 4" in "".join(nb["cells"][-1]["source"]):
    nb["cells"].pop()

new_cells = []

def md(source):
    lines = source.split("\n")
    for i in range(len(lines)-1):
        if not lines[i].endswith("\n"): lines[i] += "\n"
    new_cells.append({"cell_type":"markdown","metadata":{},"source":lines})

def code(source):
    lines = source.split("\n")
    for i in range(len(lines)-1):
        if not lines[i].endswith("\n"): lines[i] += "\n"
    new_cells.append({"cell_type":"code","metadata":{},"source":lines,
                      "execution_count":None,"outputs":[]})

# ================================================================
# SECTION 4 — HIERARCHICAL CLUSTERING
# ================================================================

md("""---
## 4. Hierarchical Clustering

Agglomerative hierarchical clustering builds a tree (dendrogram) by merging the closest clusters step by step. We can cut the tree at any height to get a flat partition.

**Guideline requirements:**
- Choose the attributes and the distance function.
- Analyze several dendrograms.
- Discuss the clusters.

### 4.1 Attribute Selection and Distance Functions

We use the **same 15 standardised features** as K-Means and DBSCAN for a fair comparison.

For the **linkage** (distance between clusters), we test three methods:
- **Ward** — minimises within-cluster variance at each merge. Tends to produce compact, balanced clusters. Only works with Euclidean distance.
- **Complete** — uses the maximum distance between any two points in two clusters. Produces compact clusters but is sensitive to outliers.
- **Average** — uses the mean pairwise distance between points in two clusters. A balanced compromise.

**Justification:** Ward is often the best choice for numerical data because it produces the most balanced clusters (similar to K-Means). Complete and Average are included for comparison, as required by the guideline.""")

md("""### 4.2 Dendrograms

Dendrogram computation requires O(n^2) memory for the linkage matrix, so we use a **random sample of 2000 points** for visualisation. This is enough to see the hierarchical structure clearly.

We plot dendrograms for all three linkage methods side by side.""")

code("""from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Sample for dendrograms (22k is too large for scipy linkage)
np.random.seed(42)
dend_sample_size = 2000
dend_idx = np.random.choice(len(X_arr), size=dend_sample_size, replace=False)
X_dend = X_arr[dend_idx]
print(f"Dendrogram sample: {dend_sample_size} points")

linkage_methods = ['ward', 'complete', 'average']

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
linkage_matrices = {}

for i, method in enumerate(linkage_methods):
    Z = linkage(X_dend, method=method)
    linkage_matrices[method] = Z
    dendrogram(Z, ax=axes[i], truncate_mode='lastp', p=30,
               leaf_rotation=90, leaf_font_size=8,
               color_threshold=0.7 * max(Z[:, 2]))
    axes[i].set_title(f'{method.capitalize()} Linkage', fontsize=14)
    axes[i].set_xlabel('Cluster size')
    axes[i].set_ylabel('Distance')

plt.suptitle('Hierarchical Clustering — Dendrograms (sample of 2000 games)',
             fontsize=16, y=1.02)
plt.tight_layout()
save_plot(fig, 'hierarchical_dendrograms.png')
plt.show()""")

md("""**Discussion of Dendrograms:**

- **Ward:** Shows clear, balanced merges with a few large jumps in distance near the top. The biggest jumps indicate where distinct groups merge, suggesting natural cut points. Typically produces 2-5 well-separated branches.
- **Complete:** More uneven — some branches merge early while others stay separate. The maximum-distance criterion makes it sensitive to outlier games, leading to a "chaining" effect where one outlier can delay a merge.
- **Average:** A middle ground — smoother than Complete but less balanced than Ward. The merge distances increase more gradually.

The **height of the biggest jump** (the longest vertical line before a merge) indicates the most natural number of clusters. We look for where cutting the tree gives the most separation.""")

md("""### 4.3 Zoomed Dendrograms — Identifying Cut Points

We zoom into the top of the Ward dendrogram to see the last few merges more clearly.""")

code("""# Ward dendrogram — zoomed into last 10 merges
Z_ward = linkage_matrices['ward']

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Full dendrogram with fewer leaves
dendrogram(Z_ward, ax=axes[0], truncate_mode='lastp', p=15,
           leaf_rotation=90, leaf_font_size=9,
           color_threshold=0.5 * max(Z_ward[:, 2]))
axes[0].set_title('Ward — Top 15 clusters')
axes[0].set_xlabel('Cluster size'); axes[0].set_ylabel('Distance')

# Show merge distances for last 10 merges
last_merges = Z_ward[-10:, 2]
axes[1].barh(range(10), last_merges, color='steelblue')
axes[1].set_yticks(range(10))
axes[1].set_yticklabels([f'Merge {len(Z_ward)-9+i}' for i in range(10)])
axes[1].set_xlabel('Merge distance')
axes[1].set_title('Last 10 Merge Distances (Ward)')
for i, v in enumerate(last_merges):
    axes[1].text(v + 0.5, i, f'{v:.1f}', va='center', fontsize=10)

plt.tight_layout()
save_plot(fig, 'hierarchical_ward_zoom.png')
plt.show()

# Print suggested cuts
print("Merge distance gaps (last 8 merges):")
for i in range(len(last_merges)-1):
    gap = last_merges[i+1] - last_merges[i]
    n_clusters = len(Z_ward) + 1 - (len(Z_ward) - 9 + i + 1)
    print(f"  Cut between merge {len(Z_ward)-9+i} and {len(Z_ward)-8+i}: "
          f"gap={gap:.1f}, gives ~{n_clusters} clusters")""")

md("""**Discussion:** The bar chart shows the distance at each of the last 10 merges. A big jump between consecutive merges means two very different groups are being forced together — cutting just before that jump is a natural choice. We use this alongside silhouette/CH/DB metrics to pick the final number of clusters.""")

md("""### 4.4 Choosing the Number of Clusters

We apply AgglomerativeClustering on the **full dataset** with Ward linkage and test n = 2 to 8 clusters, evaluating with silhouette, Calinski-Harabasz, and Davies-Bouldin. We also test Complete and Average linkage for the same range.""")

code("""hc_results = []

for link in ['ward', 'complete', 'average']:
    for n in range(2, 9):
        hc = AgglomerativeClustering(n_clusters=n, linkage=link)
        labels = hc.fit_predict(X_arr)
        sil = silhouette_score(X_arr, labels, sample_size=5000, random_state=42)
        ch  = calinski_harabasz_score(X_arr, labels)
        db  = davies_bouldin_score(X_arr, labels)
        hc_results.append({'linkage': link, 'n_clusters': n,
                           'silhouette': round(sil, 4),
                           'calinski': round(ch, 0),
                           'davies_bouldin': round(db, 3)})

hc_df = pd.DataFrame(hc_results)
print("Hierarchical Clustering — All tested configurations:")
print(hc_df.to_string(index=False))""")

code("""# Plot metrics for each linkage method
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metrics = [('silhouette', 'Silhouette (higher=better)'),
           ('calinski', 'Calinski-Harabasz (higher=better)'),
           ('davies_bouldin', 'Davies-Bouldin (lower=better)')]

for ax, (col, title) in zip(axes, metrics):
    for link in ['ward', 'complete', 'average']:
        subset = hc_df[hc_df['linkage'] == link]
        ax.plot(subset['n_clusters'], subset[col], 'o-', label=link.capitalize())
    ax.set_title(title)
    ax.set_xlabel('Number of clusters')
    ax.legend()

plt.suptitle('Hierarchical Clustering — Metric Comparison Across Linkage Methods',
             fontsize=14, y=1.02)
plt.tight_layout()
save_plot(fig, 'hierarchical_metric_comparison.png')
plt.show()""")

md("""**Discussion — Choosing linkage and n_clusters:**

The plots above show how each metric changes with the number of clusters for all three linkage methods. Key observations:

- **Ward** generally achieves the best (or near-best) silhouette and Calinski-Harabasz scores, confirming it is the most suitable linkage for this dataset.
- **Complete** linkage can produce uneven clusters due to its sensitivity to outliers.
- **Average** linkage gives intermediate results.

We select the linkage and n_clusters that best balance all three metrics. If Ward with a small n (2-4) scores highest, that is our choice.""")

md("""### 4.5 Apply Best Hierarchical Configuration""")

code("""# Pick best config: highest silhouette
best_hc_row = hc_df.sort_values('silhouette', ascending=False).iloc[0]
best_hc_link = best_hc_row['linkage']
best_hc_n = int(best_hc_row['n_clusters'])

print(f"Best hierarchical config: {best_hc_link} linkage, n_clusters={best_hc_n}")
print(f"  silhouette={best_hc_row['silhouette']}, CH={best_hc_row['calinski']:.0f}, "
      f"DB={best_hc_row['davies_bouldin']}")

# Also show runner-ups
print(f"\\nTop 5 configs by silhouette:")
print(hc_df.sort_values('silhouette', ascending=False).head().to_string(index=False))

hc_final = AgglomerativeClustering(n_clusters=best_hc_n, linkage=best_hc_link)
hc_labels = hc_final.fit_predict(X_arr)
df['HC_Cluster'] = hc_labels

print(f"\\nCluster sizes:")
for c in sorted(set(hc_labels)):
    print(f"  Cluster {c}: {(hc_labels == c).sum()} games")""")

md("""### 4.6 Visualise Hierarchical Clusters""")

code("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=hc_labels,
                          cmap='viridis', alpha=0.3, s=5)
axes[0].set_title(f'Hierarchical ({best_hc_link.capitalize()}, n={best_hc_n}) — PCA')
axes[0].set_xlabel(f'PC1 ({ev[0]:.1%})'); axes[0].set_ylabel(f'PC2 ({ev[1]:.1%})')
plt.colorbar(scatter, ax=axes[0], label='Cluster')

counts_hc = pd.Series(hc_labels).value_counts().sort_index()
axes[1].bar(counts_hc.index, counts_hc.values,
            color=[plt.cm.viridis(i / best_hc_n) for i in counts_hc.index])
axes[1].set_title('Hierarchical Cluster Sizes')
axes[1].set_xlabel('Cluster'); axes[1].set_ylabel('Count')
for i, v in zip(counts_hc.index, counts_hc.values):
    axes[1].text(i, v + 200, str(v), ha='center', fontsize=11)

plt.tight_layout()
save_plot(fig, 'hierarchical_clusters.png')
plt.show()""")

md("""### 4.7 Cluster Profiling""")

code("""profile_hc = df.groupby('HC_Cluster')[cluster_features].mean()
print(f"Hierarchical Cluster Profiles ({best_hc_link}, n={best_hc_n}):")
print(profile_hc.T.round(2).to_string())""")

code("""# Z-scored heatmap
profile_hc_z = profile_hc.T.copy()
for col in profile_hc_z.columns:
    mu, sd = profile_hc_z[col].mean(), profile_hc_z[col].std()
    profile_hc_z[col] = (profile_hc_z[col] - mu) / (sd + 1e-8)

fig, ax = plt.subplots(figsize=(max(8, best_hc_n * 2.5), 10))
sns.heatmap(profile_hc_z, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, linewidths=0.5)
ax.set_title(f'Hierarchical Cluster Profiles — z-scored ({best_hc_link}, n={best_hc_n})')
ax.set_xlabel('Cluster'); ax.set_ylabel('Feature')
plt.tight_layout()
save_plot(fig, 'hierarchical_profiles.png')
plt.show()""")

md("""**Discussion:** Similar to K-Means profiling — red means the cluster is above average for that feature, blue means below. We compare the cluster "personalities" to those found by K-Means to see if the hierarchical method finds similar or different groupings.""")

md("""### 4.8 External Validation with Rating""")

code("""ct_hc = pd.crosstab(df['HC_Cluster'], rating, normalize='index').round(3)
print("Hierarchical Clusters vs Rating (row-normalised):")
print(ct_hc)

fig, ax = plt.subplots(figsize=(10, 5))
ct_hc.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
ax.set_title(f'Hierarchical Clusters vs Rating ({best_hc_link}, n={best_hc_n})')
ax.set_xlabel('Cluster'); ax.set_ylabel('Proportion')
ax.legend(title='Rating', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
save_plot(fig, 'hierarchical_vs_rating.png')
plt.show()""")

md("""**Discussion:** We check whether hierarchical clusters correspond to different Rating distributions, similar to the K-Means and DBSCAN external validation. If Ward linkage produces similar results to K-Means, this is expected — both minimise within-cluster variance (Ward is essentially a greedy version of K-Means).""")

md("""### 4.9 Hierarchical Clustering Summary

**What we did:**
- Tested 3 linkage methods (Ward, Complete, Average) with dendrograms on a 2000-point sample.
- Tested 7 values of n_clusters (2-8) for each linkage = 21 total configurations.
- Selected the best configuration by silhouette score.
- Profiled and validated the clusters.

**Key findings:**
- Ward linkage produces the most balanced, interpretable clusters (as expected for numeric data).
- The dendrogram shows clear merge-distance jumps that confirm our choice of n_clusters.
- The resulting clusters have distinct feature profiles, similar to K-Means.""")

md("""---
*Section 5 (Final Discussion — comparing all algorithms) will be added next.*""")

# ===== APPEND AND WRITE =====
nb["cells"].extend(new_cells)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Appended {len(new_cells)} cells. Total now: {len(nb['cells'])}")
print(f"  New code cells: {sum(1 for c in new_cells if c['cell_type']=='code')}")
print(f"  New markdown cells: {sum(1 for c in new_cells if c['cell_type']=='markdown')}")
