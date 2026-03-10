"""Add Task 2.4 (Final Clustering Comparison) cells to 02_clustering.ipynb"""
import json

nb = json.load(open('notebooks/02_clustering.ipynb', 'r', encoding='utf-8'))

# Remove placeholder cell 67
nb['cells'] = nb['cells'][:67]

def md(src):
    lines = src.split('\n')
    source = [line + '\n' for line in lines]
    source[-1] = source[-1].rstrip('\n')
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(src):
    lines = src.split('\n')
    source = [line + '\n' for line in lines]
    source[-1] = source[-1].rstrip('\n')
    return {"cell_type": "code", "metadata": {}, "source": source,
            "execution_count": None, "outputs": []}

new_cells = []

# Section header
new_cells.append(md(
"## 5. Final Discussion - Comparing All Clustering Algorithms\n"
"\n"
"In this section we compare K-Means, DBSCAN, and Hierarchical clustering side by side. "
"We look at internal quality metrics, cluster balance, and how well each algorithm separates "
"the Rating target variable."
))

# Comparison table code
new_cells.append(code(
"# Build comparison summary table\n"
"comparison = []\n"
"\n"
"# --- K-Means (best: k=5) ---\n"
"km_labels = df['KMeans_Cluster'].values\n"
"km_sil = silhouette_score(X_arr, km_labels, sample_size=5000, random_state=42)\n"
"km_ch  = calinski_harabasz_score(X_arr, km_labels)\n"
"km_db  = davies_bouldin_score(X_arr, km_labels)\n"
"km_sizes = np.bincount(km_labels)\n"
"km_balance = km_sizes.min() / km_sizes.max()\n"
"\n"
"comparison.append({\n"
"    'Algorithm': 'K-Means (k=5)',\n"
"    'Silhouette': round(km_sil, 4),\n"
"    'Calinski-Harabasz': round(km_ch, 0),\n"
"    'Davies-Bouldin': round(km_db, 3),\n"
"    'Num Clusters': len(set(km_labels)),\n"
"    'Noise Points': 0,\n"
"    'Min/Max Balance': round(km_balance, 3),\n"
"})\n"
"\n"
"# --- DBSCAN (best: eps=3.13, ms=5) ---\n"
"db_labels = df['DBSCAN_Cluster'].values\n"
"db_mask = db_labels != -1\n"
"if db_mask.sum() > 0 and len(set(db_labels[db_mask])) >= 2:\n"
"    db_sil = silhouette_score(X_arr[db_mask], db_labels[db_mask],\n"
"                               sample_size=min(5000, db_mask.sum()), random_state=42)\n"
"else:\n"
"    db_sil = float('nan')\n"
"db_non_noise = db_labels[db_mask]\n"
"if len(db_non_noise) > 0:\n"
"    db_ch = calinski_harabasz_score(X_arr[db_mask], db_non_noise)\n"
"    db_db_score = davies_bouldin_score(X_arr[db_mask], db_non_noise)\n"
"    db_sizes = np.bincount(db_non_noise)\n"
"    db_balance = db_sizes.min() / db_sizes.max()\n"
"else:\n"
"    db_ch = db_db_score = float('nan')\n"
"    db_balance = 0\n"
"\n"
"comparison.append({\n"
"    'Algorithm': 'DBSCAN (eps=3.13, ms=5)',\n"
"    'Silhouette': round(db_sil, 4),\n"
"    'Calinski-Harabasz': round(db_ch, 0),\n"
"    'Davies-Bouldin': round(db_db_score, 3),\n"
"    'Num Clusters': len(set(db_labels[db_mask])),\n"
"    'Noise Points': int((~db_mask).sum()),\n"
"    'Min/Max Balance': round(db_balance, 3),\n"
"})\n"
"\n"
"# --- Hierarchical (best: ward, n=3) ---\n"
"hc_labels_final = df['HC_Cluster'].values\n"
"hc_sil = silhouette_score(X_arr, hc_labels_final, sample_size=5000, random_state=42)\n"
"hc_ch  = calinski_harabasz_score(X_arr, hc_labels_final)\n"
"hc_db  = davies_bouldin_score(X_arr, hc_labels_final)\n"
"hc_sizes = np.bincount(hc_labels_final)\n"
"hc_balance = hc_sizes.min() / hc_sizes.max()\n"
"\n"
"comparison.append({\n"
"    'Algorithm': 'Hierarchical (Ward, n=3)',\n"
"    'Silhouette': round(hc_sil, 4),\n"
"    'Calinski-Harabasz': round(hc_ch, 0),\n"
"    'Davies-Bouldin': round(hc_db, 3),\n"
"    'Num Clusters': len(set(hc_labels_final)),\n"
"    'Noise Points': 0,\n"
"    'Min/Max Balance': round(hc_balance, 3),\n"
"})\n"
"\n"
"comp_df = pd.DataFrame(comparison)\n"
'print("=== Clustering Algorithm Comparison ===")\n'
"print(comp_df.to_string(index=False))"
))

# Discussion markdown
new_cells.append(md(
"**Discussion - Which algorithm is best for this dataset?**\n"
"\n"
"| Criterion | K-Means | DBSCAN | Hierarchical |\n"
"|-----------|---------|--------|-------------|\n"
"| Cluster balance | Good (5 roughly even groups) | Moderate (few large clusters + noise) | Poor (1 dominant cluster) |\n"
"| Silhouette | Low (~0.18) | High (~0.66, non-noise only) | Moderate (~0.36) |\n"
"| Noise handling | No | Yes (identifies outliers) | No |\n"
"| Shape assumption | Spherical | Arbitrary | Hierarchical merges |\n"
"| Rating separation | Partial | Strong for noise vs. core | Strong (small clusters = high-quality games) |\n"
"\n"
"**Key findings:**\n"
"\n"
"1. **K-Means** produces the most balanced clusters (5 groups of roughly equal size). However, the silhouette "
"score is low (0.18), meaning the clusters overlap significantly in feature space. This is expected since the "
"data does not have well-separated spherical groups.\n"
"\n"
"2. **DBSCAN** has the highest silhouette score on non-noise points, but this is partly because it classifies "
"ambiguous points as noise. It is useful for identifying outlier games, but the core clusters are dominated by "
"one large group.\n"
"\n"
"3. **Hierarchical (Ward)** creates an interesting split: one massive mainstream cluster and two smaller clusters "
"of high-quality or niche games. The silhouette (0.36) sits between K-Means and DBSCAN.\n"
"\n"
"**Best algorithm:** There is no single \"best\" algorithm. Each gives a different perspective:\n"
"- Use **K-Means** if you want to segment games into roughly equal market groups (e.g., for recommendation).\n"
"- Use **DBSCAN** if you want to identify outlier games that do not fit any group.\n"
"- Use **Hierarchical** if you want a coarse split between mainstream and premium/niche games.\n"
"\n"
"For predicting the Rating variable, **K-Means with k=5** gives the most actionable segmentation because it "
"separates games into groups with different rating distributions, while keeping clusters balanced enough to be useful."
))

# Side-by-side PCA plot
new_cells.append(code(
"# Side-by-side PCA visualisation of all 3 algorithms\n"
"fig, axes = plt.subplots(1, 3, figsize=(20, 6))\n"
"\n"
"for ax, (col, title) in zip(axes, [\n"
"    ('KMeans_Cluster', 'K-Means (k=5)'),\n"
"    ('DBSCAN_Cluster', 'DBSCAN (eps=3.13, ms=5)'),\n"
"    ('HC_Cluster', 'Hierarchical (Ward, n=3)')\n"
"]):\n"
"    labels_plot = df[col].values\n"
"    if col == 'DBSCAN_Cluster':\n"
"        noise = labels_plot == -1\n"
"        ax.scatter(X_pca[noise, 0], X_pca[noise, 1], c='lightgrey', s=3, alpha=0.3, label='Noise')\n"
"        scatter = ax.scatter(X_pca[~noise, 0], X_pca[~noise, 1],\n"
"                             c=labels_plot[~noise], cmap='tab10', s=5, alpha=0.5)\n"
"    else:\n"
"        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],\n"
"                             c=labels_plot, cmap='tab10', s=5, alpha=0.5)\n"
"    ax.set_title(title, fontsize=13)\n"
"    ax.set_xlabel('PC1')\n"
"    ax.set_ylabel('PC2')\n"
"\n"
"plt.suptitle('Clustering Comparison - PCA Projection', fontsize=15, y=1.02)\n"
"plt.tight_layout()\n"
"save_plot(fig, 'clustering_comparison.png')\n"
"plt.show()"
))

new_cells.append(md(
"**Discussion:** The PCA projections above show how each algorithm partitions the same 2D space. "
"K-Means creates regular regions, DBSCAN separates dense cores from noise, and Hierarchical clustering "
"splits a small premium group from the mainstream."
))

# External validation: ARI and NMI
new_cells.append(code(
"# Adjusted Rand Index (ARI) and Normalised Mutual Information (NMI) vs Rating\n"
"from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score\n"
"\n"
"rating_encoded = df['Rating'].map({'Low': 0, 'Medium': 1, 'High': 2}).values\n"
"\n"
"ext_results = []\n"
"for col, name in [('KMeans_Cluster', 'K-Means'),\n"
"                   ('DBSCAN_Cluster', 'DBSCAN'),\n"
"                   ('HC_Cluster', 'Hierarchical')]:\n"
"    labels_ext = df[col].values\n"
"    if col == 'DBSCAN_Cluster':\n"
"        mask = labels_ext != -1\n"
"        ari = adjusted_rand_score(rating_encoded[mask], labels_ext[mask])\n"
"        nmi = normalized_mutual_info_score(rating_encoded[mask], labels_ext[mask])\n"
"    else:\n"
"        ari = adjusted_rand_score(rating_encoded, labels_ext)\n"
"        nmi = normalized_mutual_info_score(rating_encoded, labels_ext)\n"
"    ext_results.append({'Algorithm': name, 'ARI': round(ari, 4), 'NMI': round(nmi, 4)})\n"
"\n"
"ext_df = pd.DataFrame(ext_results)\n"
'print("External Validation vs Rating:")\n'
"print(ext_df.to_string(index=False))"
))

new_cells.append(md(
"**Discussion:** Adjusted Rand Index (ARI) and Normalised Mutual Information (NMI) measure how well "
"each clustering aligns with the true Rating labels. Higher values mean the clusters correspond more closely "
"to the Low/Medium/High rating categories. This helps us understand which algorithm best captures the structure "
"that defines game quality."
))

# End of notebook
new_cells.append(md(
"---\n"
"\n"
"## Summary\n"
"\n"
"This notebook implemented three clustering algorithms on the board game dataset:\n"
"\n"
"1. **K-Means** (k=5): Balanced segmentation with modest separation.\n"
"2. **DBSCAN** (eps=3.13, ms=5): Density-based clustering that identifies outliers.\n"
"3. **Hierarchical (Ward, n=3)**: Coarse split revealing premium vs mainstream games.\n"
"\n"
"All algorithms were evaluated with internal metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin) "
"and external validation against the Rating variable. The final comparison shows each algorithm offers "
"different insights into the dataset's structure."
))

for cell in new_cells:
    nb['cells'].append(cell)

json.dump(nb, open('notebooks/02_clustering.ipynb', 'w', encoding='utf-8'), indent=1)
print(f'Added {len(new_cells)} cells. Total cells: {len(nb["cells"])}')
