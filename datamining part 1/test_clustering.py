"""Quick test of clustering code — run as script to check timing."""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.join('src'))
os.environ['MPLBACKEND'] = 'Agg'
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             calinski_harabasz_score, davies_bouldin_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

t0 = time.time()

# 1. Load
df = pd.read_csv('dataset/processed/DM1_game_dataset_clean.csv')
rating = df['Rating'].copy()
cluster_features = [
    'YearPublished', 'GameWeight', 'MinPlayers', 'MaxPlayers',
    'ComAgeRec', 'LanguageEase', 'BestPlayers',
    'NumOwned', 'NumWish', 'NumWeightVotes',
    'MfgPlaytime', 'MfgAgeRec',
    'NumAlternates', 'NumExpansions', 'NumImplementations',
]
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(df[cluster_features]), columns=cluster_features)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
print(f"[{time.time()-t0:.1f}s] Loaded: {X.shape}")

# 2. K-Means
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    print(f"  k={k}: sil={sil:.4f}")
print(f"[{time.time()-t0:.1f}s] K-Means sweep done")

km3 = KMeans(n_clusters=3, n_init=20, random_state=42, max_iter=500)
kl = km3.fit_predict(X)
print(f"[{time.time()-t0:.1f}s] K-Means k=3 done")

# 3. DBSCAN k-dist (NO n_jobs)
np.random.seed(42)
si = np.random.choice(len(X), size=5000, replace=False)
Xs = X.values[si]
nn = NearestNeighbors(n_neighbors=15)
nn.fit(Xs)
distances, _ = nn.kneighbors(Xs)
k_dist = np.sort(distances[:, -1])
print(f"[{time.time()-t0:.1f}s] k-NN done")

eps_cands = sorted(set([round(np.percentile(k_dist, p), 2) for p in [90, 95, 97, 99]]))
ms_cands = [5, 10, 15, 20]
print(f"eps: {eps_cands}")

for eps_val in eps_cands:
    for ms in ms_cands:
        db = DBSCAN(eps=eps_val, min_samples=ms)
        lbl = db.fit_predict(X)
        n_cl = len(set(lbl)) - (1 if -1 in lbl else 0)
        n_noise = (lbl == -1).sum()
        sil = -1.0
        if n_cl >= 2:
            mask = lbl != -1
            if mask.sum() > n_cl:
                sil = silhouette_score(X.values[mask], lbl[mask])
        print(f"  eps={eps_val} ms={ms}: {n_cl}cl {n_noise}noise sil={sil:.4f}")

print(f"[{time.time()-t0:.1f}s] DONE")
