#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TDA analysis with Bray-Curtis distance."""

import numpy as np
import pandas as pd
import ripser
from sklearn.metrics import pairwise_distances
from persim import wasserstein
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def clean_diagram(dgm):
    if dgm.shape[0] == 0:
        return dgm
    mask = np.isfinite(dgm[:,0]) & np.isfinite(dgm[:,1]) & (dgm[:,0] < dgm[:,1])
    return dgm[mask].astype(np.float64)

def safe_wasserstein(dgm1, dgm2):
    dgm1 = clean_diagram(dgm1)
    dgm2 = clean_diagram(dgm2)
    if len(dgm1) == 0 and len(dgm2) == 0:
        return 0.0
    if len(dgm1) == 0:
        return np.mean(dgm2[:,1] - dgm2[:,0]) if len(dgm2) else 0.0
    if len(dgm2) == 0:
        return np.mean(dgm1[:,1] - dgm1[:,0]) if len(dgm1) else 0.0
    try:
        return wasserstein(dgm1, dgm2)
    except Exception:
        p1 = np.mean(dgm1[:,1] - dgm1[:,0])
        p2 = np.mean(dgm2[:,1] - dgm2[:,0])
        return p1 + p2

def plot_diagrams(dgms, title, filename):
    plt.figure(figsize=(12,5))
    for i, dgm in enumerate(dgms):
        plt.subplot(1, len(dgms), i+1)
        d = clean_diagram(dgm)
        if len(d):
            plt.scatter(d[:,0], d[:,1], s=20)
            max_val = np.max(d[:,1]) if len(d) else 1
            plt.plot([0, max_val], [0, max_val], 'r--')
        plt.title(f"Dim {i}")
        plt.xlabel("Birth")
        plt.ylabel("Death")
    plt.suptitle(title)
    plt.savefig(filename)
    plt.close()

# Загрузка данных
print("Loading data...")
metadata = pd.read_csv("sampleID.csv")
mag_table = pd.read_csv("vect_atlas.csv", index_col=0)

healthy = metadata[metadata['Disease'] == "Healthy"]["sample.ID"].tolist()
crc = metadata[metadata['Disease'] == "CRC"]["sample.ID"].tolist()
healthy = [s for s in healthy if s in mag_table.columns]
crc = [s for s in crc if s in mag_table.columns]

healthy_raw = mag_table[healthy].T.values
crc_raw = mag_table[crc].T.values

# Фильтрация MAG (встречаемость ≥5%)
combined = np.vstack([healthy_raw, crc_raw])
freq = (combined > 0).sum(axis=0) / combined.shape[0]
keep = freq >= 0.05
healthy_filt = healthy_raw[:, keep]
crc_filt = crc_raw[:, keep]

# Нормализация к сумме 1 для Bray-Curtis
def normalize_to_sum1(data):
    row_sums = data.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return data / row_sums

healthy_norm = normalize_to_sum1(healthy_filt)
crc_norm = normalize_to_sum1(crc_filt)

print(f"Healthy: {healthy_norm.shape}, CRC: {crc_norm.shape}")

# Вычисление матриц расстояний Брея-Кёртиса
print("Computing Bray-Curtis distance matrices...")
D_healthy = pairwise_distances(healthy_norm, metric='braycurtis')
D_crc = pairwise_distances(crc_norm, metric='braycurtis')

# Персистентная гомология на матрицах расстояний
print("Computing persistence for healthy...")
dgms_h = ripser.ripser(D_healthy, maxdim=1, distance_matrix=True)['dgms']
print("Computing persistence for CRC...")
dgms_c = ripser.ripser(D_crc, maxdim=1, distance_matrix=True)['dgms']

dist0 = safe_wasserstein(dgms_h[0], dgms_c[0])
dist1 = safe_wasserstein(dgms_h[1], dgms_c[1])

with open("results_braycurtis.txt", "w") as f:
    f.write(f"Wasserstein H0 (Bray-Curtis): {dist0}\n")
    f.write(f"Wasserstein H1 (Bray-Curtis): {dist1}\n")

print(f"H0 distance: {dist0}")
print(f"H1 distance: {dist1}")

plot_diagrams(dgms_h, "Healthy (Bray-Curtis)", "persistence_diagrams_healthy_braycurtis.png")
plot_diagrams(dgms_c, "CRC (Bray-Curtis)", "persistence_diagrams_crc_braycurtis.png")

print("Done. Results saved to results_braycurtis.txt")
