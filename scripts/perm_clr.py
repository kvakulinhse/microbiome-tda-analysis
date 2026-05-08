#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Перестановочный тест для CLR + Aitchison (с балансировкой групп)
"""

import numpy as np
import pandas as pd
import ripser
from persim import wasserstein
import sys
import time

# ============================================
# Функции
# ============================================
def clr_transform(data, pseudocount=1e-6):
    """CLR (Centered Log-Ratio) transformation"""
    data = data.copy().astype(np.float64)
    data[data == 0] = pseudocount
    data = data / data.sum(axis=1, keepdims=True)
    log_data = np.log(data)
    return log_data - log_data.mean(axis=1, keepdims=True)

def clean_diagram(dgm):
    """Удалить точки с inf/NaN и где birth >= death"""
    if dgm.shape[0] == 0:
        return dgm
    mask = np.isfinite(dgm[:, 0]) & np.isfinite(dgm[:, 1]) & (dgm[:, 0] < dgm[:, 1])
    return dgm[mask].astype(np.float64)

def wasserstein_safe(dgm1, dgm2):
    """Безопасное вычисление расстояния Вассерштейна"""
    dgm1 = clean_diagram(dgm1)
    dgm2 = clean_diagram(dgm2)
    if len(dgm1) == 0 and len(dgm2) == 0:
        return 0.0
    if len(dgm1) == 0:
        return np.mean(dgm2[:, 1] - dgm2[:, 0]) if len(dgm2) else 0.0
    if len(dgm2) == 0:
        return np.mean(dgm1[:, 1] - dgm1[:, 0]) if len(dgm1) else 0.0
    try:
        return wasserstein(dgm1, dgm2)
    except Exception:
        p1 = np.mean(dgm1[:, 1] - dgm1[:, 0])
        p2 = np.mean(dgm2[:, 1] - dgm2[:, 0])
        return p1 + p2

def compute_diagram(data, maxdim=1):
    """Вычисляет диаграмму персистентности"""
    result = ripser.ripser(data, maxdim=maxdim)
    return result['dgms']

# ============================================
# Загрузка и подготовка данных
# ============================================
print("=" * 60)
print("ПЕРЕСТАНОВОЧНЫЙ ТЕСТ (CLR + Aitchison)")
print("=" * 60)

print("\n1. Загрузка данных...")
metadata = pd.read_csv("sampleID.csv")
mag_table = pd.read_csv("vect_atlas.csv", index_col=0)

healthy_labels = metadata[metadata['Disease'] == "Healthy"]["sample.ID"].tolist()
crc_labels = metadata[metadata['Disease'] == "CRC"]["sample.ID"].tolist()

healthy_cols = [c for c in healthy_labels if c in mag_table.columns]
crc_cols = [c for c in crc_labels if c in mag_table.columns]

print(f"   Здоровых (исходно): {len(healthy_cols)}")
print(f"   CRC (исходно): {len(crc_cols)}")

# Извлекаем данные (строки = MAG, столбцы = образцы)
healthy_raw = mag_table[healthy_cols].values
crc_raw = mag_table[crc_cols].values

# Фильтрация MAG (встречаемость ≥5% в объединённой группе)
combined = np.hstack([healthy_raw, crc_raw])
freq = (combined > 0).sum(axis=1) / combined.shape[1]
keep = freq >= 0.05
healthy_filt = healthy_raw[keep, :]
crc_filt = crc_raw[keep, :]
print(f"   MAG после фильтрации: {keep.sum()}")

# Транспонируем: строки = образцы, столбцы = MAG
healthy_data = healthy_filt.T  # (n_healthy, n_mag)
crc_data = crc_filt.T          # (n_crc, n_mag)

# CLR-преобразование
print("2. CLR-преобразование...")
healthy_clr = clr_transform(healthy_data)
crc_clr = clr_transform(crc_data)

# Балансировка: будем использовать одинаковое число образцов из каждой группы
n_balanced = min(healthy_clr.shape[0], crc_clr.shape[0])
print(f"3. Балансировка групп до {n_balanced} образцов в каждой")

# Случайная подвыборка здоровых до размера CRC (фиксируем seed для воспроизводимости)
np.random.seed(42)
healthy_indices = np.random.choice(healthy_clr.shape[0], n_balanced, replace=False)
healthy_balanced = healthy_clr[healthy_indices, :]
crc_all = crc_clr  # все CRC (их и так 662)

print(f"   Здоровых для теста: {healthy_balanced.shape[0]}")
print(f"   CRC для теста: {crc_all.shape[0]}")

# Реальное расстояние (между сбалансированной подвыборкой здоровых и всеми CRC)
print("4. Вычисление реального расстояния...")
dgms_h_real = compute_diagram(healthy_balanced)
dgms_c_real = compute_diagram(crc_all)
dist_real_h0 = wasserstein_safe(dgms_h_real[0], dgms_c_real[0])
dist_real_h1 = wasserstein_safe(dgms_h_real[1], dgms_c_real[1])
print(f"   Реальное расстояние H0: {dist_real_h0:.6f}")
print(f"   Реальное расстояние H1: {dist_real_h1:.6f}")

# ============================================
# Перестановочный тест
# ============================================
n_permutations = 200
print(f"\n5. Перестановочный тест ({n_permutations} итераций)...")

# Объединяем данные
all_data = np.vstack([healthy_balanced, crc_all])
n_healthy_bal = healthy_balanced.shape[0]
n_crc = crc_all.shape[0]
labels = np.array([0]*n_healthy_bal + [1]*n_crc)

distances_h0 = []
distances_h1 = []

start_time = time.time()

for perm in range(n_permutations):
    # Перемешиваем метки
    perm_labels = labels.copy()
    np.random.shuffle(perm_labels)
    
    # Формируем группы (сбалансированные!)
    group1 = all_data[perm_labels == 0]
    group2 = all_data[perm_labels == 1]
    
    # Если группы сильно разного размера — балансируем (хотя при больших числах это редко)
    min_size = min(len(group1), len(group2))
    if len(group1) > min_size:
        idx = np.random.choice(len(group1), min_size, replace=False)
        group1 = group1[idx]
    if len(group2) > min_size:
        idx = np.random.choice(len(group2), min_size, replace=False)
        group2 = group2[idx]
    
    # Вычисляем диаграммы
    dgms1 = compute_diagram(group1)
    dgms2 = compute_diagram(group2)
    
    # Расстояния
    dist0 = wasserstein_safe(dgms1[0], dgms2[0])
    dist1 = wasserstein_safe(dgms1[1], dgms2[1])
    
    distances_h0.append(dist0)
    distances_h1.append(dist1)
    
    if (perm + 1) % 50 == 0:
        elapsed = time.time() - start_time
        print(f"   {perm+1}/{n_permutations} итераций (за {elapsed:.1f} сек)")

# ============================================
# Результаты
# ============================================
print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ")
print("=" * 60)

# H0
p_value_h0 = np.mean(np.array(distances_h0) >= dist_real_h0)
print(f"\nH0 (компоненты связности):")
print(f"   Реальное расстояние: {dist_real_h0:.6f}")
print(f"   Среднее перестановочное: {np.mean(distances_h0):.6f} ± {np.std(distances_h0):.6f}")
print(f"   p-value: {p_value_h0:.6f}")

# H1
p_value_h1 = np.mean(np.array(distances_h1) >= dist_real_h1)
print(f"\nH1 (петли):")
print(f"   Реальное расстояние: {dist_real_h1:.6f}")
print(f"   Среднее перестановочное: {np.mean(distances_h1):.6f} ± {np.std(distances_h1):.6f}")
print(f"   p-value: {p_value_h1:.6f}")

# Сохраняем результаты
with open("perm_results_clr.txt", "w") as f:
    f.write("ПЕРЕСТАНОВОЧНЫЙ ТЕСТ (CLR + Aitchison)\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Число перестановок: {n_permutations}\n")
    f.write(f"Размер групп (балансированный): {n_balanced}\n\n")
    f.write(f"H0 (компоненты связности):\n")
    f.write(f"  Real distance: {dist_real_h0:.6f}\n")
    f.write(f"  Mean perm: {np.mean(distances_h0):.6f}\n")
    f.write(f"  Std perm: {np.std(distances_h0):.6f}\n")
    f.write(f"  p-value: {p_value_h0:.6f}\n\n")
    f.write(f"H1 (петли):\n")
    f.write(f"  Real distance: {dist_real_h1:.6f}\n")
    f.write(f"  Mean perm: {np.mean(distances_h1):.6f}\n")
    f.write(f"  Std perm: {np.std(distances_h1):.6f}\n")
    f.write(f"  p-value: {p_value_h1:.6f}\n")

print("\nРезультаты сохранены в perm_results_clr.txt")
print("Готово!")
