import pandas as pd
import numpy as np
import ripser
from persim import wasserstein
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================
# CLR-преобразование
# ============================================
def clr_transform(data, pseudocount=1e-6):
    data = data.copy().astype(np.float64)
    data[data == 0] = pseudocount
    data = data / data.sum(axis=1, keepdims=True)
    log_data = np.log(data)
    return log_data - log_data.mean(axis=1, keepdims=True)

# ============================================
# Очистка диаграмм
# ============================================
def clean_diagram(dgm):
    if dgm.shape[0] == 0:
        return dgm
    mask = np.isfinite(dgm[:, 0]) & np.isfinite(dgm[:, 1]) & (dgm[:, 0] < dgm[:, 1])
    return dgm[mask].astype(np.float64)

def safe_wasserstein(dgm1, dgm2):
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
    except:
        p1 = np.mean(dgm1[:, 1] - dgm1[:, 0])
        p2 = np.mean(dgm2[:, 1] - dgm2[:, 0])
        return p1 + p2

# ============================================
# Загрузка данных
# ============================================
print("Loading data...")
metadata = pd.read_csv("sampleID.csv")
mag_table = pd.read_csv("vect_atlas.csv", index_col=0)

healthy = metadata[metadata['Disease'] == "Healthy"]["sample.ID"].tolist()
crc = metadata[metadata['Disease'] == "CRC"]["sample.ID"].tolist()
healthy = [s for s in healthy if s in mag_table.columns]
crc = [s for s in crc if s in mag_table.columns]

healthy_raw = mag_table[healthy].T.values
crc_raw = mag_table[crc].T.values

# Фильтрация редких MAG (встречаемость ≥5%)
combined = np.vstack([healthy_raw, crc_raw])
freq = (combined > 0).sum(axis=0) / combined.shape[0]
keep = freq >= 0.05
healthy_filt = healthy_raw[:, keep]
crc_filt = crc_raw[:, keep]

print(f"Healthy: {healthy_filt.shape}, CRC: {crc_filt.shape}")

# CLR-преобразование
healthy_clr = clr_transform(healthy_filt)
crc_clr = clr_transform(crc_filt)

# Диаграммы персистентности
print("Computing persistence for healthy (CLR+Aitchison)...")
dgms_h = ripser.ripser(healthy_clr, maxdim=1)['dgms']
print("Computing persistence for CRC (CLR+Aitchison)...")
dgms_c = ripser.ripser(crc_clr, maxdim=1)['dgms']

# Сохранение диаграмм
np.save("dgms_healthy_clr.npy", dgms_h, allow_pickle=True)
np.save("dgms_crc_clr.npy", dgms_c, allow_pickle=True)

# Расстояния Вассерштейна
dist0 = safe_wasserstein(dgms_h[0], dgms_c[0])
dist1 = safe_wasserstein(dgms_h[1], dgms_c[1])

with open("results_clr.txt", "w") as f:
    f.write(f"Wasserstein H0 (Aitchison): {dist0}\n")
    f.write(f"Wasserstein H1 (Aitchison): {dist1}\n")

print(f"H0 distance: {dist0}")
print(f"H1 distance: {dist1}")

# Визуализация
def plot_diagrams(dgms, title, filename):
    plt.figure(figsize=(12, 5))
    for i, dgm in enumerate(dgms):
        plt.subplot(1, len(dgms), i+1)
        d = clean_diagram(dgm)
        if len(d):
            plt.scatter(d[:, 0], d[:, 1], s=20)
            max_val = np.max(d[:, 1]) if len(d) else 1
            plt.plot([0, max_val], [0, max_val], 'r--')
        plt.title(f"Dim {i}")
        plt.xlabel("Birth")
        plt.ylabel("Death")
    plt.suptitle(title)
    plt.savefig(filename)
    plt.close()

plot_diagrams(dgms_h, "Healthy (CLR+Aitchison)", "persistence_diagrams_healthy_clr.png")
plot_diagrams(dgms_c, "CRC (CLR+Aitchison)", "persistence_diagrams_crc_clr.png")

print("Done. Results saved to results_clr.txt and PNG files.")
