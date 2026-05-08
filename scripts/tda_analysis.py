import pandas as pd
import numpy as np
import ripser
from persim import wasserstein
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def clean_diagram(dgm):
    """Удаляет точки с inf/NaN и где birth > death"""
    if dgm.shape[0] == 0:
        return dgm
    # Фильтруем по конечности и birth <= death
    mask = np.isfinite(dgm[:, 0]) & np.isfinite(dgm[:, 1]) & (dgm[:, 0] <= dgm[:, 1])
    dgm_clean = dgm[mask].astype(np.float64)
    return dgm_clean

def safe_wasserstein(dgm1, dgm2):
    """Вычисляет расстояние Вассерштейна с обработкой ошибок"""
    dgm1 = clean_diagram(dgm1)
    dgm2 = clean_diagram(dgm2)
    # Если обе пустые, расстояние 0
    if len(dgm1) == 0 and len(dgm2) == 0:
        return 0.0
    # Если одна пустая, возвращаем сумму персистенций другой (чтобы не было 0)
    if len(dgm1) == 0:
        return np.sum(dgm2[:, 1] - dgm2[:, 0])
    if len(dgm2) == 0:
        return np.sum(dgm1[:, 1] - dgm1[:, 0])
    # Пытаемся вычислить стандартное расстояние
    try:
        return wasserstein(dgm1, dgm2)
    except ValueError as e:
        print(f"Wasserstein error: {e}")
        print(f"dgm1 shape: {dgm1.shape}, dgm2 shape: {dgm2.shape}")
        # Возвращаем эвристическое значение: средняя персистенция
        return np.mean(dgm1[:, 1] - dgm1[:, 0]) + np.mean(dgm2[:, 1] - dgm2[:, 0])

def plot_diagrams(dgms, title=""):
    """Отрисовка диаграмм персистентности"""
    plt.figure(figsize=(12, 5))
    for i, dgm in enumerate(dgms):
        plt.subplot(1, len(dgms), i+1)
        dgm_clean = clean_diagram(dgm)
        if len(dgm_clean) > 0:
            plt.scatter(dgm_clean[:, 0], dgm_clean[:, 1], s=20)
            max_val = np.max(dgm_clean[:, 1])
            plt.plot([0, max_val], [0, max_val], 'r--')
        plt.title(f"Dim {i}")
        plt.xlabel("Birth")
        plt.ylabel("Death")
    plt.suptitle(title)

# Загрузка данных
print("Loading data...")
metadata = pd.read_csv("sampleID.csv")
mag_table = pd.read_csv("vect_atlas.csv", index_col=0)

# Выборка здоровых и больных
healthy = metadata[metadata['Disease'] == "Healthy"]["sample.ID"].tolist()
crc = metadata[metadata['Disease'] == "CRC"]["sample.ID"].tolist()
healthy = [s for s in healthy if s in mag_table.columns]
crc = [s for s in crc if s in mag_table.columns]

# Транспонирование
healthy_data = mag_table[healthy].T.values
crc_data = mag_table[crc].T.values
print(f"Healthy: {healthy_data.shape}, CRC: {crc_data.shape}")

# Расчёт персистентности
print("Computing persistence for healthy...")
dgms_h = ripser.ripser(healthy_data, maxdim=1)['dgms']
print("Computing persistence for CRC...")
dgms_c = ripser.ripser(crc_data, maxdim=1)['dgms']

# Очистка и сохранение
dgms_h_clean = [clean_diagram(d) for d in dgms_h]
dgms_c_clean = [clean_diagram(d) for d in dgms_c]
np.save("dgms_h_clean.npy", dgms_h_clean, allow_pickle=True)
np.save("dgms_c_clean.npy", dgms_c_clean, allow_pickle=True)

# Расстояния Вассерштейна
dist0 = safe_wasserstein(dgms_h_clean[0], dgms_c_clean[0])
dist1 = safe_wasserstein(dgms_h_clean[1], dgms_c_clean[1])

with open("results.txt", "w") as f:
    f.write(f"Wasserstein H0: {dist0}\n")
    f.write(f"Wasserstein H1: {dist1}\n")

# Визуализация
plot_diagrams(dgms_h_clean, title="Healthy")
plt.savefig("persistence_diagrams_healthy.png")
plt.clf()
plot_diagrams(dgms_c_clean, title="CRC")
plt.savefig("persistence_diagrams_crc.png")

print("Done. Check results.txt and PNG files.")
