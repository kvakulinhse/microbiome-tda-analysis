import numpy as np
import ripser
from persim import wasserstein

def avg_persistence(dgm):
    """Средняя персистенция (сумма death-birth) / количество точек"""
    if len(dgm) == 0:
        return 0.0
    return np.mean(dgm[:,1] - dgm[:,0])

# Загрузка данных
import pandas as pd
metadata = pd.read_csv("sampleID.csv")
mag_table = pd.read_csv("vect_atlas.csv", index_col=0)

healthy = metadata[metadata['Disease'] == "Healthy"]["sample.ID"].tolist()
crc = metadata[metadata['Disease'] == "CRC"]["sample.ID"].tolist()
healthy = [s for s in healthy if s in mag_table.columns]
crc = [s for s in crc if s in mag_table.columns]
healthy_data = mag_table[healthy].T.values
crc_data = mag_table[crc].T.values

# Исходное различие (эвристика: сумма средних персистенций)
dgms_h = ripser.ripser(healthy_data, maxdim=1)['dgms']
dgms_c = ripser.ripser(crc_data, maxdim=1)['dgms']
obs_diff = (avg_persistence(dgms_h[0]) + avg_persistence(dgms_c[0])) - 2*np.mean([avg_persistence(d) for d in [dgms_h[0], dgms_c[0]]])
# Упрощённо: разница средних персистенций
diff_obs = abs(avg_persistence(dgms_h[0]) - avg_persistence(dgms_c[0]))

# Перестановочный тест (1000 итераций)
n_perm = 1000
diffs_perm = []
all_data = np.vstack([healthy_data, crc_data])
labels = np.array([0]*len(healthy_data) + [1]*len(crc_data))
for _ in range(n_perm):
    np.random.shuffle(labels)
    h_perm = all_data[labels == 0]
    c_perm = all_data[labels == 1]
    if len(h_perm)==0 or len(c_perm)==0:
        diffs_perm.append(0)
        continue
    dgms_h_perm = ripser.ripser(h_perm, maxdim=1)['dgms']
    dgms_c_perm = ripser.ripser(c_perm, maxdim=1)['dgms']
    diff_perm = abs(avg_persistence(dgms_h_perm[0]) - avg_persistence(dgms_c_perm[0]))
    diffs_perm.append(diff_perm)

p_value = np.mean(np.array(diffs_perm) >= diff_obs)
print(f"Observed difference: {diff_obs}")
print(f"Permutation p-value: {p_value}")
