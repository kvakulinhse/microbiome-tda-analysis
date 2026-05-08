import numpy as np
import pandas as pd
import ripser

def avg_persistence(dgm):
    if dgm.shape[0] == 0:
        return 0.0
    return np.mean(dgm[:, 1] - dgm[:, 0])

metadata = pd.read_csv("sampleID.csv")
mag_table = pd.read_csv("vect_atlas.csv", index_col=0)

healthy = metadata[metadata['Disease'] == "Healthy"]["sample.ID"].tolist()
crc = metadata[metadata['Disease'] == "CRC"]["sample.ID"].tolist()
healthy = [s for s in healthy if s in mag_table.columns]
crc = [s for s in crc if s in mag_table.columns]

healthy_data = mag_table[healthy].T.values
crc_data = mag_table[crc].T.values

# Используем 8 ядер
dgms_h = ripser.ripser(healthy_data, maxdim=1, n_jobs=8)['dgms']
dgms_c = ripser.ripser(crc_data, maxdim=1, n_jobs=8)['dgms']
obs = abs(avg_persistence(dgms_h[0]) - avg_persistence(dgms_c[0]))
print(f"Observed diff: {obs}")

n_perm = 100   # или больше, теперь быстрее
diffs = []
all_data = np.vstack([healthy_data, crc_data])
labels = np.array([0]*len(healthy_data) + [1]*len(crc_data))

for i in range(n_perm):
    np.random.shuffle(labels)
    h = all_data[labels == 0]
    c = all_data[labels == 1]
    if len(h) == 0 or len(c) == 0:
        diffs.append(0.0)
        continue
    dh = ripser.ripser(h, maxdim=1, n_jobs=8)['dgms']
    dc = ripser.ripser(c, maxdim=1, n_jobs=8)['dgms']
    diff = abs(avg_persistence(dh[0]) - avg_persistence(dc[0]))
    diffs.append(diff)

pval = np.mean(np.array(diffs) >= obs)
print(f"p-value: {pval}")
