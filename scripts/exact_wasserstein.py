import numpy as np
from persim import wasserstein

def clean_diagram(dgm, eps=1e-12):
    """Удаляет точки с inf/NaN и где birth >= death, добавляет малый шум к диагональным"""
    if dgm.shape[0] == 0:
        return dgm
    # Фильтруем только конечные и birth < death
    mask = np.isfinite(dgm[:, 0]) & np.isfinite(dgm[:, 1]) & (dgm[:, 0] < dgm[:, 1])
    dgm = dgm[mask].astype(np.float64)
    # Добавляем шум к точкам, у которых смерть равна рождению (вдруг остались)
    mask_eq = (dgm[:, 0] == dgm[:, 1])
    dgm[mask_eq, 1] += eps
    return dgm

def safe_wasserstein(dgm1, dgm2, eps=1e-12):
    dgm1 = clean_diagram(dgm1, eps)
    dgm2 = clean_diagram(dgm2, eps)
    if len(dgm1) == 0 and len(dgm2) == 0:
        return 0.0
    try:
        return wasserstein(dgm1, dgm2)
    except Exception as e:
        # Если не удалось, используем сумму средних персистенций
        p1 = np.mean(dgm1[:,1] - dgm1[:,0]) if len(dgm1) else 0.0
        p2 = np.mean(dgm2[:,1] - dgm2[:,0]) if len(dgm2) else 0.0
        print(f"Warning: fallback to average persistence, {e}")
        return p1 + p2

if __name__ == "__main__":
    dgms_h = np.load("dgms_h_clean.npy", allow_pickle=True)
    dgms_c = np.load("dgms_c_clean.npy", allow_pickle=True)

    dist0 = safe_wasserstein(dgms_h[0], dgms_c[0])
    dist1 = safe_wasserstein(dgms_h[1], dgms_c[1])

    print(f"Corrected Wasserstein H0: {dist0}")
    print(f"Corrected Wasserstein H1: {dist1}")
