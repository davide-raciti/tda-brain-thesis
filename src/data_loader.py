import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    EMOTIONS_DIR, EMOTION_PKL, FMRI_DIR,
    FILM_LENGTHS, TR_OFFSET, EXCLUDED_SUBJECTS,
)


# ─────────────────────────────────────────────────────────────────────────────
# fMRI
# ─────────────────────────────────────────────────────────────────────────────

def load_fmri(subject_id, film_name, resolution):
    """
    Load and standardize the fMRI time series for one subject and film.

    Parameters
    ----------
    subject_id  : str  — 'sub-S01' ... 'sub-S32', or 'avg' for group mean
    film_name   : str  — film key as in FILM_LENGTHS (e.g. 'Sintel')
    resolution  : str  — number of ROIs, e.g. '100', '200', '400', '800'

    Returns
    -------
    X : ndarray, shape (T, n_rois), standardized (mean=0, std=1 per ROI)
      - T : number of TR
      - n_rois : number of ROIs (regions of interest of the brain)
    """
    data_dir = FMRI_DIR / f"{resolution}_sub"
    L = FILM_LENGTHS[film_name]
    scaler = StandardScaler() # normalizes to mean = 0 and standard deviation = 1

    # creation of the average activity matrix for the given film
    if subject_id == 'avg':
        subject_ids = [
            f"sub-S{i:02d}" for i in range(1, 33)
            if i not in EXCLUDED_SUBJECTS
        ]
        matrices = []
        for sid in subject_ids:
            mat = _load_single(sid, film_name, resolution, data_dir, L)
            if mat is not None:
                matrices.append(mat)
        if not matrices:
            return None
        return scaler.fit_transform(np.mean(matrices, axis=0)) # average standarized signal for the group

    # creation of the activity matrix for the given (subject, film) pair
    mat = _load_single(subject_id, film_name, resolution, data_dir, L)
    if mat is None:
        return None
    return scaler.fit_transform(mat) # standardizd signal for the subject

def _load_single(subject_id, film_name, resolution, data_dir, L):
    if resolution == '100':
        path = data_dir / f"TC_114_{subject_id}_{film_name}.txt"
        if not path.exists():
            print(f"    -> [SKIPPED] {subject_id} / {film_name}")
            return None
        return np.loadtxt(path)
    else:
        path = data_dir / f"{subject_id}_task-{film_name}.npy"
        if not path.exists():
            print(f"    -> [SKIPPED] {subject_id} / {film_name}")
            return None
        raw = np.load(path)
        return raw[TR_OFFSET: TR_OFFSET + L, :]


# ─────────────────────────────────────────────────────────────────────────────
# EMOTIONS
# ─────────────────────────────────────────────────────────────────────────────

# 3 emotional values to ticket the nodes of the mapper graphs
def load_emotions_3fa(film_name):
    """
    Load the 3-component emotion signal (Valence, Arousal, Dominance) for one film.

    Returns
    -------
    E : ndarray, shape (T, 3)  — raw continuous values
    y : DataFrame, shape (T, 3) — one-hot dominant component per TR
    """
    path = EMOTIONS_DIR / f"3FA_13_{film_name}_stim.tsv"
    E = np.loadtxt(path)
    
    # assigns 1 to the component with the highest absolute value, 0 to the two others
    # we need it to color the mapper nodes
    dominant_idx = np.argmax(np.abs(E), axis=1)
    y = pd.DataFrame(
        np.eye(3, dtype=int)[dominant_idx], 
        columns=['Valence', 'Arousal', 'Dominance']
    ).reset_index(drop=True)

    return E, y

# 50 emotional values for the analysis of correlation between emotional activity and geometry
def load_all50():
    """
    Load the full 50-item CPM emotion timeseries for all films.

    Returns
    -------
    dict : {film_name: ndarray(T, 50)}  — raw values, no normalization
    """
    with open(EMOTION_PKL, 'rb') as f:
        raw = pickle.load(f)
    return {
        film: raw[film]['All50_Emotion'].astype(float)
        for film in raw
    }


# ─────────────────────────────────────────────────────────────────────────────
# FRAME — combined input for the Mapper pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_frame(subject_id, film_name, resolution='100'):
    """
    Build the complete input frame for one (subject, film) pair.

    Returns
    -------
    dict with keys:
        'fMRI_frame'    : ndarray (T, n_rois)  — standardized fMRI
        'emotions_frame': ndarray (T, 3)       — raw 3FA signals
        'labels_frame'  : DataFrame (T, 3)     — one-hot dominant emotion
    None if any required file is missing or shapes are inconsistent.
    """
    X = load_fmri(subject_id, film_name, resolution)
    if X is None:
        return None

    E, y = load_emotions_3fa(film_name)

    if X.shape[0] != len(E):
        print(f"    -> [ERROR] shape mismatch: fMRI has {X.shape[0]} TRs, "
              f"emotions have {len(E)} for {subject_id} / {film_name}")
        return None

    return {
        'fMRI_frame':     X,
        'emotions_frame': E,
        'labels_frame':   y,
    }
