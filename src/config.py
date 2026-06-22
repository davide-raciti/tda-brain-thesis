from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# ROOT — change this if the project moves to a different machine or directory
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(r"C:\Users\dvdrc\Desktop\BrainMapper_Movie")

# Repo-internal locations (relative to this file, independent of PROJECT_ROOT)
REPO_ROOT   = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"   # CSV/PNG outputs written by the analysis scripts (5_*)

# ─────────────────────────────────────────────────────────────────────────────
# DATA PATHS
# ─────────────────────────────────────────────────────────────────────────────

DATA_RAW       = PROJECT_ROOT / "data" / "raw"
EMOTIONS_DIR   = DATA_RAW / "3FA_films"           # emotion TSV files (3FA_13_<film>_stim.tsv)
EMOTION_PKL    = DATA_RAW / "emotion_ts_dict.pkl"  # All50_Emotion time series
FMRI_DIR       = DATA_RAW / "TC_emofilms"          # fMRI time series by resolution
GRAPHS_DIR     = PROJECT_ROOT / "0_graphs_html"    # saved Mapper graphs (JSON)
PRIMARY_CONFIG     = "100_riso"                     # reference configuration for the main analyses
PRIMARY_GRAPHS_DIR = GRAPHS_DIR / PRIMARY_CONFIG / "graphs"  # JSON graphs read by the figure scripts

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE OUTPUT
# Folder where the fig_*.py scripts save the thesis images.
# EDIT THIS to point at your own thesis image directory.
# ─────────────────────────────────────────────────────────────────────────────

FIGURES_DIR    = PROJECT_ROOT / "tesi_Davide_overleaf" / "images"

# ─────────────────────────────────────────────────────────────────────────────
# SUBJECTS
# ─────────────────────────────────────────────────────────────────────────────

EXCLUDED_SUBJECTS = {12, 18}                       # missing/corrupted acquisitions
SUBJECT_IDS = [
    f"sub-S{i:02d}" for i in range(1, 33)
    if i not in EXCLUDED_SUBJECTS
]

# ─────────────────────────────────────────────────────────────────────────────
# FILMS
# ─────────────────────────────────────────────────────────────────────────────

FILMS = [
    "AfterTheRain", "BetweenViewings", "BigBuckBunny", "Chatter",
    "FirstBite",    "LessonLearned",   "Payload",      "Sintel",
    "Spaceman",     "Superhero",       "TearsOfSteel", "TheSecretNumber",
    "ToClaireFromSonny", "YouAgain",
]

FILM_LENGTHS = {
    "AfterTheRain":      382,
    "BetweenViewings":   622,
    "BigBuckBunny":      377,
    "Chatter":           312,
    "FirstBite":         461,
    "LessonLearned":     513,
    "Payload":           775,
    "Sintel":            555,
    "Spaceman":          619,
    "Superhero":         791,
    "TearsOfSteel":      452,
    "TheSecretNumber":   603,
    "ToClaireFromSonny": 309,
    "YouAgain":          614,
}

T_REST = 460  # number of TRs for the resting-state condition

# ─────────────────────────────────────────────────────────────────────────────
# MAPPER PARAMETERS  (Saggar et al. 2018)
# ─────────────────────────────────────────────────────────────────────────────

TR_OFFSET         = 71           # leading TRs to discard from .npy files (HRF stabilization)
COVER_OVERLAP     = 0.7          # CubicalCover overlap fraction
REF_DENSITY       = 1017 / 30.0  # reference TR/min ratio used to auto-set n_intervals
RISOMAP_NEIGHBORS = 15           # n_neighbors for ReciprocalIsomap lens
CLUSTERER_K       = 3            # k for k-NN distance threshold estimation
CLUSTERER_P       = 90.0         # percentile for agglomerative clustering threshold

# ─────────────────────────────────────────────────────────────────────────────
# EMOTION ANNOTATION
# ─────────────────────────────────────────────────────────────────────────────

# CPM component categories → column slices in All50_Emotion (0-based)
CPM_SLICES = {
    "Appraisal":             slice(0,  10),
    "Motor_Expression":      slice(10, 15),
    "Action_Tendencies":     slice(15, 25),
    "Subjective_Feeling":    slice(25, 32),
    "Physiological_Arousal": slice(32, 37),
}

# 7 discrete emotions → column index in All50_Emotion (0-based)
DISCRETE_EMOTIONS = {
    "Anger":     37,
    "Guilt":     38,
    "Disgust":   40,
    "Happiness": 41,
    "Fear":      42,
    "Surprise":  47,
    "Sad":       49,
}
