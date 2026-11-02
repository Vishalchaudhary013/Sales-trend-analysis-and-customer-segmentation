"""Central configuration: paths, constants, colour theme and column schema.

Every tunable value in the project lives here so that UI pages, scripts and
library modules never hard-code paths or magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Paths (all relative to the repository root; pathlib only)
# --------------------------------------------------------------------------- #
ROOT_DIR: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = ROOT_DIR / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
UPLOADS_DIR: Final[Path] = RAW_DIR / "uploads"  # original uploads, preserved unchanged
MODELS_DIR: Final[Path] = ROOT_DIR / "models"
REPORTS_DIR: Final[Path] = ROOT_DIR / "reports"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"
METRICS_DIR: Final[Path] = REPORTS_DIR / "metrics"
DB_PATH: Final[Path] = DATA_DIR / "app.db"

RAW_FILES: Final[dict[str, str]] = {"math": "student-mat.csv", "portuguese": "student-por.csv"}
PROCESSED_FILE: Final[Path] = PROCESSED_DIR / "students_clean.csv"

# --------------------------------------------------------------------------- #
# Data source
# --------------------------------------------------------------------------- #
DATASET_PAGE_URL: Final[str] = "https://archive.ics.uci.edu/dataset/320/student+performance"
DATASET_ZIP_URL: Final[str] = (
    "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
)
CSV_SEPARATOR: Final[str] = ";"
EXPECTED_ROWS: Final[dict[str, int]] = {"math": 395, "portuguese": 649}
DATASET_CITATION: Final[str] = (
    "P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student "
    "Performance. In A. Brito and J. Teixeira (Eds.), Proceedings of 5th FUture BUsiness "
    "TEChnology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal, April 2008, EUROSIS."
)
ALLOWED_UPLOAD_SUFFIXES: Final[tuple[str, ...]] = (".csv", ".xlsx")
MAX_UPLOAD_MB: Final[int] = 50

# --------------------------------------------------------------------------- #
# Column schema (UCI Student Performance, 33 attributes)
# --------------------------------------------------------------------------- #
YES_NO: Final[tuple[str, ...]] = ("yes", "no")


@dataclass(frozen=True)
class ColumnSpec:
    """Validation rule for a single column.

    Attributes:
        name: Column name as it appears in the source file.
        kind: ``"numeric"`` or ``"categorical"``.
        min_value: Inclusive lower bound for numeric columns.
        max_value: Inclusive upper bound for numeric columns.
        allowed: Permitted values for categorical columns.
        description: Human-readable meaning (from the UCI data dictionary).
    """

    name: str
    kind: str
    min_value: float | None = None
    max_value: float | None = None
    allowed: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""


def _num(name: str, lo: float, hi: float, desc: str) -> ColumnSpec:
    return ColumnSpec(name, "numeric", lo, hi, (), desc)


def _cat(name: str, allowed: tuple[str, ...], desc: str) -> ColumnSpec:
    return ColumnSpec(name, "categorical", None, None, allowed, desc)


_JOBS = ("teacher", "health", "services", "at_home", "other")

SCHEMA: Final[tuple[ColumnSpec, ...]] = (
    _cat("school", ("GP", "MS"), "School (GP = Gabriel Pereira, MS = Mousinho da Silveira)"),
    _cat("sex", ("F", "M"), "Student's sex"),
    _num("age", 15, 22, "Student's age"),
    _cat("address", ("U", "R"), "Home address type (U = urban, R = rural)"),
    _cat("famsize", ("LE3", "GT3"), "Family size (<=3 or >3)"),
    _cat("Pstatus", ("T", "A"), "Parents' cohabitation status (T = together, A = apart)"),
    _num("Medu", 0, 4, "Mother's education (0 none - 4 higher)"),
    _num("Fedu", 0, 4, "Father's education (0 none - 4 higher)"),
    _cat("Mjob", _JOBS, "Mother's job"),
    _cat("Fjob", _JOBS, "Father's job"),
    _cat("reason", ("home", "reputation", "course", "other"), "Reason to choose this school"),
    _cat("guardian", ("mother", "father", "other"), "Student's guardian"),
    _num("traveltime", 1, 4, "Home-to-school travel time (1 <15 min - 4 >1 hour)"),
    _num("studytime", 1, 4, "Weekly study time (1 <2 h - 4 >10 h)"),
    _num("failures", 0, 4, "Number of past class failures"),
    _cat("schoolsup", YES_NO, "Extra educational support"),
    _cat("famsup", YES_NO, "Family educational support"),
    _cat("paid", YES_NO, "Extra paid classes in the course subject"),
    _cat("activities", YES_NO, "Extra-curricular activities"),
    _cat("nursery", YES_NO, "Attended nursery school"),
    _cat("higher", YES_NO, "Wants to take higher education"),
    _cat("internet", YES_NO, "Internet access at home"),
    _cat("romantic", YES_NO, "In a romantic relationship"),
    _num("famrel", 1, 5, "Quality of family relationships (1 very bad - 5 excellent)"),
    _num("freetime", 1, 5, "Free time after school (1 very low - 5 very high)"),
    _num("goout", 1, 5, "Going out with friends (1 very low - 5 very high)"),
    _num("Dalc", 1, 5, "Workday alcohol consumption (1 very low - 5 very high)"),
    _num("Walc", 1, 5, "Weekend alcohol consumption (1 very low - 5 very high)"),
    _num("health", 1, 5, "Current health status (1 very bad - 5 very good)"),
    _num("absences", 0, 93, "Number of school absences"),
    _num("G1", 0, 20, "First period grade"),
    _num("G2", 0, 20, "Second period grade"),
    _num("G3", 0, 20, "Final grade (outcome)"),
)
SCHEMA_BY_NAME: Final[dict[str, ColumnSpec]] = {c.name: c for c in SCHEMA}
REQUIRED_COLUMNS: Final[tuple[str, ...]] = tuple(c.name for c in SCHEMA)
NUMERIC_COLUMNS: Final[tuple[str, ...]] = tuple(c.name for c in SCHEMA if c.kind == "numeric")
CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = tuple(
    c.name for c in SCHEMA if c.kind == "categorical"
)

# Attributes used by Cortez & Silva to identify the same student across both files.
STUDENT_MATCH_KEYS: Final[tuple[str, ...]] = (
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "nursery", "internet",
)  # fmt: skip

# --------------------------------------------------------------------------- #
# Derived columns & targets
# --------------------------------------------------------------------------- #
SUBJECT_COL: Final[str] = "subject"
TARGET_COL: Final[str] = "pass"
OUTCOME_COL: Final[str] = "G3"
PASS_THRESHOLD: Final[int] = 10  # pass = 1 if G3 >= 10
POSITIVE_CLASS: Final[int] = 1  # "pass"; fail-class metrics treat 0 as positive
ABSENCE_BAND_COL: Final[str] = "absence_band"
ABSENCE_BINS: Final[tuple[float, ...]] = (-0.5, 0.5, 5.5, 10.5, 20.5, float("inf"))
ABSENCE_LABELS: Final[tuple[str, ...]] = ("0", "1–5", "6–10", "11–20", "20+")

# Never used as predictors: outcome and anything derived from it.
LEAKAGE_COLUMNS: Final[tuple[str, ...]] = (OUTCOME_COL, TARGET_COL)
PRIOR_GRADE_COLUMNS: Final[tuple[str, ...]] = ("G1", "G2")

# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
RANDOM_STATE: Final[int] = 42
CV_FOLDS: Final[int] = 5
CV_SHUFFLE: Final[bool] = True
DT_MAX_DEPTH: Final[int] = 5
RF_N_ESTIMATORS: Final[int] = 300
LR_MAX_ITER: Final[int] = 5000
DEFAULT_THRESHOLD: Final[float] = 0.5
FEATURE_SETTINGS: Final[dict[str, str]] = {
    "A": "Includes prior-period grades G1 and G2",
    "B": "Excludes G1 and G2 (early-warning scenario)",
}
DEFAULT_EARLY_WARNING_SETTING: Final[str] = "B"

# Risk tiers on P(fail)
RISK_HIGH: Final[float] = 0.7
RISK_MEDIUM: Final[float] = 0.4
TOP_N_REASONS: Final[int] = 3

# --------------------------------------------------------------------------- #
# Visual theme (Okabe-Ito colour-blind-safe palette)
# --------------------------------------------------------------------------- #
PALETTE: Final[dict[str, str]] = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "purple": "#CC79A7",
    "grey": "#7F7F7F",
}
OUTCOME_COLORS: Final[dict[str, str]] = {"Pass": PALETTE["blue"], "Fail": PALETTE["vermillion"]}
OUTCOME_PATTERNS: Final[dict[str, str]] = {"Pass": "", "Fail": "/"}  # redundant encoding
SUBJECT_COLORS: Final[dict[str, str]] = {"math": PALETTE["sky"], "portuguese": PALETTE["orange"]}
RISK_COLORS: Final[dict[str, str]] = {
    "High": PALETTE["vermillion"],
    "Medium": PALETTE["orange"],
    "Low": PALETTE["sky"],
}
SEQUENTIAL_SCALE: Final[str] = "Blues"
DIVERGING_SCALE: Final[str] = "RdBu"
FONT_FAMILY: Final[str] = "Inter, Segoe UI, Helvetica, Arial, sans-serif"
BASE_FONT_SIZE: Final[int] = 14
PLOT_TEMPLATE: Final[str] = "plotly_white"
EXPORT_DPI: Final[int] = 300

# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
APP_TITLE: Final[str] = "Student Performance Analytics & Early-Warning Dashboard"
FILTER_STATE_KEY: Final[str] = "global_filters"
DATASET_STATE_KEY: Final[str] = "active_dataset"


def ensure_directories() -> None:
    """Create all runtime directories if they do not already exist."""
    for path in (RAW_DIR, PROCESSED_DIR, UPLOADS_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR):
        path.mkdir(parents=True, exist_ok=True)
