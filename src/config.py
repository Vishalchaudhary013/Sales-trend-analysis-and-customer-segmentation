"""Central configuration: paths, constants, column schema, modelling and theme.

Every tunable value lives here so UI pages, scripts and library modules never
hard-code paths, thresholds or magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Paths (relative to the repository root; pathlib only)
# --------------------------------------------------------------------------- #
ROOT_DIR: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = ROOT_DIR / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
UPLOADS_DIR: Final[Path] = RAW_DIR / "uploads"  # original uploads, preserved unchanged
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
MODELS_DIR: Final[Path] = ROOT_DIR / "models"
REPORTS_DIR: Final[Path] = ROOT_DIR / "reports"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"
METRICS_DIR: Final[Path] = REPORTS_DIR / "metrics"
DB_PATH: Final[Path] = DATA_DIR / "app.db"

RAW_FILE: Final[Path] = RAW_DIR / "online_retail.csv"  # canonical bundled raw file
PROCESSED_TRANSACTIONS: Final[Path] = PROCESSED_DIR / "transactions_clean.csv"
PROCESSED_CUSTOMERS: Final[Path] = PROCESSED_DIR / "customers_rfm.csv"
PROCESSED_CACHE: Final[Path] = PROCESSED_DIR / "transactions_clean.joblib"  # fast app load
LARGE_LINE_QUANTITY: Final[int] = 10_000  # lines at/above this are surfaced as outliers

# --------------------------------------------------------------------------- #
# Data source: UCI Online Retail (dataset 352)
# --------------------------------------------------------------------------- #
DATASET_NAME: Final[str] = "UCI Online Retail"
DATASET_PAGE_URL: Final[str] = "https://archive.ics.uci.edu/dataset/352/online+retail"
DATASET_ZIP_URL: Final[str] = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
DATASET_ZIP_MEMBER: Final[str] = "Online Retail.xlsx"
# Public CSV copy of the same data (Databricks, "Spark: The Definitive Guide").
DATASET_MIRROR_URL: Final[str] = (
    "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/"
    "data/retail-data/all/online-retail-dataset.csv"
)
EXPECTED_ROWS: Final[int] = 541_909
CSV_ENCODINGS: Final[tuple[str, ...]] = ("utf-8-sig", "latin-1")
DATASET_CITATION: Final[str] = (
    "D. Chen, S. L. Sain and K. Guo. Data mining for the online retail industry: A case "
    "study of RFM model-based customer segmentation using data mining. Journal of Database "
    "Marketing and Customer Strategy Management, 19(3), 197-208, 2012. "
    "Dataset: UCI Machine Learning Repository, https://doi.org/10.24432/C5BW33."
)
ALLOWED_UPLOAD_SUFFIXES: Final[tuple[str, ...]] = (".csv", ".xlsx")
MAX_UPLOAD_MB: Final[int] = 200

# --------------------------------------------------------------------------- #
# Column schema
# --------------------------------------------------------------------------- #
COL_INVOICE: Final[str] = "InvoiceNo"
COL_STOCK: Final[str] = "StockCode"
COL_DESC: Final[str] = "Description"
COL_QTY: Final[str] = "Quantity"
COL_DATE: Final[str] = "InvoiceDate"
COL_PRICE: Final[str] = "UnitPrice"
COL_CUSTOMER: Final[str] = "CustomerID"
COL_COUNTRY: Final[str] = "Country"


@dataclass(frozen=True)
class ColumnSpec:
    """Validation rule for one source column.

    Attributes:
        name: Canonical column name.
        kind: ``"string"``, ``"integer"``, ``"float"`` or ``"datetime"``.
        required: Whether the column must be present for the app to work.
        nullable: Whether missing values are acceptable (reported, never imputed).
        description: Human-readable meaning.
        aliases: Alternative source names that are explicitly mapped to ``name``
            (e.g. UCI Online Retail II spellings). Mappings are always reported.
    """

    name: str
    kind: str
    required: bool = True
    nullable: bool = False
    description: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)


SCHEMA: Final[tuple[ColumnSpec, ...]] = (
    ColumnSpec(COL_INVOICE, "string", description="Invoice number; prefix 'C' = cancellation",
               aliases=("Invoice", "Invoice No", "InvoiceNumber")),
    ColumnSpec(COL_STOCK, "string", description="Product (item) code",
               aliases=("Stock Code", "ProductCode", "SKU")),
    ColumnSpec(COL_DESC, "string", required=False, nullable=True,
               description="Product name", aliases=("ProductName",)),
    ColumnSpec(COL_QTY, "integer", description="Units per line; negative for returns",
               aliases=("Qty", "Units")),
    ColumnSpec(COL_DATE, "datetime", description="Invoice date and time",
               aliases=("Invoice Date", "OrderDate", "Date")),
    ColumnSpec(COL_PRICE, "float", description="Unit price in GBP",
               aliases=("Price", "Unit Price")),
    ColumnSpec(COL_CUSTOMER, "string", nullable=True,
               description="Customer number; missing for guest / unregistered sales",
               aliases=("Customer ID", "Customer", "CustomerNo")),
    ColumnSpec(COL_COUNTRY, "string", required=False, nullable=True,
               description="Customer's country", aliases=("Region",)),
)  # fmt: skip
SCHEMA_BY_NAME: Final[dict[str, ColumnSpec]] = {c.name: c for c in SCHEMA}
REQUIRED_COLUMNS: Final[tuple[str, ...]] = tuple(c.name for c in SCHEMA if c.required)
ALL_COLUMNS: Final[tuple[str, ...]] = tuple(c.name for c in SCHEMA)

CANCELLATION_PREFIX: Final[str] = "C"
ADJUSTMENT_PREFIX: Final[str] = "A"  # accounting adjustments (e.g. "Adjust bad debt")
# Date formats tried in order; the first that parses (almost) every value wins.
# The bundled CSV uses US month/day order ("12/1/2010 8:26" = 1 Dec 2010).
DATE_FORMATS: Final[tuple[str, ...]] = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
    "%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S",
)  # fmt: skip
DATE_PARSE_MIN_SHARE: Final[float] = 0.95  # = 1 - MAX_INVALID_ROW_SHARE
# Unparseable values are dropped (and reported); above this share it is an error.
MAX_INVALID_ROW_SHARE: Final[float] = 0.05
# Stock codes that are fees/adjustments rather than products (flagged, not deleted).
NON_PRODUCT_STOCK_CODES: Final[frozenset[str]] = frozenset(
    {"POST", "DOT", "M", "m", "C2", "D", "S", "B", "CRUK", "PADS", "BANK CHARGES",
     "AMAZONFEE", "ADJUST", "ADJUST2", "TEST001", "TEST002"}
)  # fmt: skip
NON_PRODUCT_PREFIXES: Final[tuple[str, ...]] = ("gift_",)

# Derived columns
COL_REVENUE: Final[str] = "Revenue"
COL_IS_CANCEL: Final[str] = "IsCancellation"
COL_HAS_CUSTOMER: Final[str] = "HasCustomer"
COL_IS_PRODUCT: Final[str] = "IsProduct"
COL_IS_ADJUST: Final[str] = "IsAdjustment"
COL_IS_REVERSED: Final[str] = "IsReversed"  # sale later reversed by a matching cancellation

# --------------------------------------------------------------------------- #
# RFM segmentation
# --------------------------------------------------------------------------- #
RFM_QUANTILES: Final[int] = 5
RFM_SNAPSHOT_OFFSET_DAYS: Final[int] = 1  # snapshot = last invoice date + 1 day
# Ordered (regex on "<R score><F score>", segment name). First match wins.
RFM_SEGMENT_RULES: Final[tuple[tuple[str, str], ...]] = (
    (r"5[4-5]", "Champions"),
    (r"[3-4][4-5]", "Loyal customers"),
    (r"[4-5][2-3]", "Potential loyalists"),
    (r"51", "New customers"),
    (r"41", "Promising"),
    (r"33", "Need attention"),
    (r"3[1-2]", "About to sleep"),
    (r"[1-2]5", "Can't lose them"),
    (r"[1-2][3-4]", "At risk"),
    (r"[1-2][1-2]", "Hibernating"),
)

# K-Means
KMEANS_K_RANGE: Final[tuple[int, int]] = (2, 10)  # inclusive, evaluated
KMEANS_N_INIT: Final[int] = 10
KMEANS_MAX_DISPLAY_K: Final[int] = 6  # one validated colour (and symbol) per cluster
RFM_FEATURES: Final[tuple[str, ...]] = ("recency", "frequency", "monetary")
# Display order of RFM segments (best to worst) for charts and tables.
RFM_SEGMENT_ORDER: Final[tuple[str, ...]] = (
    "Champions", "Loyal customers", "Potential loyalists", "New customers", "Promising",
    "Need attention", "About to sleep", "Can't lose them", "At risk", "Hibernating",
)  # fmt: skip
RFM_SEGMENT_ACTIONS: Final[dict[str, str]] = {
    "Champions": "Reward; early access; ask for reviews and referrals.",
    "Loyal customers": "Upsell higher-value ranges; loyalty programme.",
    "Potential loyalists": "Offer membership or bundles to build frequency.",
    "New customers": "Onboarding journey; second-purchase incentive.",
    "Promising": "Build awareness of the range; time-limited offers.",
    "Need attention": "Personalised recommendations; reactivation offers.",
    "About to sleep": "Win-back reminders before they lapse.",
    "Can't lose them": "High-value but lapsing: personal outreach, strong win-back.",
    "At risk": "Targeted win-back campaigns; ask for feedback.",
    "Hibernating": "Low-cost reactivation; otherwise deprioritise.",
}
CLUSTER_SYMBOLS: Final[tuple[str, ...]] = (
    "circle", "square", "diamond", "triangle-up", "x", "star",
)  # fmt: skip
# Validated (CVD + contrast, with secondary encoding: marker symbols and labels).
CLUSTER_COLORS: Final[tuple[str, ...]] = (
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00",
)  # fmt: skip

# --------------------------------------------------------------------------- #
# Churn early-warning model
# --------------------------------------------------------------------------- #
RANDOM_STATE: Final[int] = 42
CV_FOLDS: Final[int] = 5
CV_SHUFFLE: Final[bool] = True
CHURN_HORIZON_DAYS: Final[int] = 90  # label window at the end of the data
CHURN_POSITIVE_LABEL: Final[str] = "churned"  # churn = 1: no purchase in the horizon
DT_MAX_DEPTH: Final[int] = 5
RF_N_ESTIMATORS: Final[int] = 300
LR_MAX_ITER: Final[int] = 5000
DEFAULT_THRESHOLD: Final[float] = 0.5
RISK_HIGH: Final[float] = 0.7
RISK_MEDIUM: Final[float] = 0.4
TOP_N_REASONS: Final[int] = 3

# --------------------------------------------------------------------------- #
# Visual theme (Okabe-Ito based, colour-blind safe)
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
SINGLE_SERIES_COLOR: Final[str] = PALETTE["blue"]
# Fixed categorical order for part-to-whole charts (<= 5 named slices + "Other").
# Validated for CVD separation with secondary encoding (direct slice labels + gaps).
CATEGORICAL_ORDER: Final[tuple[str, ...]] = (
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
)  # fmt: skip
OTHER_COLOR: Final[str] = "#9AA0A6"
MAX_PIE_SLICES: Final[int] = 6  # including "Other"
NEUTRAL_COLOR: Final[str] = "#4D4D4D"
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
CURRENCY_SYMBOL: Final[str] = "£"  # symbol of the base (source) currency

# --------------------------------------------------------------------------- #
# Currency conversion
# --------------------------------------------------------------------------- #
BASE_CURRENCY: Final[str] = "GBP"  # currency of the source data
FX_FILE: Final[Path] = RAW_DIR / "fx_daily_per_usd.csv"
# Daily Federal Reserve (H.10, via FRED) rates, packaged by datahub.io (ODC-PDDL).
# Convention verified against market anchors: every series = units of currency per 1 USD.
FX_URL: Final[str] = "https://raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv"
FX_SOURCE: Final[str] = (
    "US Federal Reserve H.10 daily rates (via FRED), packaged by datahub.io "
    "core/exchange-rates (ODC-PDDL)"
)
# ISO code -> (name, display prefix, country label in the FX file; None = USD itself).
CURRENCIES: Final[dict[str, tuple[str, str, str | None]]] = {
    "GBP": ("British pound", "£", "United Kingdom"),
    "USD": ("US dollar", "$", None),
    "EUR": ("Euro", "€", "Euro"),
    "INR": ("Indian rupee", "₹", "India"),
    "JPY": ("Japanese yen", "¥", "Japan"),
    "CNY": ("Chinese yuan", "CN¥", "China"),
    "AUD": ("Australian dollar", "A$", "Australia"),
    "CAD": ("Canadian dollar", "C$", "Canada"),
    "CHF": ("Swiss franc", "CHF ", "Switzerland"),
    "SGD": ("Singapore dollar", "S$", "Singapore"),
    "HKD": ("Hong Kong dollar", "HK$", "Hong Kong"),
    "NZD": ("New Zealand dollar", "NZ$", "New Zealand"),
    "SEK": ("Swedish krona", "SEK ", "Sweden"),
    "NOK": ("Norwegian krone", "NOK ", "Norway"),
    "DKK": ("Danish krone", "DKK ", "Denmark"),
    "ZAR": ("South African rand", "R", "South Africa"),
    "BRL": ("Brazilian real", "R$", "Brazil"),
    "MXN": ("Mexican peso", "MX$", "Mexico"),
    "KRW": ("South Korean won", "₩", "South Korea"),
    "MYR": ("Malaysian ringgit", "RM", "Malaysia"),
    "THB": ("Thai baht", "฿", "Thailand"),
    "TWD": ("New Taiwan dollar", "NT$", "Taiwan"),
}
FX_MODES: Final[dict[str, str]] = {
    "daily": "Daily rate on each invoice date",
    "average": "Single average rate over the data period",
}
CURRENCY_STATE_KEY: Final[str] = "currency_setting"

# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
APP_TITLE: Final[str] = "Sales Trend Analysis & Customer Segmentation Dashboard"
FILTER_STATE_KEY: Final[str] = "global_filters"
DATASET_STATE_KEY: Final[str] = "active_dataset"


def ensure_directories() -> None:
    """Create all runtime directories if they do not already exist."""
    for path in (RAW_DIR, UPLOADS_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR):
        path.mkdir(parents=True, exist_ok=True)
