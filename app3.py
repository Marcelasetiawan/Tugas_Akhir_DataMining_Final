"""
╔══════════════════════════════════════════════════════╗
║  AgriYield · Data Mining Dashboard                  ║
║  Mini Project — Agricultural Yield Prediction       ║
║  Model: Random Forest Regressor (Tuned)             ║
╚══════════════════════════════════════════════════════╝
"""
import os, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="AgriYield · Data Mining",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════════════
BG      = "#F7F4CF"
CARD    = "#c6d384"
CARD2   = "#f0f4ec"
BORDER  = "#d4e0cc"
BORDER2 = "#a8c89a"
CYAN    = "#79823A"
CYAN2   = "#D2D857"
GREEN   = "#5FA83A"
GOLD    = "#c07c00"
PURPLE  = "#6d4dc9"
RED     = "#dc2626"
ORANGE  = "#c2500a"
BLUE    = "#2e8b57"
T1      = "#1a2e12"
T2      = "#FFFFFF"
T3      = "#6b8c5a"

# ══════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"], p, div, label {{
    font-family: 'Space Grotesk', sans-serif !important;
    color: {T1};
}}

/* ── App shell ── */
.stApp {{ background: {BG} !important; }}
.main .block-container {{
    background: {BG};
    padding: 0 2rem 3rem 2rem;
    max-width: 1400px;
}}
.stApp > header,
[data-testid="stHeader"] {{
    background: {BG} !important;
    border-bottom: 1px solid {BORDER} !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {CARD} !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="stSidebar"] * {{ color: {T2} !important; }}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] select {{
    background: {CARD2} !important;
    border: 1px solid {BORDER} !important;
    color: {T1} !important;
    border-radius: 8px !important;
}}
[data-testid="stSidebar"] .stSlider > div > div {{
    background: {BORDER2};
}}
[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {{
    background: {CYAN} !important; color: {BG} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important; border-radius: 4px;
}}


/* ── Number inputs — no double arrows ── */
input[type="number"] {{
    background: {CARD2} !important;
    border: 1px solid {BORDER} !important;
    color: {T1} !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    -moz-appearance: textfield !important;
}}
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button {{
    -webkit-appearance: none !important;
    display: none !important;
    margin: 0 !important;
}}
input[type="number"]:focus {{
    border-color: {CYAN} !important;
    box-shadow: 0 0 0 2px rgba(126,200,80,0.18) !important;
    outline: none !important;
}}
/* ── Number input step buttons — hide keyboard_double text, show arrows ── */
[data-testid="stNumberInput"] > div {{
    background: {CARD2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
}}
[data-testid="stNumberInput"] button {{
    background: transparent !important;
    border: none !important;
    padding: 0 6px !important;
    min-width: 28px !important;
    position: relative !important;
    overflow: hidden !important;
    color: transparent !important;
    font-size: 0 !important;
    line-height: 0 !important;
    text-indent: -9999px !important;
}}
[data-testid="stNumberInput"] button * {{
    color: transparent !important;
    font-size: 0 !important;
    line-height: 0 !important;
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    text-indent: -9999px !important;
    position: absolute !important;
    pointer-events: none !important;
}}
[data-testid="stNumberInput"] button::after {{
    font-size: 11px !important;
    line-height: 1 !important;
    display: block !important;
    visibility: visible !important;
    color: {T2} !important;
    text-indent: 0 !important;
    position: static !important;
    font-family: 'Arial', sans-serif !important;
    width: auto !important;
    height: auto !important;
}}
[data-testid="stNumberInput"] button[aria-label*="decrement"]::after,
[data-testid="stNumberInput"] button[aria-label*="Decrement"]::after,
[data-testid="stNumberInput"] button:first-of-type::after {{
    content: "▼" !important;
}}
[data-testid="stNumberInput"] button[aria-label*="increment"]::after,
[data-testid="stNumberInput"] button[aria-label*="Increment"]::after,
[data-testid="stNumberInput"] button:last-of-type::after {{
    content: "▲" !important;
}}
[data-testid="stNumberInput"] button:hover::after {{
    color: {CYAN} !important;
}}

/* ── Selectbox ── */
.stSelectbox > div > div {{
    background: {CARD2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {T1} !important;
    font-size: 13px !important;
}}
.stSelectbox > div > div:hover {{ border-color: {BORDER2} !important; }}

/* ── Tabs ── */
[data-baseweb="tab-list"] {{
    background: transparent !important;
    border-bottom: 1px solid {BORDER} !important;
    gap: 0 !important;
    padding: 0 !important;
}}
[data-baseweb="tab"] {{
    background: transparent !important;
    color: {T3} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 18px !important;
    transition: all 0.15s !important;
}}
[data-baseweb="tab"]:hover {{ color: {T2} !important; background: {CARD2} !important; }}
[aria-selected="true"] {{
    color: {CYAN} !important;
    border-bottom: 2px solid {CYAN} !important;
    background: transparent !important;
}}
[data-testid="stTabsContent"] {{
    background: transparent !important;
    padding: 0 !important;
    border: none !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background: linear-gradient(135deg, #3d7a1a, {CYAN}) !important;
    color: {BG} !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px !important;
    padding: 12px 28px !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(126,200,80,0.25) !important;
}}

.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(126,200,80,0.4) !important;
}}
.stButton > button:active {{ transform: translateY(0) !important; }}

/* ── Metrics ── */
[data-testid="stMetric"] {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    transition: border-color 0.2s !important;
}}
[data-testid="stMetric"]:hover {{ border-color: {BORDER2} !important; }}
[data-testid="stMetricLabel"] p {{
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    color: {T3} !important;
    font-family: 'JetBrains Mono', monospace !important;
}}
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
    color: {CYAN} !important;
    font-size: 22px !important;
}}
[data-testid="stMetricDelta"] {{ font-size: 11px !important; }}

/* ── Dataframe ── */
[data-testid="stDataFrame"] iframe {{
    border-radius: 10px !important;
    border: 1px solid {BORDER} !important;
}}

/* ── Progress ── */
[data-testid="stProgress"] > div > div > div {{
    background: linear-gradient(90deg, #ef4444, #fb923c, #fbbf24, #5aaa25, #7ec850) !important;
    border-radius: 100px !important;
}}
[data-testid="stProgress"] > div > div {{
    background: {CARD2} !important;
    border-radius: 100px !important;
}}

/* ── Input Section Cards (menggantikan expander) ── */
.inp-section {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    padding: 16px 18px 14px 18px !important;
    margin-bottom: 10px !important;
}}
.inp-section-title {{
    font-size: 12px !important;
    font-weight: 700 !important;
    color: {CYAN} !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    font-family: 'JetBrains Mono', monospace !important;
    margin-bottom: 12px !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid {BORDER} !important;
    display: block !important;
}}

/* ── Divider ── */
hr {{ border-color: {BORDER} !important; margin: 1.5rem 0 !important; }}

/* ── Alert ── */
[data-testid="stAlert"] {{ border-radius: 10px !important; border: none !important; }}

/* ── Spinner ── */
[data-testid="stSpinner"] * {{ color: {CYAN} !important; }}

/* ── Download button ── */
.stDownloadButton > button {{
    background: {CARD2} !important;
    border: 1px solid {BORDER2} !important;
    color: {T2} !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    padding: 8px 16px !important;
    width: auto !important;
    box-shadow: none !important;
}}
.stDownloadButton > button:hover {{
    border-color: {CYAN} !important;
    color: {CYAN} !important;
    transform: none !important;
    box-shadow: none !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER2}; border-radius: 2px; }}

/* ── File uploader — fix double "upload" text ── */
[data-testid="stFileUploader"] section {{
    background: {CARD2} !important;
    border: 1.5px dashed {BORDER2} !important;
    border-radius: 10px !important;
    padding: 18px 20px !important;
}}
[data-testid="stFileUploader"] section > div {{
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {{
    font-size: 12px !important;
    color: {T3} !important;
}}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] span {{
    display: none !important;
}}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"]::after {{
    content: "Seret file CSV ke sini atau klik Browse files" !important;
    font-size: 12px !important;
    color: {T3} !important;
    display: block !important;
    visibility: visible !important;
}}
[data-testid="stFileUploader"] button[data-testid="baseButton-secondary"] {{
    background: {CYAN} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 7px 16px !important;
}}
[data-testid="stFileUploader"] button[data-testid="baseButton-secondary"]:hover {{
    opacity: 0.85 !important;
}}

/* ── Custom components ── */
.page-header {{
    padding: 28px 0 20px;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 28px;
}}
.page-header .eyebrow {{
    font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 3px; color: {CYAN}; margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
}}
.page-header h1 {{
    font-size: 26px; font-weight: 700; color: {T1}; margin: 0;
    line-height: 1.2;
}}
.page-header .sub {{
    font-size: 13px; color: {T3}; margin-top: 6px;
}}

.kpi-row {{ display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }}
.kpi {{
    flex: 1; min-width: 140px;
    background: {CARD}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 16px 18px;
    border-top: 2px solid var(--c);
    transition: transform 0.2s;
}}
.kpi:hover {{ transform: translateY(-2px); }}
.kpi .kpi-label {{
    font-size: 9px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 2px; color: {T3};
    font-family: 'JetBrains Mono', monospace; margin-bottom: 6px;
}}
.kpi .kpi-val {{
    font-size: 26px; font-weight: 700; color: var(--c);
    font-family: 'JetBrains Mono', monospace; line-height: 1;
}}
.kpi .kpi-sub {{
    font-size: 10px; color: {T3}; margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}}

.sec-hdr {{
    font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 2.5px; color: {T3};
    font-family: 'JetBrains Mono', monospace;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 6px; margin: 20px 0 14px;
}}
.tag {{
    display: inline-block;
    background: rgba(126,200,80,0.1); border: 1px solid rgba(126,200,80,0.25);
    border-radius: 100px; padding: 2px 10px;
    font-size: 10px; font-weight: 600; color: {CYAN};
    font-family: 'JetBrains Mono', monospace; letter-spacing: 0.5px;
}}

.result-hero {{
    background: linear-gradient(135deg, {CARD}, #0c200f);
    border: 1px solid {BORDER}; border-left: 3px solid var(--c);
    border-radius: 14px; padding: 28px 32px; text-align: center;
    box-shadow: 0 0 40px rgba(126,200,80,0.08);
    transition: all 0.3s;
}}
.result-hero .rh-label {{
    font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 3px; color: {T3}; margin-bottom: 12px;
    font-family: 'JetBrains Mono', monospace;
}}
.result-hero .rh-val {{
    font-size: 72px; font-weight: 700; color: var(--c); line-height: 1;
    font-family: 'JetBrains Mono', monospace;
}}
.result-hero .rh-unit {{
    font-size: 14px; color: {T3}; margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}}
.result-hero .rh-cat {{
    display: inline-block; margin-top: 14px;
    background: rgba(255,255,255,0.06); border: 1px solid {BORDER2};
    border-radius: 100px; padding: 6px 20px;
    font-size: 14px; font-weight: 600; color: var(--c);
}}

.info-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 14px;
    background: {CARD2}; border: 1px solid {BORDER};
    border-radius: 8px; margin-bottom: 5px;
}}
.info-row .ik {{ font-size: 12px; color: {T3}; }}
.info-row .iv {{
    font-size: 12px; font-weight: 600; color: {T2};
    font-family: 'JetBrains Mono', monospace;
}}

.model-chip {{
    display: inline-flex; align-items: center; gap: 6px;
    background: {CARD2}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 6px 12px;
    font-size: 11px; font-family: 'JetBrains Mono', monospace; color: {T2};
}}

.sidebar-section {{
    font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 2.5px; color: {T3};
    font-family: 'JetBrains Mono', monospace;
    border-bottom: 1px solid {BORDER}; padding-bottom: 5px;
    margin: 16px 0 10px;
}}

.nav-item {{
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; border-radius: 8px; cursor: pointer;
    margin-bottom: 4px; transition: all 0.15s;
    border: 1px solid transparent;
    font-size: 13px; font-weight: 500; color: {T2};
}}
.nav-item:hover, .nav-item.active {{
    background: {CARD2}; border-color: {BORDER};
    color: {T1};
}}
.nav-item.active {{ border-color: {CYAN}; color: {CYAN}; }}

.gauge-wrap {{
    background: {CARD2}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 14px 16px;
}}
.gauge-labels {{
    display: flex; justify-content: space-between;
    font-size: 9px; color: {T3}; margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
}}
.gauge-track {{
    height: 8px; background: {BG};
    border-radius: 100px; overflow: hidden;
}}
.gauge-fill {{
    height: 100%; border-radius: 100px;
    background: linear-gradient(90deg, #ef4444 0%, #fb923c 25%, #fbbf24 50%, #5aaa25 75%, #7ec850 100%);
    transition: width 0.8s cubic-bezier(0.34,1.56,0.64,1);
}}
.gauge-center {{
    text-align: center; margin-top: 7px;
    font-size: 11px; font-family: 'JetBrains Mono', monospace; color: {T3};
}}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# MPL THEME
# ══════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor":   "#ffffff",
    "axes.facecolor":     "#f8faf6",
    "axes.edgecolor":     BORDER,
    "axes.labelcolor":    T2,
    "xtick.color":        T3,
    "ytick.color":        T3,
    "text.color":         T1,
    "grid.color":         BORDER,
    "grid.alpha":         0.5,
    "grid.linewidth":     0.6,
    "font.family":        "monospace",
    "font.size":          8.5,
    "axes.titlesize":     10,
    "axes.titlepad":      10,
    "axes.labelsize":     8.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         130,
    "legend.framealpha":  0.15,
    "legend.edgecolor":   BORDER,
    "legend.fontsize":    7.5,
})
CMAP_CUSTOM = LinearSegmentedColormap.from_list("agri", ["#d4e8c2", GREEN, CYAN, CYAN2])

def rf(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

import io
def rf_tab(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)

# ══════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ══════════════════════════════════════════════════════
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(show_spinner=False)
def load_model():
    d = os.path.join(BASE, "saved_model")
    return (
        joblib.load(f"{d}/final_model.pkl"),
        joblib.load(f"{d}/scaler.pkl"),
        joblib.load(f"{d}/feature_columns.pkl"),
        joblib.load(f"{d}/ordinal_maps.pkl"),
    )

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(os.path.join(BASE, "Agri_yield_prediction.csv"))

with st.spinner("🌾  Loading model artifacts…"):
    try:
        MODEL, SCALER, FEAT_COLS, ORDINAL_MAPS = load_model()
        MODEL_OK  = True
        MODEL_NAME = type(MODEL).__name__
    except Exception as e:
        MODEL_OK = False; MODEL_ERR = str(e)
        MODEL_NAME = "Unknown"

try:
    DF = load_data(); DATA_OK = True
except: DATA_OK = False; DF = None

if not MODEL_OK:
    st.error(f"❌ **Model gagal dimuat:** {MODEL_ERR}")
    st.info("Pastikan folder `saved_model/` berisi `final_model.pkl`, `scaler.pkl`, "
            "`feature_columns.pkl`, `ordinal_maps.pkl`")
    st.stop()

# ══════════════════════════════════════════════════════
# REFERENCE DATA (dari notebook output)
# ══════════════════════════════════════════════════════
ALL_MODELS_DF = pd.DataFrame([
    {"Model":"Random Forest (Tuned) ★","Train_R2":0.6821,"Val_R2":-0.0008,"Train_RMSE":1.4921,"Val_RMSE":2.6069,"Test_R2":-0.006, "Test_RMSE":2.6500,"Test_MAE":2.3089},
    {"Model":"Random Forest",          "Train_R2":0.7102,"Val_R2":-0.0038,"Train_RMSE":1.4244,"Val_RMSE":2.6108,"Test_R2":-0.0118,"Test_RMSE":2.6577,"Test_MAE":2.3150},
    {"Model":"Gradient Boosting",      "Train_R2":0.5431,"Val_R2":-0.0105,"Train_RMSE":1.7908,"Val_RMSE":2.6195,"Test_R2":-0.0181,"Test_RMSE":2.6659,"Test_MAE":2.3161},
    {"Model":"XGBoost",                "Train_R2":0.8123,"Val_R2":-0.0416,"Train_RMSE":1.1502,"Val_RMSE":2.6595,"Test_R2":-0.0435,"Test_RMSE":2.6990,"Test_MAE":2.3402},
    {"Model":"Decision Tree",          "Train_R2":0.9999,"Val_R2":-0.0566,"Train_RMSE":0.0412,"Val_RMSE":2.6785,"Test_R2":-0.0630,"Test_RMSE":2.7240,"Test_MAE":2.3483},
    {"Model":"LightGBM",               "Train_R2":0.7215,"Val_R2":-0.0632,"Train_RMSE":1.4013,"Val_RMSE":2.6869,"Test_R2":-0.0645,"Test_RMSE":2.7260,"Test_MAE":2.3710},
])

META = dict(
    test_r2=-0.006, test_rmse=2.6500, test_mae=2.3089,
    val_r2=-0.0008, val_rmse=2.6069,
    n_features=len(FEAT_COLS), samples=10000,
    algo=MODEL_NAME,
    split="70 / 15 / 15",
)

# ══════════════════════════════════════════════════════
# INFERENCE FUNCTION (exact dari notebook cell 6.8)
# ══════════════════════════════════════════════════════
def run_inference(d: dict) -> float:
    row = pd.DataFrame([d])
    # Ordinal encoding
    for col, order in ORDINAL_MAPS.items():
        if col in row.columns:
            row[col] = OrdinalEncoder(categories=[order]).fit_transform(row[[col]])
    # One-hot encoding
    nom = ["Soil_Type","Crop_Type","Fertilizer_Type","Region","Season"]
    row = pd.get_dummies(row, columns=[c for c in nom if c in row.columns], dtype=int)
    # Engineered features
    for a,b,k in [("N","P","NPK_Sum"),("N","P","N_P_ratio"),("P","K","P_K_ratio")]:
        if a in row.columns and b in row.columns:
            if k == "NPK_Sum":    row[k] = row.get("N",0)+row.get("P",0)+row.get("K",0)
            elif k == "N_P_ratio":row[k] = row.get("N",0)/(row.get("P",1e-6)+1e-6)
            else:                  row[k] = row.get("P",0)/(row.get("K",1e-6)+1e-6)
    if "Temperature" in row.columns:
        row["Temp_Rain"]      = row.get("Temperature",0)*row.get("Rainfall",0)/1000
        row["NDVI_GDD"]       = row.get("NDVI",0)*row.get("GDD",0)
        row["Soil_pH_EC"]     = row.get("pH",0)*row.get("EC",0)
        row["WHC_Bulk"]       = row.get("Water_Holding_Capacity",0)/(row.get("Bulk_Density",1)+1e-6)
        row["SandClay_ratio"] = row.get("Sand",0)/(row.get("Clay",1e-6)+1e-6)
    row = row.reindex(columns=FEAT_COLS, fill_value=0)
    return round(float(MODEL.predict(SCALER.transform(row))[0]), 4)

def yield_info(y):
    if   y < 2: return "Sangat Rendah", RED,    "🔴", 0.10
    elif y < 4: return "Rendah",        ORANGE,  "🟠", 0.30
    elif y < 6: return "Sedang",        GOLD,    "🟡", 0.55
    elif y < 8: return "Tinggi",        GREEN,   "🟢", 0.77
    else:       return "Sangat Tinggi", CYAN,    "🌟", 0.95

# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    # Logo
    st.markdown(f"""
    <div style='padding:20px 4px 12px;text-align:center'>
      <div style='font-size:32px;margin-bottom:6px'>🌾</div>
      <div style='font-size:17px;font-weight:700;color:{T1}'>AgriYield</div>
      <div style='font-size:10px;color:{T3};margin-top:2px;
                  font-family:"JetBrains Mono",monospace'>
        Data Mining Dashboard
      </div>
    </div>
    <hr style='border-color:{BORDER};margin:0 0 4px'>
    """, unsafe_allow_html=True)

    # Navigation
    pages = {
        "Prediksi Yield":   "predict",
        "Analisis Data":    "data",
        "Preprocessing":   "preprocessing",
        "Perbandingan Model": "model",
        "Batch Testing":   "batch",
        "Riwayat":         "history",
    }
    if "page" not in st.session_state: st.session_state["page"] = "predict"
    for label, key in pages.items():
        active = "active" if st.session_state["page"] == key else ""
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state["page"] = key
            st.rerun()

    st.divider()

    # Model info chip
    st.markdown(f"""
    <div class='sidebar-section'>MODEL INFO</div>
    <div style='font-size:11px;color:{T3};font-family:"JetBrains Mono",monospace;line-height:2'>
      Algo &nbsp;&nbsp;&nbsp;&nbsp;: <span style='color:{CYAN}'>{MODEL_NAME}</span><br>
      Fitur &nbsp;&nbsp;&nbsp;: <span style='color:{CYAN}'>{len(FEAT_COLS)}</span> kolom<br>
      Scaler &nbsp;&nbsp;: <span style='color:{CYAN}'>RobustScaler</span><br>
      Split &nbsp;&nbsp;&nbsp;: <span style='color:{CYAN}'>70/15/15</span><br>
      CV &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <span style='color:{CYAN}'>5-Fold KFold</span><br>
      Tuning &nbsp;&nbsp;: <span style='color:{CYAN}'>GridSearchCV</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

# ══════════════════════════════════════════════════════
# COLLECT INPUT — handled inside PAGE: PREDICT below
# ══════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════
PAGE_META = {
    "predict":      ("MINI PROJECT DATA MINING",       "🎯  Prediksi Hasil Panen",
                     "Isi parameter lahan di bawah ini, lalu klik tombol Prediksi Yield"),
    "data":         ("EXPLORATORY DATA ANALYSIS",      "📊  Analisis Dataset",
                     "Distribusi, korelasi, dan karakteristik data pertanian"),
    "preprocessing":("DATA PREPROCESSING PIPELINE",   "🔬  Tahapan Preprocessing",
                     "Data sebelum & sesudah noise, balancing, encoding, scaling — lengkap per tahap"),
    "model":        ("MODEL EVALUATION",               "🤖  Perbandingan & Evaluasi Model",
                     "5 algoritma regresi, tuning, dan feature importance"),
    "batch":        ("BATCH TESTING & EVALUATION",     "🧪  Batch Testing",
                     "Upload CSV, jalankan prediksi massal, dan lihat metrik RMSE · R² · MAE · MAPE"),
    "history":      ("SESSION HISTORY",                "📋  Riwayat Prediksi",
                     "Semua prediksi yang dilakukan dalam sesi ini"),
}
eyebrow, title, sub = PAGE_META[st.session_state["page"]]
st.markdown(f"""
<div class='page-header'>
  <div class='eyebrow'>{eyebrow}</div>
  <h1>{title}</h1>
  <div class='sub'>{sub}</div>
</div>
""", unsafe_allow_html=True)

# ── KPI Strip ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='kpi-row'>
  <div class='kpi' style='--c:{RED}'>
    <div class='kpi-label'>Test R²</div>
    <div class='kpi-val'>{META['test_r2']:.4f}</div>
    <div class='kpi-sub'>Best model</div>
  </div>
  <div class='kpi' style='--c:{ORANGE}'>
    <div class='kpi-label'>Test RMSE</div>
    <div class='kpi-val'>{META['test_rmse']:.4f}</div>
    <div class='kpi-sub'>ton/ha</div>
  </div>
  <div class='kpi' style='--c:{GOLD}'>
    <div class='kpi-label'>Test MAE</div>
    <div class='kpi-val'>{META['test_mae']:.4f}</div>
    <div class='kpi-sub'>ton/ha</div>
  </div>
  <div class='kpi' style='--c:{CYAN}'>
    <div class='kpi-label'>Algoritma</div>
    <div class='kpi-val' style='font-size:14px;padding-top:6px'>{MODEL_NAME.replace("Regressor","")}</div>
    <div class='kpi-sub'>Best tuned</div>
  </div>
  <div class='kpi' style='--c:{GREEN}'>
    <div class='kpi-label'>Dataset</div>
    <div class='kpi-val'>10K</div>
    <div class='kpi-sub'>samples · 46 fitur</div>
  </div>
  <div class='kpi' style='--c:{PURPLE}'>
    <div class='kpi-label'>Fitur Model</div>
    <div class='kpi-val'>{len(FEAT_COLS)}</div>
    <div class='kpi-sub'>consensus selected</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["page"] == "predict":

    st.markdown("<div class='sec-hdr'>⚙️ Parameter Input</div>", unsafe_allow_html=True)

    # ── 1. Kondisi Iklim ──
    st.markdown("<div class='inp-section'><span class='inp-section-title'>🌤 Kondisi Iklim</span>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        Temperature = st.slider("Suhu / Temperature (°C)", 10.0, 45.0, 26.5, 0.1)
        Rainfall    = st.slider("Curah Hujan / Rainfall (mm)", 0.0, 350.0, 180.0, 1.0)
    with c2:
        Humidity        = st.slider("Kelembapan / Humidity (%)", 30.0, 95.0, 65.0, 0.5)
        Solar_Radiation = st.slider("Radiasi Surya / Solar Radiation (MJ/m²)", 500.0, 3000.0, 2000.0, 10.0)
    with c3:
        Wind_Speed = st.slider("Kecepatan Angin / Wind Speed (km/h)", 0.0, 40.0, 12.0, 0.1)
        GDD        = st.slider("GDD — Growing Degree Days (°C·day)", 200.0, 3500.0, 1800.0, 10.0)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 2. Kimia Tanah ──
    st.markdown("<div class='inp-section'><span class='inp-section-title'>🧪 Kimia Tanah</span>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:{T3};margin-bottom:6px'>Keasaman & Konduksi</div>", unsafe_allow_html=True)
        pH  = st.slider("pH Tanah", 4.0, 9.0, 6.8, 0.01)
        EC  = st.slider("EC (dS/m)", 0.0, 5.0, 1.2, 0.01)
        OC  = st.slider("OC (%)", 0.0, 5.0, 1.0, 0.01)
        CEC = st.slider("CEC (cmolc/kg)", 0.0, 60.0, 25.0, 0.1)
    with c2:
        st.markdown(f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:{T3};margin-bottom:6px'>Makronutrien N-P-K</div>", unsafe_allow_html=True)
        N  = st.slider("N (mg/kg)", 0.0, 300.0, 120.0, 1.0)
        P  = st.slider("P (mg/kg)", 0.0, 200.0, 80.0, 1.0)
        K  = st.slider("K (mg/kg)", 0.0, 400.0, 160.0, 1.0)
        Ca = st.slider("Ca (mg/kg)", 0.0, 2000.0, 800.0, 10.0)
    with c3:
        st.markdown(f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:{T3};margin-bottom:6px'>Makro Sekunder</div>", unsafe_allow_html=True)
        Mg = st.slider("Mg (mg/kg)", 0.0, 1000.0, 300.0, 10.0)
        S  = st.slider("S (mg/kg)", 0.0, 200.0, 30.0, 1.0)
        Zn = st.slider("Zn (mg/kg)", 0.0, 10.0, 2.0, 0.01)
        Fe = st.slider("Fe (mg/kg)", 0.0, 60.0, 20.0, 0.1)
    with c4:
        st.markdown(f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:{T3};margin-bottom:6px'>Mikronutrien</div>", unsafe_allow_html=True)
        Cu = st.slider("Cu (mg/kg)", 0.0, 10.0, 1.0, 0.01)
        Mn = st.slider("Mn (mg/kg)", 0.0, 50.0, 15.0, 0.1)
        B  = st.slider("B (mg/kg)", 0.0, 5.0, 1.5, 0.01)
        Mo = st.slider("Mo (mg/kg)", 0.0, 2.0, 0.4, 0.01)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 3. Fisika Tanah ──
    st.markdown("<div class='inp-section'><span class='inp-section-title'>🌱 Fisika Tanah</span>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:{T3};margin-bottom:6px'>Tekstur (%)</div>", unsafe_allow_html=True)
        Sand = st.slider("Pasir / Sand (%)", 0.0, 100.0, 30.0, 0.5)
        Silt = st.slider("Debu / Silt (%)", 0.0, 100.0, 40.0, 0.5)
        Clay = st.slider("Liat / Clay (%)", 0.0, 100.0, 30.0, 0.5)
    with c2:
        st.markdown(f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:{T3};margin-bottom:6px'>Sifat Fisik</div>", unsafe_allow_html=True)
        Bulk_Density           = st.slider("Bulk Density (g/cm³)", 0.8, 2.0, 1.3, 0.01)
        Water_Holding_Capacity = st.slider("WHC (%)", 5.0, 70.0, 35.0, 0.1)
    with c3:
        st.markdown(f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:{T3};margin-bottom:6px'>Jenis Tanah</div>", unsafe_allow_html=True)
        Soil_Type = st.selectbox("Soil Type", ["Loamy","Sandy","Clayey","Silty"])
    with c4:
        st.markdown(f"""
        <div style='background:{CARD2};border:1px solid {BORDER};border-radius:8px;
                    padding:10px 12px;font-size:11px;color:{T3};font-family:"JetBrains Mono",monospace;line-height:1.9'>
          <div style='color:{CYAN};font-weight:700;font-size:10px;margin-bottom:4px'>Fitur Turunan</div>
          SandClay = Sand/Clay<br>WHC_Bulk = WHC/BD
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 4. Topografi & Vegetasi ──
    st.markdown("<div class='inp-section'><span class='inp-section-title'>⛰ Topografi & Vegetasi</span>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:{T3};margin-bottom:6px'>Topografi</div>", unsafe_allow_html=True)
        Slope     = st.slider("Slope (°)", 0.0, 45.0, 5.0, 0.1)
        Aspect    = st.slider("Aspect (°)", 0.0, 360.0, 180.0, 1.0)
        Elevation = st.slider("Elevasi (mdpl)", 100.0, 4000.0, 200.0, 1.0)
    with c2:
        st.markdown(f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:{T3};margin-bottom:6px'>Indeks Vegetasi</div>", unsafe_allow_html=True)
        NDVI = st.slider("NDVI", -1.0, 1.0, 0.65, 0.001)
        EVI  = st.slider("EVI",  -1.0, 1.0, 0.50, 0.001)
        LAI  = st.slider("LAI",   0.0,  8.0,  3.5, 0.01)
    with c3:
        st.markdown(f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:{T3};margin-bottom:6px'>Klorofil</div>", unsafe_allow_html=True)
        Chlorophyll = st.slider("Chlorophyll SPAD (mg/m²)", 5.0, 70.0, 35.0, 0.5)
    with c4:
        st.markdown(f"""
        <div style='background:{CARD2};border:1px solid {BORDER};border-radius:8px;
                    padding:10px 12px;font-size:11px;color:{T3};font-family:"JetBrains Mono",monospace;line-height:1.9'>
          <div style='color:{CYAN};font-weight:700;font-size:10px;margin-bottom:4px'>Fitur Turunan</div>
          NDVI_GDD = NDVI×GDD<br>Temp_Rain = T×R/1000<br>Soil_pH_EC = pH×EC
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 5. Manajemen Pertanian ──
    st.markdown("<div class='inp-section'><span class='inp-section-title'>🚜 Manajemen Pertanian</span>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:{T3};margin-bottom:6px'>Tanaman & Fase</div>", unsafe_allow_html=True)
        Crop_Type    = st.selectbox("Jenis Tanaman", ["Maize","Rice","Soybean","Wheat"])
        Growth_Stage = st.selectbox("Fase Pertumbuhan", ["Vegetative","Reproductive","Maturity"], index=1)
    with c2:
        st.markdown(f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:{T3};margin-bottom:6px'>Input Pertanian</div>", unsafe_allow_html=True)
        Fertilizer_Type = st.selectbox("Jenis Pupuk", ["Mixed","Organic","Chemical"])
        Pesticide_Usage = st.selectbox("Pestisida", ["Low","Medium","High"], index=1)
    with c3:
        st.markdown(f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:{T3};margin-bottom:6px'>Wilayah & Musim</div>", unsafe_allow_html=True)
        Region = st.selectbox("Region", ["North","South","East","West"], index=1)
        Season = st.selectbox("Musim Tanam", ["Kharif","Rabi","Zaid"])
    with c4:
        st.markdown(f"<div style='font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;color:{T3};margin-bottom:6px'>Irigasi</div>", unsafe_allow_html=True)
        Irrigation_Frequency = st.number_input("Frekuensi Irigasi (x/minggu)", 0, 14, 7)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 6. Waktu Tanam ──
    st.markdown("<div class='inp-section'><span class='inp-section-title'>📅 Waktu Tanam</span>", unsafe_allow_html=True)
    bln = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]
    c1, c2, c3 = st.columns(3)
    with c1:
        Planting_Month = st.selectbox("Bulan Tanam", range(1,13), index=5,
                                      format_func=lambda x: f"{bln[x-1]} ({x})")
    with c2:
        Harvest_Month  = st.selectbox("Bulan Panen", range(1,13), index=8,
                                      format_func=lambda x: f"{bln[x-1]} ({x})")
    with c3:
        Growing_Days = st.number_input("Hari Tumbuh (Growing Days)", 30, 365, 90)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, col_spc = st.columns([1, 3])
    with col_btn:
        predict_btn = st.button("🌾  Prediksi Yield Sekarang", use_container_width=True)

    # ── Collect & run inference ──
    INPUT = dict(
        Temperature=Temperature, Humidity=Humidity, Rainfall=Rainfall,
        Solar_Radiation=Solar_Radiation, Wind_Speed=Wind_Speed, GDD=GDD,
        pH=pH, EC=EC, OC=OC, N=N, P=P, K=K,
        Ca=Ca, Mg=Mg, S=S, Zn=Zn, Fe=Fe, Cu=Cu, Mn=Mn, B=B, Mo=Mo, CEC=CEC,
        Sand=Sand, Silt=Silt, Clay=Clay,
        Bulk_Density=Bulk_Density, Water_Holding_Capacity=Water_Holding_Capacity,
        Slope=Slope, Aspect=Aspect, Elevation=Elevation,
        NDVI=NDVI, EVI=EVI, LAI=LAI, Chlorophyll=Chlorophyll,
        Soil_Type=Soil_Type, Crop_Type=Crop_Type, Growth_Stage=Growth_Stage,
        Irrigation_Frequency=Irrigation_Frequency, Fertilizer_Type=Fertilizer_Type,
        Pesticide_Usage=Pesticide_Usage, Region=Region, Season=Season,
        Growing_Days=Growing_Days, Planting_Month=Planting_Month, Harvest_Month=Harvest_Month,
    )
    if predict_btn:
        with st.spinner("Memproses prediksi…"):
            try:
                pred = run_inference(INPUT)
                lbl, col, emo, gauge = yield_info(pred)
                st.session_state.update({"pred":pred,"lbl":lbl,"col":col,"emo":emo,
                                         "gauge":gauge,"inp":INPUT.copy()})
                if "history" not in st.session_state: st.session_state["history"] = []
                st.session_state["history"].insert(0,{
                    "Crop":Crop_Type,"Region":Region,"Season":Season,
                    "Fertilizer":Fertilizer_Type,"Pesticide":Pesticide_Usage,
                    "Yield (ton/ha)":pred,"Kategori":f"{emo} {lbl}",
                    "Sumber":"🎯 Single"
                })
            except Exception as e:
                st.error(f"❌ Inference error: {e}")

    # ── Tampilkan hasil jika sudah pernah prediksi ──
    if "pred" not in st.session_state:
        st.markdown(f"""
        <div style='background:{CARD};border:1px dashed {BORDER2};border-radius:12px;
                    padding:32px;text-align:center;color:{T3};margin-top:16px'>
          <div style='font-size:36px;margin-bottom:10px'>🌾</div>
          <div style='font-size:15px;font-weight:700;color:{T2};margin-bottom:6px'>
            Atur parameter di atas, lalu klik <span style='color:{CYAN}'>🌾 Prediksi Yield Sekarang</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        pred  = st.session_state["pred"]
        lbl   = st.session_state["lbl"]
        col   = st.session_state["col"]
        emo   = st.session_state["emo"]
        gauge = st.session_state["gauge"]
        inp   = st.session_state["inp"]

        # ── Result + Input Summary ──────────────────────────────────────────────────
        r1, r2 = st.columns([1, 1], gap="large")

        with r1:
            # Big result card
            st.markdown(f"""
            <div class='result-hero' style='--c:{col}'>
              <div class='rh-label'>HASIL PREDIKSI</div>
              <div class='rh-val'>{pred:.4f}</div>
              <div class='rh-unit'>ton / hektar</div>
              <div class='rh-cat'>{emo} &nbsp;{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge
            pct = int(gauge * 100)
            st.markdown(f"""
            <br>
            <div class='gauge-wrap'>
              <div class='gauge-labels'>
                <span>1.0 — Sangat Rendah</span>
                <span>5.5 — Sedang</span>
                <span>10.0 — Sangat Tinggi</span>
              </div>
              <div class='gauge-track'>
                <div class='gauge-fill' style='width:{pct}%'></div>
              </div>
              <div class='gauge-center'>
                Posisi: <strong style='color:{col}'>{pred:.4f}</strong> / 10.0 &nbsp; ({pct}%)
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Position in dataset distribution
            if DATA_OK:
                st.markdown(f"<div class='sec-hdr'>Posisi dalam Distribusi Dataset</div>",
                            unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5.5, 2.8))
                ax.hist(DF["Yield"], bins=45, color=CARD2, edgecolor=BG, lw=0.3, alpha=0.9)
                ax.axvline(pred, color=col, lw=2.5, ls="--",
                           label=f"Prediksi: {pred:.4f}")
                ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1]>0 else 500],
                                  pred-0.3, pred+0.3, color=col, alpha=0.08)
                ax.set_xlabel("Yield (ton/ha)"); ax.set_ylabel("Frekuensi")
                ax.set_title("Posisi Nilai Prediksi vs Distribusi Dataset", pad=8)
                ax.legend(); ax.grid(True, alpha=0.3)
                rf(fig)

        with r2:
            st.markdown(f"<div class='sec-hdr'>Ringkasan Input Parameter</div>",
                        unsafe_allow_html=True)
            npk = inp["N"] + inp["P"] + inp["K"]
            summary = [
                ("🌾 Tanaman",      inp["Crop_Type"]),
                ("🗺  Region",       inp["Region"]),
                ("☀️  Musim",        inp["Season"]),
                ("🌡  Suhu",         f"{inp['Temperature']:.1f} °C"),
                ("💧  Curah Hujan",  f"{inp['Rainfall']:.0f} mm"),
                ("💦  Kelembapan",   f"{inp['Humidity']:.1f} %"),
                ("🌿  NDVI",         f"{inp['NDVI']:.3f}"),
                ("☀️  GDD",          f"{inp['GDD']:.0f} °C·day"),
                ("🧪  pH Tanah",     f"{inp['pH']:.2f}"),
                ("⚗️  Total NPK",    f"{npk:.0f} mg/kg"),
                ("💊  Pupuk",        inp["Fertilizer_Type"]),
                ("🐛  Pestisida",    inp["Pesticide_Usage"]),
                ("🌱  Fase Tumbuh",  inp["Growth_Stage"]),
                ("📅  Durasi",       f"{inp['Growing_Days']} hari"),
            ]
            for k, v in summary:
                st.markdown(f"""
                <div class='info-row'>
                  <span class='ik'>{k}</span>
                  <span class='iv'>{v}</span>
                </div>""", unsafe_allow_html=True)

            # Engineered features display
            st.markdown(f"<div class='sec-hdr'>Fitur Turunan (Engineered)</div>",
                        unsafe_allow_html=True)
            eng = [
                ("NPK_Sum",       f"{npk:.2f}"),
                ("N_P_ratio",     f"{inp['N']/(inp['P']+1e-6):.4f}"),
                ("P_K_ratio",     f"{inp['P']/(inp['K']+1e-6):.4f}"),
                ("Temp_Rain",     f"{inp['Temperature']*inp['Rainfall']/1000:.4f}"),
                ("NDVI_GDD",      f"{inp['NDVI']*inp['GDD']:.2f}"),
                ("Soil_pH_EC",    f"{inp['pH']*inp['EC']:.4f}"),
                ("WHC_Bulk",      f"{inp['Water_Holding_Capacity']/(inp['Bulk_Density']+1e-6):.4f}"),
                ("SandClay_ratio",f"{inp['Sand']/(inp['Clay']+1e-6):.4f}"),
            ]
            for k, v in eng:
                st.markdown(f"""
                <div class='info-row'>
                  <span class='ik' style='font-family:"JetBrains Mono",monospace;
                                          font-size:11px;color:{T3}'>{k}</span>
                  <span class='iv' style='color:{CYAN}'>{v}</span>
                </div>""", unsafe_allow_html=True)

            # Rekomendasi
            if pred < 4:
                rec_c, rec = ORANGE, "⚠️ Yield rendah — Tingkatkan dosis pupuk NPK, perbaiki drainase, dan optimalkan irigasi."
            elif pred < 7:
                rec_c, rec = GOLD,   "✅ Yield cukup — Optimasi nutrisi tanah dan varietas bisa meningkatkan produktivitas."
            else:
                rec_c, rec = GREEN,  "🌟 Yield sangat baik — Kondisi lahan dan manajemen sudah mendukung produktivitas optimal!"
            st.markdown(f"""
            <div style='background:{CARD2};border:1px solid {rec_c};border-radius:10px;
                        padding:14px 16px;margin-top:12px'>
              <div style='font-size:10px;font-weight:700;color:{rec_c};
                          text-transform:uppercase;letter-spacing:1.5px;margin-bottom:5px'>
                Rekomendasi
              </div>
              <div style='font-size:12px;color:{T2};line-height:1.7'>{rec}</div>
            </div>
            """, unsafe_allow_html=True)

            # Radar chart
            st.markdown(f"<div class='sec-hdr'>Profil Parameter (Radar)</div>",
                        unsafe_allow_html=True)
            categories = ["Suhu","Kelembapan","Curah Hujan","NDVI","NPK","GDD","pH"]
            norm_v = [
                min((inp["Temperature"]-10)/35, 1),
                inp["Humidity"]/100,
                min(inp["Rainfall"]/350, 1),
                max(min((inp["NDVI"]+1)/2, 1), 0),
                min((inp["N"]+inp["P"]+inp["K"])/600, 1),
                min(inp["GDD"]/3500, 1),
                min((inp["pH"]-4)/6, 1),
            ]
            N_r = len(categories)
            angles = [n/N_r*2*np.pi for n in range(N_r)]
            vals_r = norm_v + [norm_v[0]]
            angs_r = angles + [angles[0]]

            fig_r, ax_r = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
            fig_r.patch.set_facecolor(BG)
            ax_r.set_facecolor(CARD)
            ax_r.plot(angs_r, vals_r, color=CYAN, lw=2.2, zorder=5)
            ax_r.fill(angs_r, vals_r, color=CYAN, alpha=0.18)
            for angle, val in zip(angles, norm_v):
                ax_r.plot([angle, angle], [0, val], color=CYAN, lw=1, alpha=0.4)
                ax_r.scatter([angle], [val], s=40, color=CYAN, zorder=6, edgecolors=BG, lw=1.5)
            ax_r.set_xticks(angles)
            ax_r.set_xticklabels(categories, color=T2, fontsize=8.5)
            ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax_r.set_yticklabels(["25%","50%","75%","100%"], color=T3, fontsize=6.5)
            ax_r.grid(color=BORDER, linewidth=0.8)
            ax_r.spines["polar"].set_color(BORDER)
            ax_r.set_title("Parameter\n(Ternormalisasi)", color=T2, fontsize=9, pad=14)
            col_r1, col_r2, col_r3 = st.columns([1,2,1])
            with col_r2: rf(fig_r)

        # ── Visualisasi Berdasarkan Input ──────────────────────────────────────
        st.markdown(f"<div class='sec-hdr'>📊 Visualisasi Parameter Input Anda</div>",
                    unsafe_allow_html=True)

        vi1, vi2 = st.columns(2)

        with vi1:
            # Bar chart nutrisi makro
            fig_n, ax_n = plt.subplots(figsize=(5, 3.5))
            nutrients = ["N", "P", "K", "Ca", "Mg", "S"]
            nut_vals  = [inp["N"], inp["P"], inp["K"], inp["Ca"], inp["Mg"], inp["S"]]
            nut_colors = [CYAN, GREEN, GOLD, ORANGE, PURPLE, RED]
            bars_n = ax_n.bar(nutrients, nut_vals, color=nut_colors, edgecolor="#ffffff", lw=0.5, width=0.6)
            for bar, v in zip(bars_n, nut_vals):
                ax_n.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                          f"{v:.0f}", ha="center", fontsize=8, fontweight="bold", color=T2)
            ax_n.set_title("Profil Nutrisi Input (mg/kg)", fontsize=10, fontweight="bold")
            ax_n.set_ylabel("mg/kg"); ax_n.grid(axis="y", ls="--", alpha=0.4)
            plt.tight_layout(); rf(fig_n)

        with vi2:
            # Kondisi iklim — gauge bars
            fig_c, ax_c = plt.subplots(figsize=(5, 3.5))
            climate_labels = ["Suhu\n(°C)", "Kelembapan\n(%)", "Curah Hujan\n/350", "NDVI\n(+1)/2", "GDD\n/3500"]
            climate_norm   = [
                (inp["Temperature"]-10)/35,
                inp["Humidity"]/100,
                min(inp["Rainfall"]/350, 1),
                max(min((inp["NDVI"]+1)/2, 1), 0),
                min(inp["GDD"]/3500, 1),
            ]
            climate_actual = [
                f"{inp['Temperature']:.1f}°C",
                f"{inp['Humidity']:.1f}%",
                f"{inp['Rainfall']:.0f}mm",
                f"{inp['NDVI']:.3f}",
                f"{inp['GDD']:.0f}",
            ]
            bar_colors_c = [RED if v>0.85 else (GOLD if v>0.6 else GREEN) for v in climate_norm]
            bars_c = ax_c.barh(climate_labels, climate_norm, color=bar_colors_c,
                                edgecolor="#ffffff", lw=0.5, height=0.55)
            for bar, lbl in zip(bars_c, climate_actual):
                ax_c.text(min(bar.get_width()+0.02, 1.02), bar.get_y()+bar.get_height()/2,
                          lbl, va="center", fontsize=8, fontweight="bold", color=T1)
            ax_c.set_xlim(0, 1.25); ax_c.axvline(1.0, color=BORDER2, ls="--", lw=1, alpha=0.5)
            ax_c.set_title("Kondisi Iklim (Ternormalisasi)", fontsize=10, fontweight="bold")
            ax_c.grid(axis="x", ls="--", alpha=0.3)
            plt.tight_layout(); rf(fig_c)

        vi3, vi4 = st.columns(2)

        with vi3:
            # Tekstur tanah — pie chart
            fig_t, ax_t = plt.subplots(figsize=(4, 3.5))
            total_tex = inp["Sand"] + inp["Silt"] + inp["Clay"]
            sizes_t   = [inp["Sand"], inp["Silt"], inp["Clay"]]
            labels_t  = [f"Sand\n{inp['Sand']:.1f}%", f"Silt\n{inp['Silt']:.1f}%", f"Clay\n{inp['Clay']:.1f}%"]
            if total_tex > 0:
                ax_t.pie(sizes_t, labels=labels_t, autopct="%1.0f%%",
                         colors=[GOLD, GREEN, CYAN], startangle=90,
                         wedgeprops={"edgecolor":"white","linewidth":1.5},
                         textprops={"fontsize":8, "color":T1})
            ax_t.set_title("Komposisi Tekstur Tanah", fontsize=10, fontweight="bold")
            plt.tight_layout(); rf(fig_t)

        with vi4:
            # Fitur turunan — bar horizontal
            fig_e, ax_e = plt.subplots(figsize=(5, 3.5))
            eng_names  = ["NPK Sum", "N/P ratio", "P/K ratio", "Temp×Rain", "NDVI×GDD", "pH×EC"]
            eng_values = [
                inp["N"]+inp["P"]+inp["K"],
                inp["N"]/(inp["P"]+1e-6),
                inp["P"]/(inp["K"]+1e-6),
                inp["Temperature"]*inp["Rainfall"]/1000,
                abs(inp["NDVI"]*inp["GDD"]),
                inp["pH"]*inp["EC"],
            ]
            # Normalize for display
            eng_max = [600, 10, 5, 15, 2500, 15]
            eng_norm = [min(v/m, 1.5) for v,m in zip(eng_values, eng_max)]
            bar_e = ax_e.barh(eng_names, eng_norm, color=CYAN, edgecolor="#ffffff",
                              lw=0.5, height=0.55, alpha=0.8)
            for bar, val in zip(bar_e, eng_values):
                ax_e.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                          f"{val:.2f}", va="center", fontsize=8, color=T2)
            ax_e.set_title("Fitur Engineered (Ternormalisasi)", fontsize=10, fontweight="bold")
            ax_e.set_xlim(0, 1.8); ax_e.grid(axis="x", ls="--", alpha=0.3)
            plt.tight_layout(); rf(fig_e)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state["page"] == "data":

    if not DATA_OK:
        st.error("Dataset `Agri_yield_prediction.csv` tidak ditemukan di folder yang sama.")
        st.stop()

    # Overview stats
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Baris",   f"{len(DF):,}")
    c2.metric("Total Kolom",   f"{DF.shape[1]}")
    c3.metric("Mean Yield",    f"{DF['Yield'].mean():.3f}")
    c4.metric("Std Yield",     f"{DF['Yield'].std():.3f}")
    c5.metric("Missing Value", "0")

    tab_dist, tab_cat, tab_corr, tab_balance = st.tabs([
        "📈  Distribusi Yield",
        "🌾  Per Kategori",
        "🔗  Korelasi",
        "⚖️  Sebaran Kelas",
    ])

    with tab_dist:
        c1, c2 = st.columns(2)
        bins5   = [0, 2, 4, 6, 8, 10]
        colors5 = [RED, ORANGE, GOLD, GREEN, CYAN]

        with c1:
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            for i in range(len(bins5)-1):
                mask = (DF["Yield"]>=bins5[i])&(DF["Yield"]<bins5[i+1])
                ax.hist(DF.loc[mask,"Yield"], bins=18, color=colors5[i],
                        alpha=0.82, edgecolor=BG, lw=0.3,
                        label=f"{bins5[i]}–{bins5[i+1]}")
            ax.set_xlabel("Yield (ton/ha)"); ax.set_ylabel("Frekuensi")
            ax.set_title("Distribusi Yield — 5 Kelas"); ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            rf(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(DF["Yield"])
            xr  = np.linspace(DF["Yield"].min(), DF["Yield"].max(), 300)
            ax.fill_between(xr, kde(xr), color=CYAN, alpha=0.15)
            ax.plot(xr, kde(xr), color=CYAN, lw=2.5)
            ax.axvline(DF["Yield"].mean(),   color=GOLD, lw=2, ls="--", label=f"Mean={DF['Yield'].mean():.3f}")
            ax.axvline(DF["Yield"].median(), color=GREEN,lw=2, ls=":",  label=f"Median={DF['Yield'].median():.3f}")
            ax.set_xlabel("Yield (ton/ha)"); ax.set_ylabel("Density")
            ax.set_title("KDE + Mean/Median Yield"); ax.legend()
            ax.grid(True, alpha=0.3)
            rf(fig)

        c1, c2 = st.columns(2)
        with c1:
            # Boxplot per season
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            seasons = sorted(DF["Season"].unique())
            grps = [DF[DF["Season"]==s]["Yield"].values for s in seasons]
            bp = ax.boxplot(grps, labels=seasons, patch_artist=True, notch=False,
                            widths=0.55)
            for patch, c_ in zip(bp["boxes"], [CYAN, GOLD, GREEN]):
                patch.set_facecolor(c_); patch.set_alpha(0.7)
            for med in bp["medians"]: med.set_color(T1); med.set_lw(1.8)
            for elem in ["whiskers","caps","fliers"]:
                for item in bp[elem]: item.set_color(T3)
            ax.set_ylabel("Yield (ton/ha)")
            ax.set_title("Distribusi Yield per Season")
            ax.grid(True, axis="y", alpha=0.3)
            rf(fig)

        with c2:
            # Violin per region
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            regions = sorted(DF["Region"].unique())
            data_r  = [DF[DF["Region"]==r]["Yield"].values for r in regions]
            vp = ax.violinplot(data_r, positions=range(len(regions)), showmedians=True)
            for i, (body, c_) in enumerate(zip(vp["bodies"], [CYAN, GOLD, GREEN, PURPLE])):
                body.set_facecolor(c_); body.set_alpha(0.65)
            vp["cmedians"].set_color(T1); vp["cmedians"].set_lw(2)
            vp["cbars"].set_color(T3); vp["cmins"].set_color(T3); vp["cmaxes"].set_color(T3)
            ax.set_xticks(range(len(regions))); ax.set_xticklabels(regions)
            ax.set_ylabel("Yield (ton/ha)")
            ax.set_title("Distribusi Yield per Region (Violin)")
            ax.grid(True, axis="y", alpha=0.3)
            rf(fig)

    with tab_cat:
        c1, c2 = st.columns(2)
        with c1:
            # Boxplot per crop
            fig, ax = plt.subplots(figsize=(5.5, 4))
            crops = sorted(DF["Crop_Type"].unique())
            grps  = [DF[DF["Crop_Type"]==c]["Yield"].values for c in crops]
            bp = ax.boxplot(grps, labels=crops, patch_artist=True, notch=False)
            pal = [CYAN, GOLD, GREEN, PURPLE]
            for patch, c_ in zip(bp["boxes"], pal):
                patch.set_facecolor(c_); patch.set_alpha(0.72)
            for med in bp["medians"]: med.set_color(T1); med.set_lw(2)
            for elem in ["whiskers","caps"]:
                for item in bp[elem]: item.set_color(BORDER2)
            ax.set_ylabel("Yield (ton/ha)")
            ax.set_title("Yield per Crop Type")
            ax.grid(True, axis="y", alpha=0.3)
            rf(fig)

        with c2:
            # Grouped bar — mean ± std
            fig, ax = plt.subplots(figsize=(5.5, 4))
            agg = DF.groupby("Crop_Type")["Yield"].agg(["mean","std"]).sort_values("mean")
            bars = ax.barh(agg.index, agg["mean"], xerr=agg["std"],
                           color=pal[:len(agg)], edgecolor=BG, lw=0.5,
                           capsize=4, error_kw={"color":T2,"lw":1.5}, alpha=0.85)
            for bar, v in zip(bars, agg["mean"]):
                ax.text(bar.get_width()+0.06, bar.get_y()+bar.get_height()/2,
                        f"{v:.3f}", va="center", fontsize=8, color=T2)
            ax.set_xlabel("Mean Yield (ton/ha)")
            ax.set_title("Rata-rata Yield per Crop Type (±Std)")
            ax.grid(True, axis="x", alpha=0.3)
            rf(fig)

        # Heatmap Crop × Season
        st.markdown(f"<div class='sec-hdr'>Heatmap Rata-rata Yield: Crop × Season</div>",
                    unsafe_allow_html=True)
        pivot = DF.groupby(["Crop_Type","Season"])["Yield"].mean().unstack()
        fig, ax = plt.subplots(figsize=(8, 3.5))
        sns.heatmap(pivot, ax=ax, cmap="YlOrBr",
                    annot=True, fmt=".2f", linewidths=0.8, linecolor=BG,
                    annot_kws={"size":10,"fontweight":"bold","color":"#0a1a06"},
                    cbar_kws={"shrink":0.7})
        ax.set_title("Mean Yield: Crop Type × Season", pad=8)
        for spine in ax.spines.values(): spine.set_visible(False)
        rf(fig)

    with tab_corr:
        num_df = DF.select_dtypes(include=[np.number]).drop(columns=["Year"], errors="ignore")
        corr   = num_df.corr()
        tc     = corr["Yield"].drop("Yield").sort_values(key=abs, ascending=False).head(15)

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5.5, 5.8))
            clrs = [GREEN if v>0 else RED for v in tc[::-1]]
            ax.barh(tc.index[::-1], tc.values[::-1], color=clrs, edgecolor=BG, lw=0.3)
            ax.axvline(0, color=BORDER2, lw=1)
            ax.set_xlabel("Pearson Correlation")
            ax.set_title("Top 15 Korelasi Fitur dengan Yield")
            ax.grid(True, axis="x", alpha=0.3)
            rf(fig)

        with c2:
            top10 = tc.abs().head(10).index.tolist() + ["Yield"]
            mini  = corr.loc[top10, top10]
            fig, ax = plt.subplots(figsize=(5.5, 5.8))
            sns.heatmap(mini, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                        annot=True, fmt=".2f", linewidths=0.8, linecolor=BG,
                        annot_kws={"size":8,"fontweight":"bold"},
                        cbar_kws={"shrink":0.8})
            ax.set_title("Heatmap Korelasi — Top 10 Fitur + Yield")
            for spine in ax.spines.values(): spine.set_visible(False)
            rf(fig)

    with tab_balance:
        labels_b = ["Sangat Rendah\n(0–2)","Rendah\n(2–4)","Sedang\n(4–6)",
                    "Tinggi\n(6–8)","Sangat Tinggi\n(8–10)"]
        before = [int(((DF["Yield"]>=bins5[i])&(DF["Yield"]<bins5[i+1])).sum())
                  for i in range(5)]
        target = int(np.mean(before))
        after  = [target]*5

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            x = np.arange(5); w = 0.35
            b1 = ax.bar(x-w/2, before, w, color=CARD2, edgecolor=BORDER2, lw=0.8, label="Sebelum")
            b2 = ax.bar(x+w/2, after,  w, color=CYAN,  edgecolor=BG,      lw=0.3, alpha=0.8, label="Setelah Balancing")
            ax.set_xticks(x); ax.set_xticklabels(["SR","R","S","T","ST"], fontsize=9)
            ax.set_ylabel("Count"); ax.set_title("Sebelum vs Sesudah Balancing")
            ax.legend(); ax.grid(True, axis="y", alpha=0.3)
            rf(fig)

        with c2:
            # Pie chart before
            fig, ax = plt.subplots(figsize=(5.5, 4))
            wedges, texts, autotexts = ax.pie(
                before, labels=["SR","R","S","T","ST"],
                autopct="%1.1f%%", startangle=90,
                colors=colors5, pctdistance=0.75,
                wedgeprops=dict(linewidth=2, edgecolor=BG))
            for at in autotexts:
                at.set_color(BG); at.set_fontsize(8.5); at.set_fontweight("bold")
            ax.set_title("Distribusi Kelas Yield (Original)")
            rf(fig)

        # Summary table
        comp_df = pd.DataFrame({
            "Kelas":    labels_b,
            "Sebelum":  before,
            "Sesudah":  after,
            "Selisih":  [a-b for a,b in zip(after,before)]
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.caption(f"Target per kelas setelah balancing: ~{target:,} sampel "
                   "(rata-rata dari 5 kelas). Metode: Random Over/Under Sampling "
                   "dengan Gaussian noise σ=0.01.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state["page"] == "model":

    # Table
    st.markdown(f"<div class='sec-hdr'>Perbandingan 5 Algoritma + Tuned</div>",
                unsafe_allow_html=True)

    def style_models(df):
        styled = df.style
        for col_s in ["Val_R2","Test_R2"]:
            styled = styled.background_gradient(subset=[col_s], cmap="RdYlGn", vmin=-0.08, vmax=0.0)
        for col_s in ["Val_RMSE","Test_RMSE","Test_MAE"]:
            styled = styled.background_gradient(subset=[col_s], cmap="RdYlGn_r", vmin=2.55, vmax=2.80)
        styled = styled.format({c:"{:.4f}" for c in ["Train_R2","Val_R2","Train_RMSE",
                                                       "Val_RMSE","Test_R2","Test_RMSE","Test_MAE"]})
        return styled

    st.dataframe(style_models(ALL_MODELS_DF), use_container_width=True,
                 hide_index=True, height=248)
    st.caption("★ = Model terbaik (Val R² tertinggi), sudah melalui hyperparameter tuning. "
               "Nilai R² negatif disebabkan distribusi Yield yang mendekati uniform random "
               "pada dataset synthetic ini.")

    # Charts
    c1, c2 = st.columns(2)
    models_short = ["RF★\n(Tuned)","RF","GB","XGB","DT","LGB"]

    with c1:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        rmse_v = ALL_MODELS_DF["Test_RMSE"].values
        clrs   = [CYAN if i==0 else CARD2 for i in range(len(rmse_v))]
        bars   = ax.bar(range(len(rmse_v)), rmse_v, color=clrs,
                        edgecolor=BORDER, lw=0.8, alpha=0.9)
        for bar, v in zip(bars, rmse_v):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                    f"{v:.3f}", ha="center", fontsize=8, color=T2)
        ax.set_xticks(range(len(rmse_v))); ax.set_xticklabels(models_short, fontsize=8)
        ax.set_ylabel("Test RMSE (ton/ha)")
        ax.set_title("Test RMSE — Semua Model (↓ lebih baik)")
        ax.set_ylim(2.5, 2.82)
        ax.axhline(rmse_v.min(), color=CYAN, lw=1, ls=":", alpha=0.6)
        ax.grid(True, axis="y", alpha=0.3)
        rf(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        r2_v = ALL_MODELS_DF["Test_R2"].values
        clrs = [CYAN if i==0 else CARD2 for i in range(len(r2_v))]
        bars = ax.bar(range(len(r2_v)), r2_v, color=clrs, edgecolor=BORDER, lw=0.8)
        for bar, v in zip(bars, r2_v):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height() - 0.005 if v < 0 else bar.get_height() + 0.001,
                    f"{v:.4f}", ha="center", va="top" if v<0 else "bottom",
                    fontsize=8, color=T2)
        ax.set_xticks(range(len(r2_v))); ax.set_xticklabels(models_short, fontsize=8)
        ax.axhline(0, color=RED, lw=1, ls="--", alpha=0.5, label="R²=0")
        ax.set_ylabel("Test R²"); ax.set_title("Test R² — Semua Model (↑ lebih baik)")
        ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
        rf(fig)

    # Feature Importance
    st.markdown(f"<div class='sec-hdr'>Feature Importance — {MODEL_NAME}</div>",
                unsafe_allow_html=True)
    if hasattr(MODEL, "feature_importances_"):
        imp = pd.DataFrame({"Feature":FEAT_COLS,
                             "Importance":MODEL.feature_importances_}
                           ).sort_values("Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(4, len(imp)*0.38)))
        q75  = imp["Importance"].quantile(0.75)
        clrs_imp = [CYAN if v>=q75 else BLUE if v>=imp["Importance"].quantile(0.5) else CARD2
                    for v in imp["Importance"]]
        bars = ax.barh(imp["Feature"], imp["Importance"], color=clrs_imp,
                       edgecolor=BG, lw=0.3)
        for bar, v in zip(bars, imp["Importance"]):
            ax.text(bar.get_width()+0.0005, bar.get_y()+bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=7.5, color=T2)
        ax.set_xlabel("Importance Score (MDI)")
        ax.set_title(f"Feature Importance — {MODEL_NAME} ({len(FEAT_COLS)} fitur)")
        ax.grid(True, axis="x", alpha=0.3)
        legend_e = [mpatches.Patch(color=CYAN, label="Top 25%"),
                    mpatches.Patch(color=BLUE,  label="Top 50%"),
                    mpatches.Patch(color=CARD2, label="Bottom 50%")]
        ax.legend(handles=legend_e, fontsize=8)
        rf(fig)
    else:
        st.info("Feature importance tidak tersedia untuk model ini.")

    # Tuning comparison
    st.markdown(f"<div class='sec-hdr'>Efek Hyperparameter Tuning</div>",
                unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    c1.metric("Val R² Sebelum Tuning",  "-0.0038")
    c1.metric("Val RMSE Sebelum Tuning","2.6108")
    c2.metric("Val R² Setelah Tuning",  "-0.0008", delta="+0.0030")
    c2.metric("Val RMSE Setelah Tuning","2.6069",  delta="-0.0039")
    with c3:
        st.markdown(f"""
        <div style='background:{CARD};border:1px solid {BORDER};border-radius:10px;
                    padding:16px;font-family:"JetBrains Mono",monospace;
                    font-size:11px;color:{T3};line-height:2'>
          <span style='color:{CYAN}'>TUNED PARAMS</span><br>
          n_estimators : <span style='color:{T1}'>300</span><br>
          max_depth &nbsp;&nbsp;: <span style='color:{T1}'>10</span><br>
          min_samples_leaf : <span style='color:{T1}'>3</span><br>
          Method &nbsp;&nbsp;&nbsp;&nbsp;: <span style='color:{T1}'>GridSearchCV</span><br>
          CV Folds &nbsp;&nbsp;: <span style='color:{T1}'>3-Fold KFold</span>
        </div>
        """, unsafe_allow_html=True)

    # Overfit chart
    st.markdown(f"<div class='sec-hdr'>Train R² vs Val R² — Overfitting Analysis</div>",
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    x = np.arange(len(ALL_MODELS_DF))
    w = 0.35
    ax.bar(x-w/2, ALL_MODELS_DF["Train_R2"], w, color=BLUE,  alpha=0.8, label="Train R²", edgecolor=BG)
    ax.bar(x+w/2, ALL_MODELS_DF["Val_R2"],   w, color=CYAN,  alpha=0.8, label="Val R²",   edgecolor=BG)
    ax.set_xticks(x); ax.set_xticklabels([m.replace(" ★","★") for m in ALL_MODELS_DF["Model"]], fontsize=8, rotation=10)
    ax.axhline(0, color=RED, lw=1, ls="--", alpha=0.4)
    ax.set_ylabel("R² Score"); ax.set_title("Train vs Val R² — Semua Model")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    rf(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state["page"] == "history":

    hist = st.session_state.get("history", [])

    if not hist:
        st.markdown(f"""
        <div style='background:{CARD};border:1px dashed {BORDER2};border-radius:16px;
                    padding:60px;text-align:center;color:{T3}'>
          <div style='font-size:40px;margin-bottom:14px'>📋</div>
          <div style='font-size:16px;font-weight:600;color:{T2};margin-bottom:6px'>
            Belum ada riwayat
          </div>
          <div style='font-size:13px;line-height:1.8'>
            Buka halaman <strong style='color:{CYAN}'>Prediksi Yield</strong> dan
            lakukan prediksi untuk mulai merekam riwayat.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        yields = [h["Yield (ton/ha)"] for h in hist]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Prediksi",  len(hist))
        c2.metric("Yield Tertinggi", f"{max(yields):.4f}")
        c3.metric("Yield Terendah",  f"{min(yields):.4f}")
        c4.metric("Rata-rata",       f"{np.mean(yields):.4f}")

        # Hitung berapa dari single vs batch
        n_single = sum(1 for h in hist if h.get("Sumber", "🎯 Single") == "🎯 Single")
        n_batch  = sum(1 for h in hist if h.get("Sumber", "") == "📦 Batch")
        c5, c6 = st.columns(2)
        c5.metric("🎯 Single Test", n_single)
        c6.metric("📦 Batch", n_batch)

        st.markdown(f"<div class='sec-hdr'>Tabel Riwayat</div>", unsafe_allow_html=True)
        hist_df = pd.DataFrame(hist)
        # pastikan kolom Sumber ada
        if "Sumber" not in hist_df.columns:
            hist_df["Sumber"] = "🎯 Single"
        hist_df.insert(0, "No", range(1, len(hist_df)+1))
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        if len(hist) > 1:
            st.markdown(f"<div class='sec-hdr'>Tren Yield Sesi Ini</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 3.2))
            xs = range(1, len(yields)+1)
            ax.plot(xs, yields[::-1], color=CYAN, lw=2.5, marker="o",
                    markersize=7, markerfacecolor=GOLD, markeredgecolor=CYAN,
                    markeredgewidth=1.5, zorder=5)
            ax.fill_between(xs, yields[::-1], alpha=0.1, color=CYAN)
            ax.axhline(np.mean(yields), color=GOLD, lw=1.5, ls="--",
                       label=f"Rata-rata: {np.mean(yields):.4f}")
            ax.set_xlabel("Prediksi ke-"); ax.set_ylabel("Yield (ton/ha)")
            ax.set_title("Tren Yield — Riwayat Prediksi")
            ax.legend(); ax.grid(True, alpha=0.3)
            rf(fig)

        c_dl, c_cl, _ = st.columns([2, 2, 4])
        with c_dl:
            csv = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️  Download CSV", csv,
                               "riwayat_prediksi.csv", "text/csv")
        with c_cl:
            if st.button("🗑  Hapus Riwayat", use_container_width=True):
                st.session_state["history"] = []
                if "pred" in st.session_state: del st.session_state["pred"]
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state["page"] == "preprocessing":

    if not DATA_OK:
        st.error("Dataset `Agri_yield_prediction.csv` tidak ditemukan.")
        st.stop()

    from scipy import stats as scipy_stats

    tabs = st.tabs([
        "📄  Data Original",
        "🔧  Feature Engineering",
        "⚖️  Sebelum Balancing",
        "✅  Setelah Preprocessing",
        "📐  Statistik Perbandingan",
    ])

    def sh(txt):
        st.markdown(f"<div class='sec-hdr'>{txt}</div>", unsafe_allow_html=True)

    bins5   = [0, 2, 4, 6, 8, 10]
    colors5 = [RED, ORANGE, GOLD, GREEN, CYAN]
    labels5 = ["Sangat Rendah (0–2)","Rendah (2–4)","Sedang (4–6)","Tinggi (6–8)","Sangat Tinggi (8–10)"]

    # ─── TAB 0: Data Original ────────────────────────────────────────────────
    with tabs[0]:
        sh("1.1  Shape & Tipe Data — SEBELUM Digunakan")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Baris",           f"{DF.shape[0]:,}")
        c2.metric("Total Kolom",           f"{DF.shape[1]}")
        c3.metric("Kolom Numerik",         f"{DF.select_dtypes(include='number').shape[1]}")
        c4.metric("Kolom Kategorikal",     f"{DF.select_dtypes(include='object').shape[1]}")
        c5.metric("Missing Values",        "0")

        sh("1.2  Tabel Atribut Dataset (Sebelum Digunakan)")
        deskripsi = {
            'Temperature':'Suhu udara rata-rata (°C)','Humidity':'Kelembapan udara relatif (%)',
            'Rainfall':'Curah hujan total (mm)','Soil_Type':'Jenis tanah — Sandy/Clayey/Loamy/Silty',
            'pH':'Keasaman tanah (skala 0–14)','EC':'Electrical Conductivity tanah (dS/m)',
            'OC':'Organic Carbon tanah (%)','N':'Nitrogen tersedia (kg/ha)',
            'P':'Fosfor tersedia (kg/ha)','K':'Kalium tersedia (kg/ha)',
            'Ca':'Kalsium (mg/kg)','Mg':'Magnesium (mg/kg)','S':'Sulfur (mg/kg)',
            'Zn':'Zinc (mg/kg)','Fe':'Besi (mg/kg)','Cu':'Tembaga (mg/kg)',
            'Mn':'Mangan (mg/kg)','B':'Boron (mg/kg)','Mo':'Molybdenum (mg/kg)',
            'CEC':'Cation Exchange Capacity (cmolc/kg)','Sand':'Persentase pasir (%)',
            'Silt':'Persentase lanau (%)','Clay':'Persentase liat (%)',
            'Bulk_Density':'Kerapatan tanah (g/cm³)','Water_Holding_Capacity':'Kapasitas simpan air (%)',
            'Slope':'Kemiringan lahan (°)','Aspect':'Arah hadap lahan (°)','Elevation':'Ketinggian mdpl (m)',
            'Solar_Radiation':'Radiasi matahari (MJ/m²/hari)','Wind_Speed':'Kecepatan angin (km/jam)',
            'NDVI':'Normalized Difference Vegetation Index','EVI':'Enhanced Vegetation Index',
            'LAI':'Leaf Area Index','Chlorophyll':'Kandungan klorofil (mg/m²)',
            'GDD':'Growing Degree Days (°C·hari)','Crop_Type':'Jenis tanaman — Maize/Rice/Soybean/Wheat',
            'Planting_Date':'Tanggal tanam (YYYY-MM-DD)','Harvest_Date':'Tanggal panen (YYYY-MM-DD)',
            'Growth_Stage':'Fase pertumbuhan — Vegetative/Reproductive/Maturity',
            'Irrigation_Frequency':'Frekuensi irigasi (kali/minggu)',
            'Fertilizer_Type':'Jenis pupuk — Organic/Chemical/Mixed',
            'Pesticide_Usage':'Tingkat pestisida — Low/Medium/High',
            'Yield':'TARGET: Hasil panen (ton/ha) — range 1–10',
            'Region':'Wilayah — North/South/East/West',
            'Season':'Musim tanam — Kharif/Rabi/Zaid','Year':'Tahun tanam (2000–2024)',
        }
        attr_rows = [{"No":i,"Nama Atribut":c,"Tipe Data":str(DF[c].dtype),
                      "Null":int(DF[c].isnull().sum()),"Unique":int(DF[c].nunique()),
                      "Contoh Nilai":str(DF[c].iloc[0]),"Keterangan":deskripsi.get(c,"—")}
                     for i,c in enumerate(DF.columns,1)]
        st.dataframe(pd.DataFrame(attr_rows), use_container_width=True, hide_index=True)

        sh("1.3  Statistik Deskriptif — Kolom Numerik (Data Original)")
        st.dataframe(DF.describe().T.round(3), use_container_width=True)

        sh("1.4  Sample Data Original (5 Baris Pertama)")
        st.dataframe(DF.head(5), use_container_width=True)

        sh("1.5  Kolom yang Dihapus — Alasan Seleksi")
        drop_reason_tbl = {
            'Planting_Date':'High-cardinality (10.000 nilai unik) → diganti Growing_Days & Planting_Month',
            'Harvest_Date': 'High-cardinality (10.000 nilai unik) → diganti Harvest_Month',
            'Year':         'Potensi data leakage; tidak mencerminkan kondisi agronomis langsung',
        }
        st.dataframe(pd.DataFrame([
            {"Nama Kolom":col,"Tipe":str(DF[col].dtype),"Unique":int(DF[col].nunique()),"Alasan":reason}
            for col,reason in drop_reason_tbl.items()
        ]), use_container_width=True, hide_index=True)

        sh("1.6  Deteksi Outlier per Kolom (IQR Method) — SEBELUM Cleaning")
        num_orig  = DF.select_dtypes(include=np.number).drop(columns=["Yield","Year"], errors="ignore")
        out_rows  = []
        for col_o in num_orig.columns:
            Q1,Q3 = num_orig[col_o].quantile(.25), num_orig[col_o].quantile(.75)
            IQR   = Q3-Q1
            n_out = ((num_orig[col_o]<Q1-1.5*IQR)|(num_orig[col_o]>Q3+1.5*IQR)).sum()
            if n_out>0:
                out_rows.append({"Fitur":col_o,"Q1":round(Q1,3),"Q3":round(Q3,3),
                                  "IQR":round(IQR,3),"N_Outlier":int(n_out),
                                  "Pct_%":round(n_out/len(DF)*100,2),"Tindakan":"Dipertahankan"})
        if out_rows:
            st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)
        else:
            st.success("Tidak ada outlier terdeteksi.")

    # ─── TAB 1: Feature Engineering ─────────────────────────────────────────
    with tabs[1]:
        sh("Kolom SEBELUM Feature Engineering")
        df_work = DF.copy()
        df_work["Planting_Date"] = pd.to_datetime(df_work["Planting_Date"])
        df_work["Harvest_Date"]  = pd.to_datetime(df_work["Harvest_Date"])
        df_work["Growing_Days"]   = (df_work["Harvest_Date"]-df_work["Planting_Date"]).dt.days
        df_work["Planting_Month"] = df_work["Planting_Date"].dt.month
        df_work["Harvest_Month"]  = df_work["Harvest_Date"].dt.month
        df_work.drop(columns=["Planting_Date","Harvest_Date","Year"], inplace=True)

        st.markdown(f"""
        <div style='background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:16px;
                    font-family:"JetBrains Mono",monospace;font-size:11px;color:{T3};line-height:2'>
          <span style='color:{CYAN};font-weight:700'>Sebelum FE:</span> {DF.shape[1]} kolom &nbsp;→&nbsp;
          <span style='color:{GOLD};font-weight:700'>Sesudah FE:</span> {df_work.shape[1]} kolom<br>
          Drop: Planting_Date, Harvest_Date, Year &nbsp;|&nbsp; Tambah: Growing_Days, Planting_Month, Harvest_Month
        </div>
        """, unsafe_allow_html=True)

        sh("Daftar Fitur Turunan (Engineered Features)")
        fe_table = pd.DataFrame([
            {"Nama Fitur":"Growing_Days",    "Formula":"Harvest_Date − Planting_Date","Tipe":"Durasi","Deskripsi":"Total hari masa tumbuh"},
            {"Nama Fitur":"Planting_Month",  "Formula":"Planting_Date.month",          "Tipe":"Kalender","Deskripsi":"Bulan penanaman (1–12)"},
            {"Nama Fitur":"Harvest_Month",   "Formula":"Harvest_Date.month",           "Tipe":"Kalender","Deskripsi":"Bulan panen (1–12)"},
            {"Nama Fitur":"NPK_Sum",         "Formula":"N + P + K",                    "Tipe":"Nutrisi","Deskripsi":"Total makronutrien NPK"},
            {"Nama Fitur":"N_P_ratio",       "Formula":"N / (P + ε)",                  "Tipe":"Nutrisi","Deskripsi":"Rasio N terhadap P"},
            {"Nama Fitur":"P_K_ratio",       "Formula":"P / (K + ε)",                  "Tipe":"Nutrisi","Deskripsi":"Rasio P terhadap K"},
            {"Nama Fitur":"Temp_Rain",       "Formula":"Temperature × Rainfall / 1000","Tipe":"Iklim","Deskripsi":"Interaksi suhu × curah hujan"},
            {"Nama Fitur":"NDVI_GDD",        "Formula":"NDVI × GDD",                   "Tipe":"Vegetasi","Deskripsi":"NDVI dikali Growing Degree Days"},
            {"Nama Fitur":"Soil_pH_EC",      "Formula":"pH × EC",                      "Tipe":"Tanah","Deskripsi":"Interaksi keasaman × konduktivitas"},
            {"Nama Fitur":"WHC_Bulk",        "Formula":"WHC / (Bulk_Density + ε)",     "Tipe":"Tanah","Deskripsi":"Rasio WHC terhadap kerapatan tanah"},
            {"Nama Fitur":"SandClay_ratio",  "Formula":"Sand / (Clay + ε)",            "Tipe":"Tekstur","Deskripsi":"Rasio pasir terhadap liat"},
        ])
        st.dataframe(fe_table, use_container_width=True, hide_index=True)

        sh("Sample Data SESUDAH Feature Engineering (5 baris pertama)")
        st.dataframe(df_work.head(5), use_container_width=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("Kolom Sebelum FE", f"{DF.shape[1]}")
        c2.metric("Kolom Sesudah FE", f"{df_work.shape[1]}")
        c3.metric("Fitur Baru",       "+11 fitur engineered")

        sh("Distribusi Fitur Turunan Utama")
        fe_show = ["Growing_Days","Planting_Month","Harvest_Month"]
        fig, axes = plt.subplots(1,3, figsize=(14,3.8))
        for ax_,col_fe in zip(axes, fe_show):
            ax_.hist(df_work[col_fe], bins=30, color=CYAN, edgecolor=BG, lw=0.3, alpha=0.85)
            ax_.set_title(col_fe, fontsize=10, pad=6); ax_.set_xlabel("Nilai"); ax_.set_ylabel("Frekuensi")
            ax_.grid(True, alpha=0.3)
        fig.suptitle("Distribusi Fitur Turunan dari Tanggal", fontsize=11, fontweight="bold")
        plt.tight_layout(); rf(fig)

    # ─── TAB 2: Sebelum Balancing ────────────────────────────────────────────
    with tabs[2]:
        before_count = [int(((DF["Yield"]>=bins5[i])&(DF["Yield"]<bins5[i+1])).sum()) for i in range(5)]
        target_bal   = int(np.mean(before_count))
        after_count  = [target_bal]*5

        sh("Distribusi Kelas Yield — SEBELUM Balancing")
        c1,c2,c3,c4,c5 = st.columns(5)
        for col_m,lbl,cnt in zip([c1,c2,c3,c4,c5],["SR","R","S","T","ST"],before_count):
            col_m.metric(lbl,f"{cnt:,}")

        c_a,c_b = st.columns(2)
        with c_a:
            fig,ax = plt.subplots(figsize=(5.5,4))
            bars = ax.bar(range(5), before_count, color=colors5, edgecolor=BG, lw=0.5, alpha=0.88)
            for bar,v in zip(bars,before_count):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+15, f"{v:,}",
                        ha="center", fontsize=8.5, color=T2, fontweight="bold")
            ax.set_xticks(range(5)); ax.set_xticklabels(["SR","R","S","T","ST"], fontsize=9)
            ax.set_ylabel("Count"); ax.set_title("Distribusi Kelas SEBELUM Balancing")
            ax.grid(True,axis="y",alpha=0.3); rf(fig)
        with c_b:
            fig,ax = plt.subplots(figsize=(5.5,4))
            wedges,_,autotexts = ax.pie(before_count, labels=["SR","R","S","T","ST"],
                autopct="%1.1f%%", startangle=90, colors=colors5, pctdistance=0.75,
                wedgeprops=dict(linewidth=2,edgecolor=BG))
            for at in autotexts: at.set_color(BG); at.set_fontsize(8.5); at.set_fontweight("bold")
            ax.set_title("Komposisi Kelas Yield (Before Balancing)"); rf(fig)

        sh("Perbandingan: SEBELUM vs SESUDAH Balancing")
        fig,ax = plt.subplots(figsize=(9,4))
        x=np.arange(5); w=0.35
        b1=ax.bar(x-w/2,before_count,w,color=[RED,ORANGE,GOLD,GREEN,CYAN],alpha=0.7,edgecolor=BG,label="Sebelum Balancing")
        b2=ax.bar(x+w/2,after_count, w,color=CYAN,alpha=0.9,edgecolor=BG,label=f"Setelah Balancing (target={target_bal:,})")
        for bar,v in zip(list(b1)+list(b2),before_count+after_count):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+15,f"{v:,}",ha="center",fontsize=7.5,color=T2)
        ax.set_xticks(x); ax.set_xticklabels(labels5,fontsize=8,rotation=10)
        ax.set_ylabel("Jumlah Sampel"); ax.set_title("Distribusi Kelas Yield: Sebelum vs Sesudah Balancing")
        ax.legend(); ax.grid(True,axis="y",alpha=0.3); ax.axhline(target_bal,color=GOLD,ls=":",lw=1.5,alpha=0.6)
        rf(fig)

        bal_comp = pd.DataFrame({"Kelas":labels5,"Sebelum":before_count,"Sesudah":after_count,
            "Selisih":[a-b for a,b in zip(after_count,before_count)],
            "Pct Perubahan":[f"{(a-b)/b*100:+.1f}%" for a,b in zip(after_count,before_count)]})
        st.dataframe(bal_comp, use_container_width=True, hide_index=True)
        st.caption(f"Metode Balancing: Random Over/Under Sampling + Gaussian Noise σ=0.01 · Target per kelas: ~{target_bal:,}")

        sh("Statistik Yield per Kelas — SEBELUM Balancing")
        stats_kelas = []
        for i in range(5):
            mask = (DF["Yield"]>=bins5[i])&(DF["Yield"]<bins5[i+1])
            g    = DF.loc[mask,"Yield"]
            stats_kelas.append({"Kelas":labels5[i],"Count":int(len(g)),
                "Min":round(g.min(),3),"Max":round(g.max(),3),"Mean":round(g.mean(),3),
                "Median":round(g.median(),3),"Std":round(g.std(),3),"Skewness":round(g.skew(),3)})
        st.dataframe(pd.DataFrame(stats_kelas), use_container_width=True, hide_index=True)

    # ─── TAB 3: Sesudah Preprocessing ────────────────────────────────────────
    with tabs[3]:
        sh("Tahapan Preprocessing yang Dilakukan")
        pipeline_steps = pd.DataFrame([
            {"Tahap":"1","Proses":"Data Cleaning","Detail":"Cek missing, duplikat, outlier IQR. Tidak ada yang dihapus.","Kolom":"Semua","Status":"✅"},
            {"Tahap":"2","Proses":"Drop Kolom","Detail":"Hapus Planting_Date, Harvest_Date, Year","Kolom":"3 kolom","Status":"✅"},
            {"Tahap":"3","Proses":"FE Tanggal","Detail":"Growing_Days, Planting_Month, Harvest_Month","Kolom":"+3","Status":"✅"},
            {"Tahap":"4","Proses":"FE Interaksi","Detail":"NPK_Sum, N_P_ratio, P_K_ratio, Temp_Rain, NDVI_GDD, Soil_pH_EC, WHC_Bulk, SandClay_ratio","Kolom":"+8","Status":"✅"},
            {"Tahap":"5","Proses":"Ordinal Encoding","Detail":"Growth_Stage (0-2), Pesticide_Usage (0-2)","Kolom":"2","Status":"✅"},
            {"Tahap":"6","Proses":"One-Hot Encoding","Detail":"Soil_Type, Crop_Type, Fertilizer_Type, Region, Season","Kolom":"5→dummies","Status":"✅"},
            {"Tahap":"7","Proses":"Feature Selection","Detail":"Konsensus 3 metode: Mutual Info + F-Reg + RF Importance","Kolom":f"{len(FEAT_COLS)} dipilih","Status":"✅"},
            {"Tahap":"8","Proses":"Train/Val/Test Split","Detail":"70/15/15 · random_state=42","Kolom":"7000/1500/1500","Status":"✅"},
            {"Tahap":"9","Proses":"RobustScaler","Detail":"fit pada X_train, transform X_val & X_test","Kolom":f"{len(FEAT_COLS)} fitur","Status":"✅"},
        ])
        st.dataframe(pipeline_steps, use_container_width=True, hide_index=True)

        sh(f"Fitur Terpilih — {len(FEAT_COLS)} Kolom (Setelah Feature Selection)")
        feat_df = pd.DataFrame({"No":range(1,len(FEAT_COLS)+1),"Nama Fitur":FEAT_COLS})
        n_each = len(FEAT_COLS)//3+1
        c_f1,c_f2,c_f3 = st.columns(3)
        for col_fi,chunk in zip([c_f1,c_f2,c_f3],
            [feat_df.iloc[:n_each],feat_df.iloc[n_each:2*n_each],feat_df.iloc[2*n_each:]]):
            col_fi.dataframe(chunk, use_container_width=True, hide_index=True)

        sh("Distribusi Fitur Utama — SEBELUM vs SESUDAH RobustScaler")
        key_feats_exist = [f for f in ["N","P","K","Temperature","Rainfall","NDVI"] if f in DF.columns]
        if key_feats_exist:
            from sklearn.preprocessing import RobustScaler as RS
            fig, axes = plt.subplots(2, len(key_feats_exist), figsize=(14,6))
            scaler_demo = RS()
            X_demo = DF[key_feats_exist].dropna()
            X_sc   = scaler_demo.fit_transform(X_demo)
            for j,feat in enumerate(key_feats_exist):
                axes[0,j].hist(X_demo[feat], bins=30, color=ORANGE, alpha=0.8, edgecolor=BG, lw=0.3)
                axes[0,j].set_title(f"{feat}\n(SEBELUM)", fontsize=8.5); axes[0,j].grid(True,alpha=0.3)
                axes[1,j].hist(X_sc[:,j], bins=30, color=CYAN, alpha=0.8, edgecolor=BG, lw=0.3)
                axes[1,j].set_title(f"{feat}\n(SESUDAH)", fontsize=8.5); axes[1,j].grid(True,alpha=0.3)
            axes[0,0].set_ylabel("Frekuensi"); axes[1,0].set_ylabel("Frekuensi")
            fig.suptitle("SEBELUM vs SESUDAH RobustScaler", fontsize=11, fontweight="bold")
            plt.tight_layout(); rf(fig)

        sh("Encoding Details")
        c_enc1,c_enc2 = st.columns(2)
        with c_enc1:
            st.markdown(f"<div style='color:{CYAN};font-weight:700;font-size:11px;margin-bottom:8px'>ORDINAL ENCODING</div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([
                {"Kolom":"Growth_Stage","Urutan":"Vegetative(0) → Reproductive(1) → Maturity(2)"},
                {"Kolom":"Pesticide_Usage","Urutan":"Low(0) → Medium(1) → High(2)"},
            ]), use_container_width=True, hide_index=True)
        with c_enc2:
            st.markdown(f"<div style='color:{GOLD};font-weight:700;font-size:11px;margin-bottom:8px'>ONE-HOT ENCODING</div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([
                {"Kolom":"Soil_Type","Kategori":"Sandy, Clayey, Loamy, Silty","Dummy":4},
                {"Kolom":"Crop_Type","Kategori":"Maize, Rice, Soybean, Wheat","Dummy":4},
                {"Kolom":"Fertilizer_Type","Kategori":"Organic, Chemical, Mixed","Dummy":3},
                {"Kolom":"Region","Kategori":"North, South, East, West","Dummy":4},
                {"Kolom":"Season","Kategori":"Kharif, Rabi, Zaid","Dummy":3},
            ]), use_container_width=True, hide_index=True)

        sh("Data Training — Preview Setelah Preprocessing + Scaling")
        st.markdown(f"""
        <div style='background:{CARD2};border:1px solid {BORDER};border-radius:8px;padding:12px;
                    font-size:11px;color:{T3};font-family:"JetBrains Mono",monospace'>
          X_train: 7.000 baris × {len(FEAT_COLS)} fitur (setelah RobustScaler)<br>
          y_train: 7.000 nilai Yield (ton/ha) — tidak discaling<br>
          Split: 70% train · 15% val · 15% test · random_state=42
        </div>
        """, unsafe_allow_html=True)
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import RobustScaler as RS
            df_sim     = DF.drop(columns=["Planting_Date","Harvest_Date","Year"], errors="ignore")
            df_sim_num = df_sim.select_dtypes(include=np.number)
            y_sim      = df_sim_num.pop("Yield")
            X_tr,_,y_tr,_ = train_test_split(df_sim_num, y_sim, test_size=0.3, random_state=42)
            sc_sim  = RS()
            X_tr_sc = pd.DataFrame(sc_sim.fit_transform(X_tr), columns=X_tr.columns)
            st.dataframe(X_tr_sc.head(5).round(4), use_container_width=True)
            c1_tr,c2_tr,c3_tr = st.columns(3)
            c1_tr.metric("X_train Shape",f"{X_tr_sc.shape[0]:,} × {X_tr_sc.shape[1]}")
            c2_tr.metric("y_train Mean", f"{y_tr.mean():.4f} ton/ha")
            c3_tr.metric("y_train Std",  f"{y_tr.std():.4f}")
        except Exception as e:
            st.warning(f"Preview tidak tersedia: {e}")

    # ─── TAB 4: Statistik Perbandingan ───────────────────────────────────────
    with tabs[4]:
        sh("Ringkasan Perubahan Data: Sebelum → Sesudah Preprocessing")
        st.dataframe(pd.DataFrame([
            {"Aspek":"Jumlah Baris",      "Sebelum":f"{DF.shape[0]:,}",                     "Sesudah":f"{DF.shape[0]:,} (tidak berubah)"},
            {"Aspek":"Jumlah Kolom",      "Sebelum":f"{DF.shape[1]}",                        "Sesudah":f"{len(FEAT_COLS)} (fitur terpilih)"},
            {"Aspek":"Kolom Numerik",     "Sebelum":f"{DF.select_dtypes(include='number').shape[1]}", "Sesudah":f"{len(FEAT_COLS)}"},
            {"Aspek":"Kolom Kategorikal", "Sebelum":f"{DF.select_dtypes(include='object').shape[1]}", "Sesudah":"0 (sudah di-encode)"},
            {"Aspek":"Kolom Tanggal",     "Sebelum":"2 (Planting, Harvest)",                 "Sesudah":"0 (dikonversi ke angka)"},
            {"Aspek":"Missing Values",    "Sebelum":"0",                                     "Sesudah":"0"},
            {"Aspek":"Outlier",           "Sebelum":"Terdeteksi",                            "Sesudah":"Dipertahankan (synthetic data)"},
            {"Aspek":"Skala Fitur",       "Sebelum":"Heterogen",                             "Sesudah":"RobustScaler"},
            {"Aspek":"Encoding",          "Sebelum":"Categorical string",                    "Sesudah":"Ordinal + One-Hot"},
            {"Aspek":"Fitur Baru",        "Sebelum":"—",                                     "Sesudah":"+11 fitur engineered"},
            {"Aspek":"Data Split",        "Sebelum":"—",                                     "Sesudah":"70/15/15 (train/val/test)"},
        ]), use_container_width=True, hide_index=True)

        sh("Statistik Yield — Data Original")
        y_stats = {"Min":DF["Yield"].min(),"Max":DF["Yield"].max(),"Mean":DF["Yield"].mean(),
                   "Median":DF["Yield"].median(),"Std":DF["Yield"].std(),
                   "Skewness":DF["Yield"].skew(),"Kurtosis":DF["Yield"].kurtosis(),
                   "Unique":int(DF["Yield"].nunique())}
        st.dataframe(pd.DataFrame({"Statistik":list(y_stats.keys()),
            "Nilai":[round(v,4) if isinstance(v,float) else v for v in y_stats.values()]}),
            use_container_width=True, hide_index=True)

        sh("Pemetaan CRISP-DM")
        st.dataframe(pd.DataFrame([
            {"Fase":"Business Understanding","Tahap":"Tahap 0","Status":"✅"},
            {"Fase":"Data Understanding",    "Tahap":"Tahap 1","Status":"✅"},
            {"Fase":"Data Preparation",      "Tahap":"Tahap 2–4","Status":"✅"},
            {"Fase":"Modeling",              "Tahap":"Tahap 5","Status":"✅"},
            {"Fase":"Evaluation",            "Tahap":"Tahap 5–6","Status":"✅"},
            {"Fase":"Deployment",            "Tahap":"Tahap 6 (App)","Status":"✅ Dashboard Ini"},
        ]), use_container_width=True, hide_index=True)

        sh("Metrik Target Bisnis vs Hasil Aktual")
        st.dataframe(pd.DataFrame([
            {"Metrik":"R²",  "Target":"≥ 0.70","Hasil":f"{META['test_r2']:.4f}","Status":"⚠️ Belum tercapai (data uniform/synthetic)"},
            {"Metrik":"RMSE","Target":"≤ 1.50","Hasil":f"{META['test_rmse']:.4f}","Status":"⚠️ Belum tercapai"},
            {"Metrik":"MAE", "Target":"≤ 1.20","Hasil":f"{META['test_mae']:.4f}","Status":"⚠️ Belum tercapai"},
        ]), use_container_width=True, hide_index=True)
        st.caption("Dataset bersifat synthetic dengan distribusi Yield mendekati uniform, sehingga R² negatif adalah wajar. Gunakan data lapangan nyata untuk produksi.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH TESTING
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state["page"] == "batch":

    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                  r2_score, mean_absolute_percentage_error)
    from scipy import stats as sp_stats

    st.markdown(f"""
    <div style='background:{CARD};border:1px solid {BORDER2};border-radius:12px;
                padding:20px 24px;margin-bottom:20px'>
      <div style='font-size:12px;font-weight:700;color:{CYAN};text-transform:uppercase;
                  letter-spacing:2px;margin-bottom:8px'>📂 Upload File CSV</div>
      <div style='font-size:12px;color:{T3};line-height:1.9'>
        Format CSV harus memiliki kolom input yang sama seperti form prediksi.<br>
        <span style='color:{GOLD}'>Opsional:</span> tambahkan kolom
        <code style='background:{CARD2};padding:1px 6px;border-radius:4px;color:{CYAN}'>Yield</code>
        untuk menghitung metrik evaluasi <em>(RMSE, R², MAE, MAPE, dll)</em>.<br>
        Jika kolom <code style='background:{CARD2};padding:1px 6px;border-radius:4px;color:{CYAN}'>Yield</code>
        tidak ada, hanya hasil prediksi yang ditampilkan.
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Pilih file CSV", type=["csv"],
        label_visibility="collapsed",
        help="Kolom opsional: Yield (ground truth untuk evaluasi)")
    use_demo = st.checkbox("Gunakan sampel data dari dataset asli sebagai demo",
                           value=(not bool(uploaded)) and DATA_OK)

    if use_demo and DATA_OK and not uploaded:
        np.random.seed(42)
        demo_idx = np.random.choice(len(DF), size=min(100,len(DF)), replace=False)
        batch_df = DF.iloc[demo_idx].copy()
        st.info(f"Demo: menggunakan {len(batch_df)} sampel acak dari dataset asli (termasuk Yield sebagai ground truth).")
    elif uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            st.success(f"File berhasil dimuat: {len(batch_df):,} baris × {batch_df.shape[1]} kolom")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}"); st.stop()
    else:
        st.markdown(f"""
        <div style='background:{CARD};border:1px dashed {BORDER2};border-radius:12px;
                    padding:40px;text-align:center;color:{T3}'>
          <div style='font-size:32px;margin-bottom:10px'>📁</div>
          <div style='font-size:14px;color:{T2}'>Upload CSV atau centang opsi demo</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    has_yield = "Yield" in batch_df.columns

    if st.button("🚀  Jalankan Batch Testing"):
        progress_bar = st.progress(0, text="Mempersiapkan…")
        preds=[]; errs_inf=[]; n=len(batch_df)
        for i,(_,row) in enumerate(batch_df.iterrows()):
            try:
                row_dict = row.to_dict(); row_dict.pop("Yield",None)
                if "Planting_Date" in row_dict and "Harvest_Date" in row_dict:
                    try:
                        p_=pd.to_datetime(row_dict["Planting_Date"]); h_=pd.to_datetime(row_dict["Harvest_Date"])
                        row_dict["Growing_Days"]=   (h_-p_).days
                        row_dict["Planting_Month"]= p_.month
                        row_dict["Harvest_Month"]=  h_.month
                    except: pass
                row_dict.pop("Planting_Date",None); row_dict.pop("Harvest_Date",None); row_dict.pop("Year",None)
                preds.append(run_inference(row_dict))
            except Exception as ex:
                preds.append(None); errs_inf.append((i,str(ex)))
            if i%10==0 or i==n-1:
                progress_bar.progress((i+1)/n, text=f"Baris {i+1}/{n}…")
        progress_bar.empty()
        st.session_state["batch_preds"]=preds; st.session_state["batch_df"]=batch_df
        st.session_state["batch_errors"]=errs_inf
        # ── Masukkan hasil batch ke riwayat ──
        if "history" not in st.session_state: st.session_state["history"] = []
        for i_h, (_, row_h) in enumerate(batch_df.iterrows()):
            if preds[i_h] is not None:
                pred_h = preds[i_h]
                lbl_h, _, emo_h, _ = yield_info(pred_h)
                st.session_state["history"].insert(0, {
                    "Crop":       row_h.get("Crop_Type", "—"),
                    "Region":     row_h.get("Region", "—"),
                    "Season":     row_h.get("Season", "—"),
                    "Fertilizer": row_h.get("Fertilizer_Type", "—"),
                    "Pesticide":  row_h.get("Pesticide_Usage", "—"),
                    "Yield (ton/ha)": pred_h,
                    "Kategori":   f"{emo_h} {lbl_h}",
                    "Sumber":     "📦 Batch",
                })
        st.success(f"Selesai! {n-len(errs_inf)}/{n} prediksi berhasil."); st.rerun()

    if "batch_preds" in st.session_state and st.session_state.get("batch_df") is not None:
        preds     = st.session_state["batch_preds"]
        b_df      = st.session_state["batch_df"].copy()
        errs_inf  = st.session_state["batch_errors"]
        has_yield = "Yield" in b_df.columns
        valid_mask= [p is not None for p in preds]
        preds_v   = [p for p in preds if p is not None]
        b_df_valid= b_df[valid_mask].copy()
        b_df_valid["Prediksi_Yield"] = preds_v

        st.markdown(f"<div class='sec-hdr'>Hasil Pengujian Batch</div>", unsafe_allow_html=True)

        if has_yield:
            y_true = b_df_valid["Yield"].values
            y_pred = b_df_valid["Prediksi_Yield"].values
            rmse   = float(np.sqrt(mean_squared_error(y_true,y_pred)))
            mae    = float(mean_absolute_error(y_true,y_pred))
            r2     = float(r2_score(y_true,y_pred))
            mape   = float(mean_absolute_percentage_error(y_true,y_pred))*100
            mse    = float(mean_squared_error(y_true,y_pred))
            resid  = y_true - y_pred

            st.markdown(f"""
            <div class='kpi-row'>
              <div class='kpi' style='--c:{CYAN}'>
                <div class='kpi-label'>R² Score</div>
                <div class='kpi-val'>{r2:.4f}</div>
                <div class='kpi-sub'>{"✅ Baik" if r2>=0.7 else "⚠️ Di bawah target"}</div>
              </div>
              <div class='kpi' style='--c:{RED}'>
                <div class='kpi-label'>RMSE</div>
                <div class='kpi-val'>{rmse:.4f}</div>
                <div class='kpi-sub'>ton/ha {"✅" if rmse<=1.5 else "⚠️"}</div>
              </div>
              <div class='kpi' style='--c:{ORANGE}'>
                <div class='kpi-label'>MAE</div>
                <div class='kpi-val'>{mae:.4f}</div>
                <div class='kpi-sub'>ton/ha {"✅" if mae<=1.2 else "⚠️"}</div>
              </div>
              <div class='kpi' style='--c:{GOLD}'>
                <div class='kpi-label'>MAPE</div>
                <div class='kpi-val'>{mape:.2f}%</div>
                <div class='kpi-sub'>Mean Abs % Error</div>
              </div>
              <div class='kpi' style='--c:{PURPLE}'>
                <div class='kpi-label'>MSE</div>
                <div class='kpi-val'>{mse:.4f}</div>
                <div class='kpi-sub'>ton²/ha²</div>
              </div>
              <div class='kpi' style='--c:{GREEN}'>
                <div class='kpi-label'>N Sampel</div>
                <div class='kpi-val'>{len(preds_v):,}</div>
                <div class='kpi-sub'>valid prediksi</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"<div class='sec-hdr'>Tabel Metrik Evaluasi Lengkap</div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame([
                {"Metrik":"R² Score",              "Nilai":f"{r2:.6f}",    "Interpretasi":"Proporsi variansi yang dijelaskan model","Target":"≥ 0.70","Status":"✅" if r2>=0.70 else "⚠️"},
                {"Metrik":"RMSE",                  "Nilai":f"{rmse:.6f}",  "Interpretasi":"Error rata-rata dalam ton/ha",             "Target":"≤ 1.50","Status":"✅" if rmse<=1.50 else "⚠️"},
                {"Metrik":"MAE",                   "Nilai":f"{mae:.6f}",   "Interpretasi":"Error absolut rata-rata",                 "Target":"≤ 1.20","Status":"✅" if mae<=1.20 else "⚠️"},
                {"Metrik":"MSE",                   "Nilai":f"{mse:.6f}",   "Interpretasi":"Rata-rata kuadrat error",                  "Target":"—","Status":"ℹ️"},
                {"Metrik":"MAPE (%)",               "Nilai":f"{mape:.4f}%", "Interpretasi":"Error rata-rata dalam persentase",         "Target":"—","Status":"ℹ️"},
                {"Metrik":"Max Error",             "Nilai":f"{float(np.max(np.abs(resid))):.4f}","Interpretasi":"Error terbesar",     "Target":"—","Status":"ℹ️"},
                {"Metrik":"Mean Residual",         "Nilai":f"{float(np.mean(resid)):.4f}", "Interpretasi":"Bias model (≈0 = tidak bias)","Target":"≈ 0","Status":"✅" if abs(float(np.mean(resid)))<0.5 else "⚠️"},
                {"Metrik":"Std Residual",          "Nilai":f"{float(np.std(resid)):.4f}",  "Interpretasi":"Penyebaran error prediksi", "Target":"—","Status":"ℹ️"},
            ]), use_container_width=True, hide_index=True)

            st.markdown(f"<div class='sec-hdr'>Visualisasi Evaluasi Batch</div>", unsafe_allow_html=True)
            col_v1,col_v2 = st.columns(2)
            with col_v1:
                fig,ax = plt.subplots(figsize=(5.5,4.5))
                sc = ax.scatter(y_true,y_pred,alpha=0.5,s=18,c=np.abs(resid),
                                cmap="RdYlGn_r",edgecolors="none",vmin=0,vmax=3)
                mn_,mx_ = min(y_true.min(),y_pred.min()), max(y_true.max(),y_pred.max())
                ax.plot([mn_,mx_],[mn_,mx_],color=CYAN,lw=2,ls="--",label="Perfect Fit")
                ax.plot([mn_,mx_],[mn_+1,mx_+1],color=ORANGE,lw=1,ls=":",alpha=0.5,label="±1 ton/ha")
                ax.plot([mn_,mx_],[mn_-1,mx_-1],color=ORANGE,lw=1,ls=":",alpha=0.5)
                plt.colorbar(sc,ax=ax,label="|Error| (ton/ha)").ax.yaxis.label.set_color(T2)
                ax.set_xlabel("Yield Aktual"); ax.set_ylabel("Yield Prediksi")
                ax.set_title("Actual vs Predicted — Batch Testing"); ax.legend(fontsize=7.5)
                ax.grid(True,alpha=0.3); rf(fig)
            with col_v2:
                fig,ax = plt.subplots(figsize=(5.5,4.5))
                ax.hist(resid,bins=40,color=PURPLE,edgecolor=BG,lw=0.3,alpha=0.85,density=True)
                from scipy.stats import gaussian_kde as gkde
                kde_r=gkde(resid); xr_=np.linspace(resid.min(),resid.max(),200)
                ax.plot(xr_,kde_r(xr_),color=CYAN,lw=2,label="KDE")
                ax.axvline(0,color=GREEN,lw=2.5,ls="--",label="Zero Error")
                ax.axvline(float(np.mean(resid)),color=GOLD,lw=1.5,ls=":",label=f"Mean={float(np.mean(resid)):.3f}")
                ax.axvline(1,color=RED,lw=1,ls=":",alpha=0.5,label="±1.0")
                ax.axvline(-1,color=RED,lw=1,ls=":",alpha=0.5)
                ax.set_xlabel("Residual"); ax.set_ylabel("Density")
                ax.set_title("Distribusi Residual (Batch Test)"); ax.legend(fontsize=7.5)
                ax.grid(True,alpha=0.3); rf(fig)

            col_v3,col_v4 = st.columns(2)
            bins5b=[0,2,4,6,8,10]
            with col_v3:
                fig,ax = plt.subplots(figsize=(5.5,4))
                b_df_valid["AbsError"]  = np.abs(resid)
                b_df_valid["KelasYield"]= pd.cut(b_df_valid["Yield"],bins=bins5b,labels=["SR","R","S","T","ST"])
                grp_err=[b_df_valid[b_df_valid["KelasYield"]==k]["AbsError"].dropna().values for k in ["SR","R","S","T","ST"]]
                bp2=ax.boxplot(grp_err,labels=["SR","R","S","T","ST"],patch_artist=True,widths=0.55)
                for patch,c_ in zip(bp2["boxes"],[RED,ORANGE,GOLD,GREEN,CYAN]):
                    patch.set_facecolor(c_); patch.set_alpha(0.7)
                for med in bp2["medians"]: med.set_color(T1); med.set_lw(2)
                ax.set_ylabel("|Error| (ton/ha)"); ax.set_xlabel("Kelas Yield")
                ax.set_title("Absolute Error per Kelas Yield"); ax.grid(True,axis="y",alpha=0.3); rf(fig)
            with col_v4:
                fig,ax = plt.subplots(figsize=(5.5,4))
                ax.scatter(y_pred,resid,alpha=0.4,s=14,color=CYAN,edgecolors="none")
                ax.axhline(0,color=RED,lw=2,ls="--",label="Zero")
                ax.axhline(1,color=ORANGE,lw=1,ls=":",alpha=0.6,label="±1.0")
                ax.axhline(-1,color=ORANGE,lw=1,ls=":",alpha=0.6)
                ax.set_xlabel("Yield Prediksi"); ax.set_ylabel("Residual")
                ax.set_title("Residual vs Predicted (Homoscedasticity)")
                ax.legend(fontsize=7.5); ax.grid(True,alpha=0.3); rf(fig)

            if "Crop_Type" in b_df_valid.columns:
                st.markdown(f"<div class='sec-hdr'>Error per Crop Type</div>", unsafe_allow_html=True)
                crop_metrics=[]
                for crop in b_df_valid["Crop_Type"].unique():
                    m_=b_df_valid[b_df_valid["Crop_Type"]==crop]
                    if len(m_)>1:
                        yt_=m_["Yield"].values; yp_=m_["Prediksi_Yield"].values
                        crop_metrics.append({"Crop":crop,"N":len(m_),
                            "RMSE":round(float(np.sqrt(mean_squared_error(yt_,yp_))),4),
                            "MAE":round(float(mean_absolute_error(yt_,yp_)),4),
                            "R²":round(float(r2_score(yt_,yp_)),4),
                            "Mean Error":round(float(np.mean(yt_-yp_)),4)})
                if crop_metrics:
                    st.dataframe(pd.DataFrame(crop_metrics), use_container_width=True, hide_index=True)

            st.markdown(f"<div class='sec-hdr'>Q-Q Plot Residual (Normalitas Error)</div>", unsafe_allow_html=True)
            fig,axes = plt.subplots(1,2,figsize=(12,4))
            sp_stats.probplot(resid,dist="norm",plot=axes[0])
            axes[0].set_title("Q-Q Plot Residual vs Normal")
            axes[0].get_lines()[0].set(color=CYAN,markersize=3,alpha=0.6)
            axes[0].get_lines()[1].set(color=RED,lw=2); axes[0].grid(True,alpha=0.3)
            z_resid=(resid-resid.mean())/resid.std()
            axes[1].hist(z_resid,bins=40,density=True,color=PURPLE,alpha=0.75,edgecolor=BG,lw=0.3)
            xn=np.linspace(-4,4,200)
            axes[1].plot(xn,sp_stats.norm.pdf(xn),color=CYAN,lw=2.5,label="N(0,1) ideal")
            axes[1].set_xlabel("Z-score"); axes[1].set_ylabel("Density")
            axes[1].set_title("Z-score Residual vs Normal Ideal"); axes[1].legend(); axes[1].grid(True,alpha=0.3)
            plt.tight_layout(); rf(fig)

            # ══════════════════════════════════════════════════════
            # VISUALISASI LENGKAP — seperti Analisis Data & Prediksi Yield
            # ══════════════════════════════════════════════════════
            st.markdown(f"<div class='sec-hdr'>📊 Analisis Dataset Batch</div>", unsafe_allow_html=True)

            # ── Tabel hasil + download (selalu muncul di atas visualisasi) ──
            st.markdown(f"<div class='sec-hdr'>Tabel Hasil Batch (10 Baris Pertama)</div>", unsafe_allow_html=True)
            show_cols = [c for c in ["Crop_Type","Region","Season","Fertilizer_Type","Pesticide_Usage","Yield"] if c in b_df_valid.columns]
            show_cols.append("Prediksi_Yield")
            if has_yield:
                b_df_valid["Error"]    = (b_df_valid["Yield"] - b_df_valid["Prediksi_Yield"]).round(4)
                b_df_valid["AbsError"] = b_df_valid["Error"].abs().round(4)
                show_cols.extend(["Error","AbsError"])
            st.dataframe(b_df_valid[show_cols].head(10), use_container_width=True, hide_index=True)

            csv_out = b_df_valid.to_csv(index=False).encode("utf-8")
            dl_col, _ = st.columns([2, 4])
            with dl_col:
                st.download_button("⬇️  Download Hasil Lengkap (CSV)", csv_out,
                                   "batch_hasil_prediksi.csv", "text/csv")

            st.divider()

            # ── TAB visualisasi batch ──────────────────────────────
            vt1, vt2, vt3, vt4, vt5 = st.tabs([
                "📈  Distribusi & Target",
                "🌾  Per Kategori",
                "🔗  Korelasi",
                "🌤  Iklim & Vegetasi",
                "🧪  Tanah & Nutrisi",
            ])

            # ── TAB 1: Distribusi & Target ──────────────────────────
            with vt1:
                bv1c1, bv1c2 = st.columns(2)
                with bv1c1:
                    # Distribusi Yield aktual vs prediksi overlay
                    fig, ax = plt.subplots(figsize=(5.5, 4))
                    ax.hist(y_true, bins=30, color=CYAN,  alpha=0.65, edgecolor=BG, lw=0.3, label="Aktual")
                    ax.hist(y_pred, bins=30, color=GOLD,  alpha=0.65, edgecolor=BG, lw=0.3, label="Prediksi")
                    ax.axvline(np.mean(y_true), color=CYAN, lw=2.2, ls="--", label=f"Mean Aktual={np.mean(y_true):.3f}")
                    ax.axvline(np.mean(y_pred), color=GOLD, lw=2.2, ls=":",  label=f"Mean Prediksi={np.mean(y_pred):.3f}")
                    ax.set_xlabel("Yield (ton/ha)"); ax.set_ylabel("Frekuensi")
                    ax.set_title("Distribusi: Yield Aktual vs Prediksi"); ax.legend(fontsize=7.5)
                    ax.grid(True, alpha=0.3); rf_tab(fig)

                with bv1c2:
                    # KDE Yield aktual
                    from scipy.stats import gaussian_kde as gkde2
                    fig, ax = plt.subplots(figsize=(5.5, 4))
                    kde_b = gkde2(y_true)
                    xr_b  = np.linspace(y_true.min(), y_true.max(), 300)
                    ax.fill_between(xr_b, kde_b(xr_b), color=CYAN, alpha=0.15)
                    ax.plot(xr_b, kde_b(xr_b), color=CYAN, lw=2.5)
                    ax.axvline(np.mean(y_true),   color=GOLD,  lw=2, ls="--", label=f"Mean={np.mean(y_true):.3f}")
                    ax.axvline(np.median(y_true),  color=GREEN, lw=2, ls=":",  label=f"Median={np.median(y_true):.3f}")
                    ax.set_xlabel("Yield (ton/ha)"); ax.set_ylabel("Density")
                    ax.set_title("KDE Yield Aktual (Batch)"); ax.legend()
                    ax.grid(True, alpha=0.3); rf_tab(fig)

                bv1c3, bv1c4 = st.columns(2)
                with bv1c3:
                    # Boxplot Yield aktual per kelas
                    fig, ax = plt.subplots(figsize=(5.5, 3.8))
                    bins5b  = [0, 2, 4, 6, 8, 10]
                    colors5b = [RED, ORANGE, GOLD, GREEN, CYAN]
                    lbls5b  = ["SR\n(0–2)", "R\n(2–4)", "S\n(4–6)", "T\n(6–8)", "ST\n(8–10)"]
                    grps_k  = []
                    for i in range(5):
                        mask = (y_true >= bins5b[i]) & (y_true < bins5b[i+1])
                        grps_k.append(y_true[mask] if mask.sum() > 0 else np.array([0]))
                    bp_k = ax.boxplot(grps_k, labels=lbls5b, patch_artist=True, widths=0.55)
                    for patch, c_ in zip(bp_k["boxes"], colors5b):
                        patch.set_facecolor(c_); patch.set_alpha(0.72)
                    for med in bp_k["medians"]: med.set_color(T1); med.set_lw(2)
                    ax.set_ylabel("Yield (ton/ha)"); ax.set_title("Boxplot Yield per Kelas")
                    ax.grid(True, axis="y", alpha=0.3); rf_tab(fig)

                with bv1c4:
                    # Q-Q plot Yield aktual
                    fig, ax = plt.subplots(figsize=(5.5, 3.8))
                    sp_stats.probplot(y_true, dist="norm", plot=ax)
                    ax.set_title("Q-Q Plot Yield Aktual (Batch)")
                    ax.get_lines()[0].set(color=CYAN, markersize=4, alpha=0.7)
                    ax.get_lines()[1].set(color=RED, lw=2)
                    ax.grid(True, alpha=0.3); rf_tab(fig)

                # Statistik Yield batch
                st.markdown(f"<div class='sec-hdr'>Statistik Deskriptif Yield — Batch</div>", unsafe_allow_html=True)
                stats_b = pd.DataFrame({
                    "Statistik": ["Min","Max","Mean","Median","Std Dev","Skewness","Kurtosis","N Sampel"],
                    "Yield Aktual": [
                        round(float(y_true.min()),4), round(float(y_true.max()),4),
                        round(float(y_true.mean()),4), round(float(np.median(y_true)),4),
                        round(float(y_true.std()),4),  round(float(sp_stats.skew(y_true)),4),
                        round(float(sp_stats.kurtosis(y_true)),4), len(y_true)
                    ],
                    "Yield Prediksi": [
                        round(float(y_pred.min()),4), round(float(y_pred.max()),4),
                        round(float(y_pred.mean()),4), round(float(np.median(y_pred)),4),
                        round(float(y_pred.std()),4),  round(float(sp_stats.skew(y_pred)),4),
                        round(float(sp_stats.kurtosis(y_pred)),4), len(y_pred)
                    ]
                })
                st.dataframe(stats_b.set_index("Statistik"), use_container_width=True)

            # ── TAB 2: Per Kategori ─────────────────────────────────
            with vt2:
                bv2c1, bv2c2 = st.columns(2)
                with bv2c1:
                    if "Crop_Type" in b_df_valid.columns:
                        fig, ax = plt.subplots(figsize=(5.5, 4))
                        crops_b = sorted(b_df_valid["Crop_Type"].unique())
                        pal4    = [CYAN, GOLD, GREEN, PURPLE]
                        grps_b  = [b_df_valid[b_df_valid["Crop_Type"]==c]["Yield"].values for c in crops_b]
                        bp_b = ax.boxplot(grps_b, labels=crops_b, patch_artist=True, widths=0.55)
                        for patch, c_ in zip(bp_b["boxes"], pal4):
                            patch.set_facecolor(c_); patch.set_alpha(0.72)
                        for med in bp_b["medians"]: med.set_color(T1); med.set_lw(2)
                        ax.set_ylabel("Yield (ton/ha)"); ax.set_title("Boxplot Yield per Crop Type")
                        ax.grid(True, axis="y", alpha=0.3); rf_tab(fig)

                with bv2c2:
                    if "Crop_Type" in b_df_valid.columns:
                        fig, ax = plt.subplots(figsize=(5.5, 4))
                        agg_b = b_df_valid.groupby("Crop_Type")["Yield"].agg(["mean","std"]).sort_values("mean")
                        bars_ag = ax.barh(agg_b.index, agg_b["mean"], xerr=agg_b["std"],
                                          color=pal4[:len(agg_b)], edgecolor=BG, lw=0.5,
                                          capsize=4, error_kw={"color":T2,"lw":1.5}, alpha=0.85)
                        for bar, v in zip(bars_ag, agg_b["mean"]):
                            ax.text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2,
                                    f"{v:.3f}", va="center", fontsize=8, color=T2)
                        ax.set_xlabel("Mean Yield (ton/ha)")
                        ax.set_title("Rata-rata Yield per Crop (±Std)"); ax.grid(True, axis="x", alpha=0.3); rf_tab(fig)

                bv2c3, bv2c4 = st.columns(2)
                with bv2c3:
                    if "Season" in b_df_valid.columns:
                        fig, ax = plt.subplots(figsize=(5.5, 3.8))
                        seasons_b = sorted(b_df_valid["Season"].unique())
                        grps_s = [b_df_valid[b_df_valid["Season"]==s]["Yield"].values for s in seasons_b]
                        bp_s = ax.boxplot(grps_s, labels=seasons_b, patch_artist=True, widths=0.55)
                        for patch, c_ in zip(bp_s["boxes"], [CYAN, GOLD, GREEN]):
                            patch.set_facecolor(c_); patch.set_alpha(0.7)
                        for med in bp_s["medians"]: med.set_color(T1); med.set_lw(2)
                        ax.set_ylabel("Yield (ton/ha)"); ax.set_title("Distribusi Yield per Season")
                        ax.grid(True, axis="y", alpha=0.3); rf_tab(fig)

                with bv2c4:
                    if "Region" in b_df_valid.columns:
                        fig, ax = plt.subplots(figsize=(5.5, 3.8))
                        regions_b = sorted(b_df_valid["Region"].unique())
                        data_rb   = [b_df_valid[b_df_valid["Region"]==r]["Yield"].values for r in regions_b]
                        vp_b = ax.violinplot(data_rb, positions=range(len(regions_b)), showmedians=True)
                        for i, (body, c_) in enumerate(zip(vp_b["bodies"], [CYAN, GOLD, GREEN, PURPLE])):
                            body.set_facecolor(c_); body.set_alpha(0.65)
                        vp_b["cmedians"].set_color(T1); vp_b["cmedians"].set_lw(2)
                        vp_b["cbars"].set_color(T3); vp_b["cmins"].set_color(T3); vp_b["cmaxes"].set_color(T3)
                        ax.set_xticks(range(len(regions_b))); ax.set_xticklabels(regions_b)
                        ax.set_ylabel("Yield (ton/ha)"); ax.set_title("Distribusi Yield per Region (Violin)")
                        ax.grid(True, axis="y", alpha=0.3); rf_tab(fig)

                # Heatmap Crop x Season
                if "Crop_Type" in b_df_valid.columns and "Season" in b_df_valid.columns:
                    st.markdown(f"<div class='sec-hdr'>Heatmap Mean Yield: Crop × Season</div>", unsafe_allow_html=True)
                    pivot_b = b_df_valid.groupby(["Crop_Type","Season"])["Yield"].mean().unstack()
                    if not pivot_b.empty and pivot_b.shape[1] > 0:
                        fig, ax = plt.subplots(figsize=(8, 3.5))
                        sns.heatmap(pivot_b, ax=ax, cmap="YlOrBr",
                                    annot=True, fmt=".2f", linewidths=0.8, linecolor=BG,
                                    annot_kws={"size":11,"fontweight":"bold","color":"#0a1a06"},
                                    cbar_kws={"shrink":0.7})
                        ax.set_title("Mean Yield Aktual: Crop Type × Season (Batch Data)", pad=8)
                        for spine in ax.spines.values(): spine.set_visible(False)
                        rf_tab(fig)

                # Pie distribusi Crop Type
                if "Crop_Type" in b_df_valid.columns:
                    st.markdown(f"<div class='sec-hdr'>Sebaran Crop Type & Fertilizer dalam Batch</div>", unsafe_allow_html=True)
                    bv2p1, bv2p2, bv2p3 = st.columns(3)
                    with bv2p1:
                        fig, ax = plt.subplots(figsize=(4.5, 4))
                        cnt_crop = b_df_valid["Crop_Type"].value_counts()
                        wedges, _, autotexts = ax.pie(cnt_crop.values, labels=cnt_crop.index,
                            autopct="%1.0f%%", startangle=90, colors=pal4[:len(cnt_crop)],
                            wedgeprops=dict(linewidth=2, edgecolor=BG), pctdistance=0.75)
                        for at in autotexts: at.set_color(BG); at.set_fontsize(9); at.set_fontweight("bold")
                        ax.set_title("Distribusi Crop Type"); rf_tab(fig)
                    with bv2p2:
                        if "Fertilizer_Type" in b_df_valid.columns:
                            fig, ax = plt.subplots(figsize=(4.5, 4))
                            cnt_fert = b_df_valid["Fertilizer_Type"].value_counts()
                            wedges2, _, autotexts2 = ax.pie(cnt_fert.values, labels=cnt_fert.index,
                                autopct="%1.0f%%", startangle=90, colors=[CYAN, GOLD, GREEN][:len(cnt_fert)],
                                wedgeprops=dict(linewidth=2, edgecolor=BG), pctdistance=0.75)
                            for at in autotexts2: at.set_color(BG); at.set_fontsize(9); at.set_fontweight("bold")
                            ax.set_title("Distribusi Fertilizer Type"); rf_tab(fig)
                    with bv2p3:
                        if "Pesticide_Usage" in b_df_valid.columns:
                            fig, ax = plt.subplots(figsize=(4.5, 4))
                            cnt_pest = b_df_valid["Pesticide_Usage"].value_counts()
                            wedges3, _, autotexts3 = ax.pie(cnt_pest.values, labels=cnt_pest.index,
                                autopct="%1.0f%%", startangle=90, colors=[RED, ORANGE, GOLD][:len(cnt_pest)],
                                wedgeprops=dict(linewidth=2, edgecolor=BG), pctdistance=0.75)
                            for at in autotexts3: at.set_color(BG); at.set_fontsize(9); at.set_fontweight("bold")
                            ax.set_title("Distribusi Pesticide Usage"); rf_tab(fig)

            # ── TAB 3: Korelasi ─────────────────────────────────────
            with vt3:
                num_batch = b_df_valid.select_dtypes(include=[np.number])
                if "Yield" in num_batch.columns and len(num_batch.columns) > 3:
                    corr_batch = num_batch.corr()
                    tc_b = corr_batch["Yield"].drop("Yield").sort_values(key=abs, ascending=False).head(15)

                    bv3c1, bv3c2 = st.columns(2)
                    with bv3c1:
                        fig, ax = plt.subplots(figsize=(5.5, 6))
                        clrs_b = [GREEN if v > 0 else RED for v in tc_b[::-1]]
                        ax.barh(tc_b.index[::-1], tc_b.values[::-1], color=clrs_b, edgecolor=BG, lw=0.3)
                        ax.axvline(0, color=BORDER2, lw=1)
                        ax.set_xlabel("Pearson Correlation")
                        ax.set_title("Top 15 Korelasi Fitur dengan Yield (Batch)")
                        ax.grid(True, axis="x", alpha=0.3); rf_tab(fig)

                    with bv3c2:
                        top10_b = tc_b.abs().head(10).index.tolist() + ["Yield"]
                        mini_b  = corr_batch.loc[top10_b, top10_b]
                        fig, ax = plt.subplots(figsize=(5.5, 6))
                        sns.heatmap(mini_b, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                                    annot=True, fmt=".2f", linewidths=0.8, linecolor=BG,
                                    annot_kws={"size":8,"fontweight":"bold"},
                                    cbar_kws={"shrink":0.8})
                        ax.set_title("Heatmap Korelasi — Top 10 Fitur + Yield")
                        for spine in ax.spines.values(): spine.set_visible(False)
                        rf_tab(fig)

                    # Scatter plots top 4 fitur vs Yield
                    st.markdown(f"<div class='sec-hdr'>Scatter Plot — Top 4 Fitur vs Yield</div>", unsafe_allow_html=True)
                    top4 = tc_b.abs().head(4).index.tolist()
                    bv3s1, bv3s2, bv3s3, bv3s4 = st.columns(4)
                    for col_sc, ax_col in zip(top4, [bv3s1, bv3s2, bv3s3, bv3s4]):
                        if col_sc in b_df_valid.columns:
                            with ax_col:
                                fig, ax = plt.subplots(figsize=(3.8, 3.5))
                                ax.scatter(b_df_valid[col_sc], b_df_valid["Yield"],
                                           alpha=0.6, s=20, color=CYAN, edgecolors="none")
                                # trendline
                                z = np.polyfit(b_df_valid[col_sc].dropna(),
                                               b_df_valid.loc[b_df_valid[col_sc].notna(), "Yield"], 1)
                                p = np.poly1d(z)
                                xsc = np.linspace(b_df_valid[col_sc].min(), b_df_valid[col_sc].max(), 100)
                                ax.plot(xsc, p(xsc), color=RED, lw=1.8, ls="--", alpha=0.8)
                                r_val = corr_batch.loc[col_sc, "Yield"]
                                ax.set_title(f"{col_sc}\nr={r_val:.3f}", fontsize=8.5)
                                ax.set_xlabel(col_sc, fontsize=7.5); ax.set_ylabel("Yield", fontsize=7.5)
                                ax.grid(True, alpha=0.3); rf_tab(fig)

            # ── TAB 4: Iklim & Vegetasi ─────────────────────────────
            with vt4:
                bv4c1, bv4c2 = st.columns(2)
                # Bar chart kondisi iklim rata-rata (ternormalisasi) — mirip single test
                with bv4c1:
                    clim_cols = [c for c in ["Temperature","Humidity","Rainfall","NDVI","GDD","Solar_Radiation","Wind_Speed"] if c in b_df_valid.columns]
                    if clim_cols:
                        clim_means = [b_df_valid[c].mean() for c in clim_cols]
                        clim_maxes = {"Temperature":45,"Humidity":100,"Rainfall":350,
                                      "NDVI":1,"GDD":3500,"Solar_Radiation":3000,"Wind_Speed":40}
                        clim_norm  = [min(v/clim_maxes.get(c,1),1) for v,c in zip(clim_means, clim_cols)]
                        clim_labels_short = {"Temperature":"Suhu","Humidity":"Kelembapan",
                                             "Rainfall":"Curah Hujan","NDVI":"NDVI",
                                             "GDD":"GDD","Solar_Radiation":"Solar Rad","Wind_Speed":"Angin"}
                        bar_colors_clim = [RED if v>0.85 else (GOLD if v>0.6 else GREEN) for v in clim_norm]
                        fig, ax = plt.subplots(figsize=(5.5, 4))
                        bars_cl = ax.barh([clim_labels_short.get(c,c) for c in clim_cols],
                                           clim_norm, color=bar_colors_clim, edgecolor=BG, lw=0.5, height=0.55)
                        for bar, mean_v, c_ in zip(bars_cl, clim_means, clim_cols):
                            unit = {"Temperature":"°C","Humidity":"%","Rainfall":"mm","Wind_Speed":"km/h"}.get(c_,"")
                            ax.text(min(bar.get_width()+0.02,1.02), bar.get_y()+bar.get_height()/2,
                                    f"{mean_v:.1f}{unit}", va="center", fontsize=8, fontweight="bold", color=T1)
                        ax.set_xlim(0, 1.3); ax.axvline(1.0, color=BORDER2, ls="--", lw=1, alpha=0.5)
                        ax.set_title("Rata-rata Kondisi Iklim (Ternormalisasi) — Batch")
                        ax.grid(axis="x", ls="--", alpha=0.3); rf_tab(fig)

                with bv4c2:
                    # Indeks vegetasi scatter NDVI vs EVI
                    veg_cols = [c for c in ["NDVI","EVI","LAI","Chlorophyll"] if c in b_df_valid.columns]
                    if len(veg_cols) >= 2:
                        fig, ax = plt.subplots(figsize=(5.5, 4))
                        sc_veg = ax.scatter(b_df_valid[veg_cols[0]], b_df_valid[veg_cols[1]],
                                            c=b_df_valid["Yield"], cmap="YlGn",
                                            alpha=0.75, s=35, edgecolors="none", vmin=0, vmax=10)
                        plt.colorbar(sc_veg, ax=ax, label="Yield (ton/ha)").ax.yaxis.label.set_color(T2)
                        ax.set_xlabel(veg_cols[0]); ax.set_ylabel(veg_cols[1])
                        ax.set_title(f"{veg_cols[0]} vs {veg_cols[1]} — warna = Yield")
                        ax.grid(True, alpha=0.3); rf_tab(fig)

                bv4c3, bv4c4 = st.columns(2)
                with bv4c3:
                    # Scatter Temperature vs Rainfall berwarna Yield
                    if "Temperature" in b_df_valid.columns and "Rainfall" in b_df_valid.columns:
                        fig, ax = plt.subplots(figsize=(5.5, 4))
                        sc_tr = ax.scatter(b_df_valid["Temperature"], b_df_valid["Rainfall"],
                                           c=b_df_valid["Yield"], cmap="RdYlGn",
                                           alpha=0.8, s=40, edgecolors="none", vmin=0, vmax=10)
                        plt.colorbar(sc_tr, ax=ax, label="Yield").ax.yaxis.label.set_color(T2)
                        ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Rainfall (mm)")
                        ax.set_title("Temperature vs Rainfall — warna = Yield")
                        ax.grid(True, alpha=0.3); rf_tab(fig)

                with bv4c4:
                    # Bar GDD rata-rata per Crop Type
                    if "GDD" in b_df_valid.columns and "Crop_Type" in b_df_valid.columns:
                        fig, ax = plt.subplots(figsize=(5.5, 4))
                        gdd_agg = b_df_valid.groupby("Crop_Type")["GDD"].mean().sort_values()
                        bars_gdd = ax.bar(gdd_agg.index, gdd_agg.values,
                                          color=[CYAN,GOLD,GREEN,PURPLE][:len(gdd_agg)],
                                          edgecolor=BG, lw=0.5, alpha=0.85)
                        for bar, v in zip(bars_gdd, gdd_agg.values):
                            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                                    f"{v:.0f}", ha="center", fontsize=8.5, fontweight="bold", color=T2)
                        ax.set_ylabel("GDD (°C·day)"); ax.set_title("Rata-rata GDD per Crop Type")
                        ax.grid(axis="y", ls="--", alpha=0.4); rf_tab(fig)

                # Radar chart rata-rata parameter iklim
                st.markdown(f"<div class='sec-hdr'>Radar Chart — Profil Iklim Rata-rata Batch</div>", unsafe_allow_html=True)
                radar_cols = {
                    "Suhu":        ("Temperature", 10, 45),
                    "Kelembapan":  ("Humidity",    0,  100),
                    "Curah Hujan": ("Rainfall",    0,  350),
                    "NDVI":        ("NDVI",        -1, 1),
                    "NPK":         (None,          0,  600),
                    "GDD":         ("GDD",         0,  3500),
                    "pH":          ("pH",          4,  10),
                }
                cat_r = list(radar_cols.keys())
                norm_r = []
                for cat, (col, lo, hi) in radar_cols.items():
                    if col and col in b_df_valid.columns:
                        v = b_df_valid[col].mean()
                    elif cat == "NPK":
                        npk_cols = [c for c in ["N","P","K"] if c in b_df_valid.columns]
                        v = sum(b_df_valid[c].mean() for c in npk_cols) if npk_cols else 0
                        lo, hi = 0, 600
                    else:
                        v = 0; lo, hi = 0, 1
                    norm_r.append(max(min((v - lo) / (hi - lo + 1e-6), 1), 0))

                N_r2 = len(cat_r)
                angles2 = [n / N_r2 * 2 * np.pi for n in range(N_r2)]
                vals_r2 = norm_r + [norm_r[0]]
                angs_r2 = angles2 + [angles2[0]]

                rc1, rc2, rc3 = st.columns([1, 2, 1])
                with rc2:
                    fig_r2, ax_r2 = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
                    fig_r2.patch.set_facecolor(BG)
                    ax_r2.set_facecolor(CARD)
                    ax_r2.plot(angs_r2, vals_r2, color=CYAN, lw=2.5, zorder=5)
                    ax_r2.fill(angs_r2, vals_r2, color=CYAN, alpha=0.2)
                    for angle, val in zip(angles2, norm_r):
                        ax_r2.plot([angle, angle], [0, val], color=CYAN, lw=1, alpha=0.4)
                        ax_r2.scatter([angle], [val], s=50, color=CYAN, zorder=6, edgecolors=BG, lw=1.5)
                    ax_r2.set_xticks(angles2); ax_r2.set_xticklabels(cat_r, color=T2, fontsize=9)
                    ax_r2.set_yticks([0.25, 0.5, 0.75, 1.0])
                    ax_r2.set_yticklabels(["25%","50%","75%","100%"], color=T3, fontsize=7)
                    ax_r2.grid(color=BORDER, linewidth=0.8); ax_r2.spines["polar"].set_color(BORDER)
                    ax_r2.set_title("Profil Iklim Rata-rata\n(Ternormalisasi)", color=T2, fontsize=10, pad=18)
                    rf_tab(fig_r2)

            # ── TAB 5: Tanah & Nutrisi ──────────────────────────────
            with vt5:
                bv5c1, bv5c2 = st.columns(2)
                with bv5c1:
                    # Bar chart nutrisi makro rata-rata
                    nut_cols = [c for c in ["N","P","K","Ca","Mg","S"] if c in b_df_valid.columns]
                    if nut_cols:
                        fig, ax = plt.subplots(figsize=(5.5, 4))
                        nut_means_b = [b_df_valid[c].mean() for c in nut_cols]
                        nut_colors  = [CYAN, GREEN, GOLD, ORANGE, PURPLE, RED][:len(nut_cols)]
                        bars_nb = ax.bar(nut_cols, nut_means_b, color=nut_colors, edgecolor=BG, lw=0.5, width=0.6)
                        for bar, v in zip(bars_nb, nut_means_b):
                            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                                    f"{v:.1f}", ha="center", fontsize=8, fontweight="bold", color=T2)
                        ax.set_title("Rata-rata Nutrisi Makro (mg/kg) — Batch")
                        ax.set_ylabel("mg/kg"); ax.grid(axis="y", ls="--", alpha=0.4); rf_tab(fig)

                with bv5c2:
                    # Pie tekstur tanah rata-rata
                    tex_cols = [c for c in ["Sand","Silt","Clay"] if c in b_df_valid.columns]
                    if len(tex_cols) == 3:
                        means_tex = [b_df_valid[c].mean() for c in tex_cols]
                        labels_tex = [f"{c}\n{v:.1f}%" for c, v in zip(tex_cols, means_tex)]
                        fig, ax = plt.subplots(figsize=(4.5, 4))
                        ax.pie(means_tex, labels=labels_tex, autopct="%1.0f%%",
                               colors=[GOLD, GREEN, CYAN], startangle=90,
                               wedgeprops={"edgecolor":"white","linewidth":1.5},
                               textprops={"fontsize":8, "color":T1})
                        ax.set_title("Rata-rata Komposisi Tekstur Tanah (Batch)"); rf_tab(fig)

                bv5c3, bv5c4 = st.columns(2)
                with bv5c3:
                    # Scatter pH vs EC berwarna Yield
                    if "pH" in b_df_valid.columns and "EC" in b_df_valid.columns:
                        fig, ax = plt.subplots(figsize=(5.5, 4))
                        sc_ph = ax.scatter(b_df_valid["pH"], b_df_valid["EC"],
                                           c=b_df_valid["Yield"], cmap="RdYlGn",
                                           alpha=0.8, s=40, edgecolors="none", vmin=0, vmax=10)
                        plt.colorbar(sc_ph, ax=ax, label="Yield").ax.yaxis.label.set_color(T2)
                        ax.set_xlabel("pH Tanah"); ax.set_ylabel("EC (dS/m)")
                        ax.set_title("pH vs EC — warna = Yield")
                        ax.grid(True, alpha=0.3); rf_tab(fig)

                with bv5c4:
                    # Fitur turunan rata-rata (engineered features) — bar horizontal
                    eng_feats_b = {}
                    if all(c in b_df_valid.columns for c in ["N","P","K"]):
                        eng_feats_b["NPK Sum"]   = (b_df_valid["N"]+b_df_valid["P"]+b_df_valid["K"]).mean()
                    if all(c in b_df_valid.columns for c in ["N","P"]):
                        eng_feats_b["N/P ratio"]  = (b_df_valid["N"]/(b_df_valid["P"]+1e-6)).mean()
                    if all(c in b_df_valid.columns for c in ["P","K"]):
                        eng_feats_b["P/K ratio"]  = (b_df_valid["P"]/(b_df_valid["K"]+1e-6)).mean()
                    if all(c in b_df_valid.columns for c in ["Temperature","Rainfall"]):
                        eng_feats_b["Temp×Rain"]  = (b_df_valid["Temperature"]*b_df_valid["Rainfall"]/1000).mean()
                    if all(c in b_df_valid.columns for c in ["NDVI","GDD"]):
                        eng_feats_b["NDVI×GDD"]   = (b_df_valid["NDVI"]*b_df_valid["GDD"]).abs().mean()
                    if all(c in b_df_valid.columns for c in ["pH","EC"]):
                        eng_feats_b["pH×EC"]      = (b_df_valid["pH"]*b_df_valid["EC"]).mean()
                    if all(c in b_df_valid.columns for c in ["Water_Holding_Capacity","Bulk_Density"]):
                        eng_feats_b["WHC/Bulk"]   = (b_df_valid["Water_Holding_Capacity"]/(b_df_valid["Bulk_Density"]+1e-6)).mean()
                    if all(c in b_df_valid.columns for c in ["Sand","Clay"]):
                        eng_feats_b["Sand/Clay"]  = (b_df_valid["Sand"]/(b_df_valid["Clay"]+1e-6)).mean()

                    if eng_feats_b:
                        eng_max_b  = [600,10,5,15,2500,15,40,10]
                        fig, ax = plt.subplots(figsize=(5.5, 4))
                        keys_e = list(eng_feats_b.keys()); vals_e = list(eng_feats_b.values())
                        max_e  = [600,10,5,15,2500,15,40,10][:len(keys_e)]
                        norm_e = [min(v/m,1.5) for v,m in zip(vals_e, max_e)]
                        bar_e  = ax.barh(keys_e, norm_e, color=CYAN, edgecolor=BG, lw=0.5, height=0.55, alpha=0.85)
                        for bar, val in zip(bar_e, vals_e):
                            ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                                    f"{val:.2f}", va="center", fontsize=8, color=T2)
                        ax.set_title("Rata-rata Fitur Engineered (Batch)")
                        ax.set_xlim(0, 1.8); ax.grid(axis="x", ls="--", alpha=0.3); rf_tab(fig)

                # Mikronutrien bar
                st.markdown(f"<div class='sec-hdr'>Profil Mikronutrien Rata-rata (Batch)</div>", unsafe_allow_html=True)
                micro_cols = [c for c in ["Zn","Fe","Cu","Mn","B","Mo"] if c in b_df_valid.columns]
                if micro_cols:
                    fig, ax = plt.subplots(figsize=(10, 3.5))
                    micro_means = [b_df_valid[c].mean() for c in micro_cols]
                    micro_colors = [CYAN, GOLD, GREEN, ORANGE, PURPLE, RED][:len(micro_cols)]
                    bars_mc = ax.bar(micro_cols, micro_means, color=micro_colors, edgecolor=BG, lw=0.5, width=0.55)
                    for bar, v in zip(bars_mc, micro_means):
                        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                                f"{v:.2f}", ha="center", fontsize=9, fontweight="bold", color=T2)
                    ax.set_title("Rata-rata Mikronutrien (mg/kg) — Batch")
                    ax.set_ylabel("mg/kg"); ax.grid(axis="y", ls="--", alpha=0.4); rf_tab(fig)

                # Distribusi Bulk Density per Soil Type
                if "Bulk_Density" in b_df_valid.columns and "Soil_Type" in b_df_valid.columns:
                    st.markdown(f"<div class='sec-hdr'>Bulk Density & Water Holding Capacity per Soil Type</div>", unsafe_allow_html=True)
                    bv5s1, bv5s2 = st.columns(2)
                    with bv5s1:
                        fig, ax = plt.subplots(figsize=(5.5, 3.5))
                        soils_b = sorted(b_df_valid["Soil_Type"].unique())
                        bd_grps = [b_df_valid[b_df_valid["Soil_Type"]==s]["Bulk_Density"].values for s in soils_b]
                        bp_bd = ax.boxplot(bd_grps, labels=soils_b, patch_artist=True, widths=0.5)
                        for patch, c_ in zip(bp_bd["boxes"], [CYAN,GOLD,GREEN,PURPLE][:len(soils_b)]):
                            patch.set_facecolor(c_); patch.set_alpha(0.7)
                        for med in bp_bd["medians"]: med.set_color(T1); med.set_lw(2)
                        ax.set_ylabel("Bulk Density (g/cm³)"); ax.set_title("Bulk Density per Soil Type")
                        ax.grid(True, axis="y", alpha=0.3); rf_tab(fig)
                    with bv5s2:
                        if "Water_Holding_Capacity" in b_df_valid.columns:
                            fig, ax = plt.subplots(figsize=(5.5, 3.5))
                            whc_grps = [b_df_valid[b_df_valid["Soil_Type"]==s]["Water_Holding_Capacity"].values for s in soils_b]
                            bp_whc = ax.boxplot(whc_grps, labels=soils_b, patch_artist=True, widths=0.5)
                            for patch, c_ in zip(bp_whc["boxes"], [CYAN,GOLD,GREEN,PURPLE][:len(soils_b)]):
                                patch.set_facecolor(c_); patch.set_alpha(0.7)
                            for med in bp_whc["medians"]: med.set_color(T1); med.set_lw(2)
                            ax.set_ylabel("WHC (%)"); ax.set_title("Water Holding Capacity per Soil Type")
                            ax.grid(True, axis="y", alpha=0.3); rf_tab(fig)

        else:
            c1_,c2_,c3_,c4_ = st.columns(4)
            c1_.metric("Total Diprediksi",f"{len(preds_v):,}")
            c2_.metric("Yield Rata-rata", f"{np.mean(preds_v):.4f} ton/ha")
            c3_.metric("Yield Tertinggi", f"{max(preds_v):.4f} ton/ha")
            c4_.metric("Yield Terendah",  f"{min(preds_v):.4f} ton/ha")
            st.info("Tambahkan kolom **Yield** pada CSV untuk menghitung RMSE, R², MAE, dll.")
            fig,ax = plt.subplots(figsize=(9,4))
            ax.hist(preds_v,bins=40,color=CYAN,edgecolor=BG,lw=0.3,alpha=0.85)
            ax.axvline(np.mean(preds_v),color=GOLD,lw=2,ls="--",label=f"Mean={np.mean(preds_v):.4f}")
            ax.set_xlabel("Prediksi Yield (ton/ha)"); ax.set_ylabel("Frekuensi")
            ax.set_title("Distribusi Hasil Prediksi Batch"); ax.legend(); ax.grid(True,alpha=0.3); rf_tab(fig)

        if errs_inf:
            st.warning(f"⚠️ {len(errs_inf)} baris gagal diproses.")
            with st.expander("Detail Error"):
                for row_idx,err_msg in errs_inf[:10]:
                    st.markdown(f"Baris {row_idx}: `{err_msg}`")

        st.markdown(f"<div class='sec-hdr'>Template CSV untuk Upload</div>", unsafe_allow_html=True)
        tpl_df = pd.DataFrame([{
            "Temperature":26.5,"Humidity":65.0,"Rainfall":180.0,"Soil_Type":"Loamy",
            "pH":6.8,"EC":1.2,"OC":1.0,"N":120.0,"P":80.0,"K":160.0,
            "Ca":800.0,"Mg":300.0,"S":30.0,"Zn":2.0,"Fe":20.0,"Cu":1.0,
            "Mn":15.0,"B":1.5,"Mo":0.4,"CEC":25.0,"Sand":30.0,"Silt":40.0,
            "Clay":30.0,"Bulk_Density":1.3,"Water_Holding_Capacity":35.0,
            "Slope":5.0,"Aspect":180.0,"Elevation":200.0,"Solar_Radiation":2000.0,
            "Wind_Speed":12.0,"NDVI":0.65,"EVI":0.50,"LAI":3.5,"Chlorophyll":35.0,
            "GDD":1800.0,"Crop_Type":"Maize","Growth_Stage":"Reproductive",
            "Irrigation_Frequency":7,"Fertilizer_Type":"Mixed","Pesticide_Usage":"Medium",
            "Region":"South","Season":"Kharif","Growing_Days":90,"Planting_Month":6,"Harvest_Month":9,"Yield":5.2
        }])
        tpl_col,_ = st.columns([2,4])
        with tpl_col:
            st.download_button("⬇️  Download Template CSV",
                               tpl_df.to_csv(index=False).encode("utf-8"),
                               "template_batch.csv","text/csv")
        st.caption("Kolom `Yield` bersifat opsional — hapus jika hanya ingin prediksi tanpa evaluasi.")


# ══════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════
st.markdown(f"""
<div style='text-align:center;color:{T3};font-family:"JetBrains Mono",monospace;
            font-size:10px;margin-top:2.5rem;padding-top:1rem;border-top:1px solid {BORDER}'>
  🌾 &nbsp; AgriYield Dashboard &nbsp;·&nbsp; Mini Project Penggalian Data &nbsp;·&nbsp;
  Model: <span style='color:{CYAN}'>{MODEL_NAME}</span> &nbsp;·&nbsp;
  Fitur: <span style='color:{CYAN}'>{len(FEAT_COLS)}</span> kolom
</div>
""", unsafe_allow_html=True)