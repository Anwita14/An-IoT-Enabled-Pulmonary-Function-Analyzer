# ====================================================
# AirVista: Smart Pulmonary Health Tracker — Updated App
# Option A (uses model.pkl, scaler.pkl). Robust feature matching.
# UI UPDATE: High Contrast Mode for better visibility
# ====================================================
import streamlit as st
st.set_page_config(page_title="AirVista - Pulmonary Monitor", page_icon="🫁", layout="wide")

import time, serial, joblib, warnings, io, os, sys, base64, json
import pandas as pd, numpy as np
import plotly.express as px
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# -----------------------------
# SETTINGS (edit if needed)
# -----------------------------
EXPECTED_MODEL_NAMES = ["model.pkl", "model (1).pkl", "airvista_model.pkl", "airvista_model (1).pkl"]
EXPECTED_SCALER_NAMES = ["scaler.pkl", "scaler (1).pkl", "airvista_scaler.pkl", "airvista_scaler (1).pkl"]
FEATURE_NAMES_FILES = ["feature_names.json","feature_names.txt","feature_names.npy","feature_names.csv"]
DATASET_FILE = "AirVista_Balanced_Dataset_with_BTPS.csv"
GROUP_MEMBERS = ["Anwita Ghosh", "Dhrubajoti Adhikari", "Mehuli Poddar", "Atmika Paul", "Aikantika Roy"]
PROJECT_UNIQUENESS = (
    "AirVista integrates low-cost IoT spirometry with BTPS correction and an interpretable "
    "ML model to detect pulmonary and cardiopulmonary conditions and produce automated PDF reports."
)

RECORD_SECONDS = 30  # user requested 20 seconds of recording

# -----------------------------
# Helper: find file by common names
# -----------------------------
def find_first_existing(names):
    for n in names:
        if os.path.exists(n):
            return n
    return None

MODEL_FILE = find_first_existing(EXPECTED_MODEL_NAMES)
SCALER_FILE = find_first_existing(EXPECTED_SCALER_NAMES)

if MODEL_FILE is None or SCALER_FILE is None:
    st.error(f"Model/scaler not found. Put model.pkl and scaler.pkl (or alternates) in the app folder.\n"
             f"Looked for: {EXPECTED_MODEL_NAMES} and {EXPECTED_SCALER_NAMES}")
    st.stop()

# -----------------------------
# Load model & scaler
# -----------------------------
try:
    model = joblib.load(MODEL_FILE)
except Exception as e:
    st.error(f"Failed to load model '{MODEL_FILE}': {e}")
    st.stop()

try:
    scaler = joblib.load(SCALER_FILE)
except Exception as e:
    st.error(f"Failed to load scaler '{SCALER_FILE}': {e}")
    st.stop()

# -----------------------------
# Determine expected feature names for scaler
# -----------------------------
expected_features = None
# 1) sklearn scalers sometimes store feature_names_in_
if hasattr(scaler, "feature_names_in_"):
    expected_features = list(scaler.feature_names_in_)
# 2) some people saved a separate feature_names.* file
if expected_features is None:
    for fname in FEATURE_NAMES_FILES:
        if os.path.exists(fname):
            try:
                if fname.endswith(".json") or fname.endswith(".txt"):
                    with open(fname, "r", encoding="utf-8") as f:
                        data = json.load(f) if fname.endswith(".json") else [ln.strip() for ln in f if ln.strip()]
                        expected_features = list(data)
                elif fname.endswith(".npy"):
                    expected_features = list(np.load(fname))
                elif fname.endswith(".csv"):
                    expected_features = list(pd.read_csv(fname, header=None).iloc[:,0].astype(str).tolist())
            except Exception:
                pass
            if expected_features:
                break

# If still None, we try to get scaler.n_features_in_ and leave names None (we will attempt a best-effort)
if expected_features is None and hasattr(scaler, "n_features_in_"):
    n_features = int(scaler.n_features_in_)
    expected_features = None  # unknown names, will handle via fallback mapping
else:
    n_features = None

# -----------------------------
# THEME & CSS (Premium dark UI, IMPROVED CONTRAST)
# -----------------------------
st.markdown("""
<style>
/* Global App Background */
.stApp {
  background: linear-gradient(180deg,#060918 0%, #07102a 100%) !important;
  font-family: 'Inter', sans-serif;
}

/* HIGH CONTRAST TEXT OVERRIDES */
/* Forces headers, paragraphs, lists, and general text to be bright white/off-white */
h1, h2, h3, h4, h5, h6, .stMarkdown, .stText, p, li {
  color: #f0f8ff !important; 
}

/* Hero Section */
.hero-title { 
  font-size: 2.6rem; 
  font-weight: 800; 
  text-align:center;
  background: linear-gradient(90deg,#39f0ff,#7ef4c4); 
  -webkit-background-clip: text; 
  -webkit-text-fill-color: transparent;
}
.hero-sub { 
  text-align:center; 
  color: #d0eaff !important; /* Brighter blue for better visibility against dark bg */
  margin-bottom:16px; 
  font-weight: 500;
}

/* Section Box Container */
.section-box { 
  background: #111b2e; /* Slightly lighter than main bg for layering */
  padding: 18px; 
  border-radius: 12px; 
  border: 1px solid rgba(255,255,255,0.15); /* Brighter border */
  box-shadow: 0 8px 20px rgba(0,0,0,0.5); 
  margin-bottom:12px; 
}

/* Input Fields (Text Input, Number Input, Selectbox) */
.stTextInput>div>div>input, .stNumberInput input, .stSelectbox>div>div {
  background-color: #16253d !important; /* Distinct dark blue */
  color: #ffffff !important; /* Pure white input text */
  border-radius: 8px !important; 
  border: 1px solid #4a6fa5 !important; /* Lighter border to make input visible */
}

/* Selectbox dropdown menu for better visibility */
.stSelectbox [data-testid="stVirtualDropdown"] {
  background-color: #16253d !important;
}
.stSelectbox [role="option"] {
  color: #ffffff !important;
  background-color: #16253d !important;
}
.stSelectbox [role="option"]:hover {
  background-color: #2bb6ff !important;
  color: #000000 !important;
}

/* Input Labels (The text above the boxes) */
.stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label {
  color: #ffffff !important;
  font-weight: 600 !important;
}

/* Buttons */
.stButton>button { 
  background: linear-gradient(90deg,#2bb6ff,#00c2a8) !important; 
  color: #000000 !important; /* Black text on bright button for max contrast */
  font-weight: 700; 
  padding: 8px 14px; 
  border-radius: 10px; 
  border: none;
}
.stButton>button:hover { 
  transform:translateY(-2px); 
  color: #000000 !important;
  box-shadow: 0 4px 12px rgba(43, 182, 255, 0.4);
}

/* Download Buttons Specific */
.stDownloadButton>button {
  background: linear-gradient(90deg,#2bb6ff,#00c2a8) !important; 
  color: #000000 !important;
  font-weight: 700; 
  padding: 8px 14px; 
  border-radius: 10px; 
  border: none;
}
.stDownloadButton>button:hover {
  transform:translateY(-2px); 
  color: #000000 !important;
  box-shadow: 0 4px 12px rgba(43, 182, 255, 0.4);
}

/* Metrics (The big numbers) */
[data-testid="stMetricLabel"] {
  color: #c0d6df !important; /* Light grey label */
}
[data-testid="stMetricValue"] {
  color: #39f0ff !important; /* Neon blue value */
}

/* Expander headers */
.st-expanderHeader { 
  color: #ffffff !important; 
  background-color: #111b2e !important;
}

/* Dataframe/Table overrides (if any appear) */
[data-testid="stDataFrame"] {
  background-color: #0c1420;
}

header, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HERO
# -----------------------------
st.markdown('<div class="hero-title">🫁 AirVista: Smart Pulmonary Health Tracker</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">IoT spirometry • BTPS correction • ML diagnosis • PDF reporting</div>', unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# Sidebar: patient + connect
# -----------------------------
with st.sidebar:
    st.header("Patient Info")
    patient_name = st.text_input("Full name", "Anwita Ghosh")
    age = st.number_input("Age (years)", min_value=5, max_value=120, value=21)
    sex = st.selectbox("Sex", ["Male", "Female"])
    height = st.number_input("Height (cm)", min_value=100, max_value=220, value=172)
    st.markdown("---")
    st.header("Device Connection")
    port = st.text_input("ESP32 Serial Port (e.g., COM3 or /dev/ttyUSB0)", value="COM5")
    if "ser" not in st.session_state:
        st.session_state["ser"] = None
        st.session_state["port_ok"] = False
    if st.button("Connect Device"):
        try:
            if st.session_state["ser"] and st.session_state["ser"].is_open:
                st.session_state["ser"].close()
            st.session_state["ser"] = serial.Serial(port, 115200, timeout=1)
            st.session_state["port_ok"] = True
            st.success(f"Connected to {port}")
        except Exception as e:
            st.session_state["port_ok"] = False
            st.error(f"Connection failed: {e}")

st.markdown("---")

# -----------------------------
# Main columns: live + controls
# -----------------------------
col1, col2 = st.columns([1,1])

with col1:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Live Lung Visualization")
    lung_placeholder = st.empty()
    st.caption("Live airflow visualization and SpO₂ color indicator.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Run Test")
    st.write(f"Ensure device connected in the sidebar. Click Start Test to record a {RECORD_SECONDS}s session.")
    start = st.button("🫁 Start Test")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Utility functions (BTPS, sanitize, flags, mapping)
# -----------------------------
def sat_vap_pressure_pa(temp_c):
    a, b, c = 6.112, 17.62, 243.12
    svp_hpa = a * np.exp((b * temp_c) / (c + temp_c))
    return svp_hpa * 100.0

def compute_btps_factor(temp_c, rh_percent, p_atm=101325.0):
    T_amb_K = 273.15 + temp_c
    T_body_K = 273.15 + 37.0
    e_ambient = (rh_percent / 100.0) * sat_vap_pressure_pa(temp_c)
    e_body = sat_vap_pressure_pa(37.0)
    denom = max(p_atm - e_body, 1.0)
    factor = (T_body_K / T_amb_K) * ((p_atm - e_ambient) / denom)
    return float(factor)

def sanitize_for_latin1(s):
    # Remove control characters and replace non-latin1 with '?'
    if s is None:
        return ""
    return s.encode('latin-1', errors='replace').decode('latin-1')

def hr_flag_from_hr(hr):
    if hr is None:
        return "Normal"
    if hr > 100:
        return "Tachycardia"
    if hr < 60:
        return "Bradycardia"
    return "Normal"

def spo2_flag_from_spo2(spo2):
    if spo2 is None:
        return "Normal"
    if spo2 < 90:
        return "Low"
    return "Normal"

# Attempt to load explicit feature names from a file named 'feature_names.*' if present
explicit_feature_names = None
for f in FEATURE_NAMES_FILES:
    if os.path.exists(f):
        try:
            if f.endswith(".json"):
                explicit_feature_names = json.load(open(f, "r", encoding="utf-8"))
            elif f.endswith(".txt"):
                explicit_feature_names = [ln.strip() for ln in open(f, "r", encoding="utf-8") if ln.strip()]
            elif f.endswith(".npy"):
                explicit_feature_names = list(np.load(f))
            elif f.endswith(".csv"):
                explicit_feature_names = list(pd.read_csv(f, header=None).iloc[:,0].astype(str).tolist())
            break
        except Exception:
            explicit_feature_names = None

if explicit_feature_names:
    expected_features = list(explicit_feature_names)

# -----------------------------
# Disease text
# -----------------------------
DISEASE_EXPLANATIONS = (
    "· Normal: FVC and FEV1 near predicted values for age/sex/height; FEV1/FVC ratio within normal limits. \n\n"
    "· Obstructive Disease: Characterized by reduced FEV1 and reduced FEV1/FVC ratio (<0.70), typically seen in asthma and COPD. Flow curve often shows concave (scooped) shape and reduced PEF.\n\n"
    "· Restrictive Disease: Characterized by reduced total lung volumes (FVC < 80% predicted) with a normal or high FEV1/FVC ratio, seen in interstitial lung disease, chest wall disorders\n\n"
    "· Mixed Disease: Features of both obstructive and restrictive physiology, low FVC and low FEV1/FVC ratio. \n\n"
    "· Tachycardia / Bradycardia: Heart rate abnormalities (HR >100 bpm or HR\n\n"
    "· Cardiopulmonary Impairment: Combined abnormal lung function plus reduced SpO2 and abnormal HR, requires urgent clinical attention."
)

# -----------------------------
# Recording & processing logic
# -----------------------------
if start:
    if not st.session_state.get("port_ok", False):
        st.warning("Please connect the device from the sidebar first.")
    else:
        ser = st.session_state["ser"]
        st.info("Get ready — recording begins in 3 seconds.")
        for i in range(3,0,-1):
            st.markdown(f"### ⏳ {i}")
            time.sleep(1)
        # ====================== SEND SEX TO ESP32 ======================
        sex_command = "Male" if sex == "Male" else "Female"
        ser.write((sex_command + "\n").encode('utf-8'))
        time.sleep(0.6)                     # give ESP32 time to read and set k_flow
        st.success(f"✅ Sent {sex_command} to device → k_flow automatically adjusted")
        # ================================================================
        st.success(f"Recording now — please blow into the spirometer ({RECORD_SECONDS} seconds).")
        timer_placeholder = st.empty()
        t0 = time.time()
        collected = []
        placeholder = st.empty()
        phase = 0
        # read lines for RECORD_SECONDS
        while time.time() - t0 < RECORD_SECONDS:
            # LIVE COUNTDOWN
            remaining = max(0, RECORD_SECONDS - int(time.time() - t0))
            timer_placeholder.markdown(f"""
            <div style="text-align:center; font-size:3.8rem; font-weight:bold; color:#39f0ff; 
            background:rgba(57,240,255,0.12); padding:25px; border-radius:20px; margin:20px 0; 
            border:3px solid #39f0ff; box-shadow:0 0 30px rgba(57,240,255,0.5);">
            ⏱️ <b>{remaining}</b> seconds remaining
            </div>
            """, unsafe_allow_html=True)
            try:
                raw = ser.readline().decode(errors='ignore').strip()
                if not raw or ',' not in raw:
                    continue
                parts = raw.split(',')
                try:
                    vals = [float(x) for x in parts[:8]]
                except:
                    continue
                if len(vals) >= 8:
                    collected.append(vals)
                    airflow = vals[1]
                    spo2 = vals[6]
                    phase += 1
                    size = 60 + 12 * np.sin(phase/2.2)
                    color = "#39f0ff" if spo2 >= 95 else "#ffd166" if spo2 >= 90 else "#ff6b6b"
                    svg = f"""
                    <div style='text-align:center;'>
                      <svg width="320" height="200">
                        <ellipse cx="110" cy="100" rx="{size}" ry="{size*1.25}" fill="{color}" opacity="0.75"/>
                        <ellipse cx="210" cy="100" rx="{size}" ry="{size*1.25}" fill="{color}" opacity="0.75"/>
                      </svg>
                      <p style='color:#ffffff; font-weight:600;'>Airflow: <b>{airflow:.2f}</b> L/s | SpO₂: <b>{spo2:.1f}%</b></p>
                    </div>"""
                    placeholder.markdown(svg, unsafe_allow_html=True)
            except Exception:
                continue
        timer_placeholder.empty()   # remove timer when done
        st.success("Recording complete — processing data...")

        if len(collected) == 0:
            st.error("No valid sensor data recorded.")
            st.stop()

        # build dataframe
        df = pd.DataFrame(collected, columns=[
            "Pressure_Pa","Airflow_Lps","Volume_L",
            "Temp_C","Humidity_Percent","HR_bpm","SpO2_Percent","Tilt_Degree"
        ]).fillna(0)

        # compute raw and BTPS corrected
        med_temp = float(df["Temp_C"].median())
        med_hum = float(df["Humidity_Percent"].median())
        btps = compute_btps_factor(med_temp, med_hum)
        # Raw lung values
        FVC_raw_L = float(df["Volume_L"].max())

        # Compute timestamps (30-second fixed test)
        df["Time_s"] = np.linspace(0, 30, len(df))
        # Detect start of exhalation (volume rising)
        vol_threshold = 0.02  # liters (tunable)
        blow_idx = df[df["Volume_L"] > vol_threshold].index
        if len(blow_idx) > 0:
            start_idx = blow_idx[0]
            start_time = df.loc[start_idx, "Time_s"]

            fev_window = df[
            (df["Time_s"] >= start_time) &
            (df["Time_s"] <= start_time + 21)
         ]
            FEV1_raw_L = float(fev_window["Volume_L"].max())
        else:
            FEV1_raw_L = 0.0

        # Peak expiratory flow
        PEF_raw_Lps = float(df["Airflow_Lps"].max())
        PEF_raw_Lmin = PEF_raw_Lps * 60.0

        # BTPS corrected
        FVC_L = round(FVC_raw_L * btps, 4)
        FEV1_L = round(FEV1_raw_L * btps, 4)
        PEF_Lps = round(PEF_raw_Lps * btps, 4)
        PEF_Lmin = round(PEF_Lps * 60.0, 4)

        med_hr = float(df["HR_bpm"].median())
        med_spo2 = float(df["SpO2_Percent"].median())
        med_tilt = float(df["Tilt_Degree"].median())

        # predicted values (same formula used when dataset created)
        def predicted_values_simple(sex_s, age_s, height_cm):
            if sex_s == "Male":
                FVC_pred_L = 0.052 * height_cm - 0.028 * age_s - 3.20
                FEV1_pred_L = 0.041 * height_cm - 0.024 * age_s - 2.19
            else:
                FVC_pred_L = 0.049 * height_cm - 0.018 * age_s - 3.59
                FEV1_pred_L = 0.034 * height_cm - 0.025 * age_s - 1.76
            FVC_pred_L = max(FVC_pred_L, 0.5)
            FEV1_pred_L = max(FEV1_pred_L, 0.4)
            PEF_pred_Lmin = max(FEV1_pred_L * 160.0, 30.0)  # same scale as dataset creator
            return float(FVC_pred_L), float(FEV1_pred_L), float(PEF_pred_Lmin)

        FVC_pred_L, FEV1_pred_L, PEF_pred_Lmin = predicted_values_simple(sex, age, height)
        PEF_pred_Lps = PEF_pred_Lmin / 60.0

        # percentages
        FVC_percent = round(100.0 * FVC_L / FVC_pred_L, 1) if FVC_pred_L > 0 else 0.0
        # We cannot use FEV1_pred_L (because that formula assumes true 1-sec FEV1)
        # So create an adjusted reference using the same ratio as the dataset:
        FEV1_ref = FVC_pred_L * 0.8   # dataset approx average ratio

        FEV1_percent = round(100.0 * FEV1_L / FEV1_ref, 1) if FEV1_ref > 0 else 0.0
        FEV1_FVC_ratio_val = round((FEV1_L / FVC_L) if FVC_L > 0 else 0.0, 3)

        # flags and raw HR/SpO2 naming variants
        HR_raw_bpm = int(np.median(df["HR_bpm"]))
        HR_flag = hr_flag_from_hr(HR_raw_bpm)
        SpO2_raw_pct = float(np.median(df["SpO2_Percent"]))
        SpO2_flag = spo2_flag_from_spo2(SpO2_raw_pct)

        # Build a very comprehensive single-row dict containing many naming variants the scaler might expect
        row = {
            # core demographic / device fields
            "Age": age,
            "Sex": 1 if sex == "Male" else 0,
            "Height_cm": height,
            "BTPS_factor": btps,
            "Pressure_Pa": float(df["Pressure_Pa"].median()),
            # raw sensor names
            "Airflow_Lps": float(np.median(df["Airflow_Lps"])),
            "Volume_L": float(np.median(df["Volume_L"])),
            # raw computed
            "FVC_raw_L": float(FVC_raw_L),
            "FEV1_raw_L": float(FEV1_raw_L),
            "PEF_raw_Lps": float(PEF_raw_Lps),
            "PEF_raw_Lmin": float(PEF_raw_Lmin),
            # BTPS corrected
            "FVC_L": float(FVC_L),
            "FEV1_L": float(FEV1_L),
            "PEF_Lps": float(PEF_Lps),
            "PEF_Lmin": float(PEF_Lmin),
            # predicted
            "FVC_pred_L": float(FVC_pred_L),
            "FEV1_pred_L": float(FEV1_pred_L),
            "PEF_pred_Lmin": float(PEF_pred_Lmin),
            "PEF_pred_Lps": float(PEF_pred_Lmin/60.0),
            # percentages and ratios (variants)
            "FVC_percent": float(FVC_percent),
            "FEV1_percent": float(FEV1_percent),
            "FEV1_FVC_ratio": float(FEV1_FVC_ratio_val),  # alternate naming
            "FEV1_FVC_Ratio": float(FEV1_FVC_ratio_val),  # another case
            # HR / SpO2 raw and flags (variants)
            "HR_raw_bpm": HR_raw_bpm,
            "HeartRate_bpm": HR_raw_bpm,
            "HR_flag": HR_flag,
            "SpO2_raw_pct": SpO2_raw_pct,
            "SpO2_Percent": SpO2_raw_pct,
            "SpO2_flag": SpO2_flag,
            # environment and tilt
            "Temp_C": med_temp,
            "Humidity_Percent": med_hum,
            "Humidity_pct": med_hum,
            "Tilt_Degree": med_tilt,
        }

        # If expected_features is known (list), create the ordered DataFrame using those names,
        # filling missing names from aliases in 'row' above, else fallback to a default column order.
        constructed_order = None
        if expected_features:
            constructed_order = []
            # lowercase mapping helper
            lower_map = {k.lower(): k for k in row.keys()}
            for feat in expected_features:
                # try direct match
                if feat in row:
                    constructed_order.append(row[feat])
                    continue
                # try case-insensitive
                if feat.lower() in lower_map:
                    constructed_order.append(row[lower_map[feat.lower()]])
                    continue
                # try common aliases:
                alias_candidates = {
                    "humidity_pct": ["Humidity_Percent","Humidity_pct","humidity_percent","humidity_pct"],
                    "pef_raw_lmin": ["PEF_raw_Lmin","PEF_raw_Lps","PEF_raw_Lmin"],
                    "pef_lmin": ["PEF_Lmin","PEF_Lps","PEF_Lmin"],
                    "pef_pred_lmin": ["PEF_pred_Lmin","PEF_pred_Lps","PEF_pred_Lmin"],
                    "hr_raw_bpm": ["HR_raw_bpm","HeartRate_bpm","HR_bpm"],
                    "spO2_raw_pct": ["SpO2_raw_pct","SpO2_Percent","SpO2_pct"],
                    "btps_factor": ["BTPS_factor","btps_factor"]
                }
                found = False
                for key, aliases in alias_candidates.items():
                    if feat.lower() == key.lower():
                        for a in aliases:
                            if a in row:
                                constructed_order.append(row[a])
                                found = True
                                break
                        if found:
                            break
                if found:
                    continue
                # if nothing found, append 0 as safe default
                constructed_order.append(0.0)
            # create DataFrame
            Xf = pd.DataFrame([constructed_order], columns=expected_features)
        else:
            # fallback: use a best-effort column list (common columns). This should match most training schemas.
            fallback_cols = [
                "Age","Sex","Height_cm","BTPS_factor","Pressure_Pa","Airflow_Lps","Volume_L",
                "FVC_raw_L","FEV1_raw_L","PEF_raw_Lmin","PEF_raw_Lps",
                "FVC_L","FEV1_L","PEF_Lps","PEF_Lmin",
                "FVC_pred_L","FEV1_pred_L","PEF_pred_Lmin","PEF_pred_Lps",
                "FVC_percent","FEV1_percent","FEV1_FVC_ratio","FEV1_FVC_Ratio",
                "HeartRate_bpm","HR_raw_bpm","HR_flag","SpO2_Percent","SpO2_raw_pct","SpO2_flag",
                "Temp_C","Humidity_Percent","Humidity_pct","Tilt_Degree"
            ]
            # populate
            vals = [row.get(c, row.get(c.replace("_"," ").title(), 0.0)) for c in fallback_cols]
            Xf = pd.DataFrame([vals], columns=fallback_cols)

        # Before scaling: ensure all columns are numeric
        for col in Xf.columns:
            try:
                Xf[col] = pd.to_numeric(Xf[col], errors='coerce').fillna(0.0)
            except Exception:
                Xf[col] = 0.0

        # Do scaler.transform, but catch and report helpful message if mismatch
        try:
            Xf_scaled = scaler.transform(Xf)
        except Exception as e:
            st.error(
                "Scaler transform error — feature mismatch.\n\n"
                "Details: " + str(e) + "\n\n"
                "What I tried:\n- detected expected features from scaler (if available)\n- built a single-row that includes many common aliases (Humidity_pct, Humidity_Percent, PEF_raw_Lmin, PEF_raw_Lps, HR_raw_bpm, HeartRate_bpm, SpO2_raw_pct, SpO2_Percent, HR_flag, SpO2_flag, BTPS_factor, etc.)\n\n"
                "If this still fails, please send me the exact list of feature names used during training (a text file named 'feature_names.txt' or 'feature_names.json' containing the ordered column names)."
            )
            st.stop()

        # Predict
        try:
            pred = model.predict(Xf_scaled)[0]
            probs = model.predict_proba(Xf_scaled)[0] if hasattr(model, "predict_proba") else None
            class_conf = {}
            if probs is not None and hasattr(model, "classes_"):
                for c,p in zip(model.classes_, probs):
                    class_conf[str(c)] = float(p)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        # Display results
        st.markdown("### 🩺 Diagnosis")
        st.success(f"**{pred}**")
        if class_conf:
            st.markdown("**Confidence per class:**")
            st.json(class_conf)
        # pick short reason paragraph
        EXPLAIN = {
            "Normal Condition": DISEASE_EXPLANATIONS.split("\n\n")[0].replace("· ", ""),
            "Obstructive Disease": DISEASE_EXPLANATIONS.split("\n\n")[1].replace("· ", ""),
            "Restrictive Disease": DISEASE_EXPLANATIONS.split("\n\n")[2].replace("· ", ""),
            "Mixed Disease": DISEASE_EXPLANATIONS.split("\n\n")[3].replace("· ", ""),
            "Tachycardia": DISEASE_EXPLANATIONS.split("\n\n")[4].replace("· ", ""),
            "Bradycardia": DISEASE_EXPLANATIONS.split("\n\n")[4].replace("· ", ""),
            "Cardiopulmonary Impairment": DISEASE_EXPLANATIONS.split("\n\n")[5].replace("· ", "")
        }
        st.info("Why: " + EXPLAIN.get(pred, "Details not available"))
        # Plot airflow
        df["Time_s"] = np.linspace(0, len(df)/10 if len(df)>0 else 1, len(df))
        fig = px.line(df, x="Time_s", y="Airflow_Lps", title="Airflow Trend (L/s)", color_discrete_sequence=["#39f0ff"])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Metrics
        st.markdown("### 📊 Measured Parameters")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("FVC (L)", f"{FVC_L:.3f}")
        c2.metric("FEV1 (L)", f"{FEV1_L:.3f}")
        c3.metric("PEF (L/s)", f"{PEF_Lps:.3f}")
        c4.metric("HR (bpm)", f"{HR_raw_bpm:.0f}")

        # Prepare summary dataframe (clean strings for PDF)
        summary_df = pd.DataFrame({
            "Field": [
                "Patient name","Age","Sex","Height (cm)","Median Temp (°C)","Median Humidity (%)",
                "FVC (L)","FEV1 (L)","PEF (L/s)","FVC %Pred","FEV1 %Pred","FEV1/FVC Ratio","Median HR (bpm)","Median SpO2 (%)","BTPS Factor"
            ],
            "Value": [
                sanitize_for_latin1(str(patient_name)),
                age, sex, height, f"{med_temp:.2f}", f"{med_hum:.2f}",
                f"{FVC_L:.4f}", f"{FEV1_L:.4f}", f"{PEF_Lps:.4f}", f"{FVC_percent:.1f}%", f"{FEV1_percent:.1f}%",
                f"{FEV1_FVC_ratio_val:.3f}", f"{HR_raw_bpm:.0f}", f"{SpO2_raw_pct:.1f}%", f"{btps:.4f}"
            ]
        })

        # CSV download
        csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV Report", data=csv_bytes, file_name=f"AirVista_Report_{patient_name.replace(' ','_')}.csv", mime="text/csv")

        # PDF generation (tries reportlab, else fpdf with sanitized text)
        def create_pdf_bytes(patient_name_local, summary_df_local, diagnosis_local, conf_local, disease_text_local):
            # try reportlab first
            try:
                from reportlab.lib.pagesizes import A4
                from reportlab.pdfgen import canvas
                from reportlab.lib.units import mm
            except Exception:
                # fallback to fpdf (sanitize strings to latin-1)
                try:
                    from fpdf import FPDF
                except Exception:
                    return None, "no_pdf_lib"
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=12)
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(0, 10, f"AirVista Report - {sanitize_for_latin1(patient_name_local)}", ln=True, align="C")
                pdf.ln(6)
                for i, row in summary_df_local.iterrows():
                    pdf.cell(0, 8, f"{sanitize_for_latin1(str(row['Field']))}: {sanitize_for_latin1(str(row['Value']))}", ln=True)
                pdf.ln(6)
                pdf.cell(0, 8, f"Diagnosis: {sanitize_for_latin1(str(diagnosis_local))}", ln=True)
                pdf.ln(6)
                if conf_local:
                    pdf.cell(0, 8, "Confidence per class:", ln=True)
                    for k,v in conf_local.items():
                        pdf.cell(0, 7, f"{sanitize_for_latin1(str(k))}: {v:.3f}", ln=True)
                pdf.ln(6)
                pdf.multi_cell(0, 6, sanitize_for_latin1("Disease explanations:"))
                pdf.multi_cell(0, 6, sanitize_for_latin1(disease_text_local))
                pdf.ln(6)
                pdf.cell(0, 8, "Project Team: " + sanitize_for_latin1(", ".join(GROUP_MEMBERS)), ln=True)
                return pdf.output(dest="S").encode("latin-1"), "fpdf"
            # reportlab path
            buf = io.BytesIO()
            c = canvas.Canvas(buf, pagesize=A4)
            width, height = A4
            margin = 18*mm
            y = height - margin
            c.setFont("Helvetica-Bold", 16)
            c.drawString(margin, y, "AirVista")
            y -= 10*mm
            c.setFont("Helvetica", 10)
            # Patient Info
            c.drawString(margin, y, "Patient Name")
            y -= 6*mm
            c.drawString(margin, y, patient_name_local)
            y -= 6*mm
            c.drawString(margin, y, "Age")
            y -= 6*mm
            c.drawString(margin, y, str(summary_df_local[summary_df_local["Field"] == "Age"]["Value"].values[0]))
            y -= 6*mm
            c.drawString(margin, y, "Sex")
            y -= 6*mm
            c.drawString(margin, y, str(summary_df_local[summary_df_local["Field"] == "Sex"]["Value"].values[0]))
            y -= 6*mm
            c.drawString(margin, y, "Height (cm)")
            y -= 6*mm
            c.drawString(margin, y, str(summary_df_local[summary_df_local["Field"] == "Height (cm)"]["Value"].values[0]))
            y -= 12*mm
            # Metrics
            for field in ["Median Temp (°C)", "Median Humidity (%)", "FVC (L)", "FEV1 (L)", "PEF (L/s)", "FVC %Pred", "FEV1 %Pred", "FEV1/FVC Ratio", "Median HR (bpm)", "Median SpO2 (%)", "BTPS Factor"]:
                if y < margin:
                    c.showPage()
                    y = height - margin
                c.drawString(margin, y, field)
                y -= 6*mm
                c.drawString(margin, y, str(summary_df_local[summary_df_local["Field"] == field]["Value"].values[0]))
                y -= 6*mm
            y -= 6*mm
            # Diagnosis
            if y < margin:
                c.showPage()
                y = height - margin
            c.setFont("Helvetica-Bold", 11)
            c.drawString(margin, y, f"Diagnosis: {diagnosis_local}")
            y -= 8*mm
            # Confidence
            if conf_local:
                if y < margin:
                    c.showPage()
                    y = height - margin
                c.drawString(margin, y, "Confidence per class")
                y -= 6*mm
                c.setFont("Helvetica", 10)
                for k, v in conf_local.items():
                    if y < margin:
                        c.showPage()
                        y = height - margin
                    c.drawString(margin, y, f"{k}: {v:.3f}")
                    y -= 6*mm
            y -= 6*mm
            # Disease explanations
            if y < margin:
                c.showPage()
                y = height - margin
            c.drawString(margin, y, "Disease explanations")
            y -= 6*mm
            c.setFont("Helvetica", 9)
            for ln in disease_text_local.split("\n\n"):
                if y < margin:
                    c.showPage()
                    y = height - margin
                c.drawString(margin, y, ln)
                y -= 5*mm
            y -= 6*mm
            # Project Notes
            if y < margin:
                c.showPage()
                y = height - margin
            c.drawString(margin, y, "Project uniqueness:")
            y -= 5*mm
            c.drawString(margin, y, PROJECT_UNIQUENESS)
            y -= 6*mm
            c.drawString(margin, y, "Project Team:")
            y -= 5*mm
            c.drawString(margin, y, ", ".join(GROUP_MEMBERS))
            c.save()
            buf.seek(0)
            return buf.read(), "reportlab"

        pdf_bytes, pdf_lib = create_pdf_bytes(patient_name, summary_df, pred, (class_conf if 'class_conf' in locals() else None), DISEASE_EXPLANATIONS)
        if pdf_bytes:
            st.download_button("⬇️ Download PDF Report", data=pdf_bytes, file_name=f"AirVista_Report_{patient_name.replace(' ','_')}.pdf", mime="application/pdf")
            st.success(f"PDF generated using: {pdf_lib}")
        else:
            st.warning("PDF libraries not available. Install 'reportlab' or 'fpdf' to enable PDF export.")
            st.info("You can still download the CSV report and convert to PDF externally.")

# -----------------------------
# Description & Team sections
# -----------------------------
st.markdown("---")
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.header("Disease Explanations")
st.write(DISEASE_EXPLANATIONS)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.header("Project Notes")
st.markdown("**Project Uniqueness:** " + PROJECT_UNIQUENESS)
st.markdown("**Project Team:** " + ", ".join(GROUP_MEMBERS))
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")
st.caption("AirVista — prototype for education & research. Not a medical device. For clinical decisions consult healthcare professionals.")