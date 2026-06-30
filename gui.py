"""
Streamlit GUI for POFA concrete compressive-strength prediction.

Run with:
    streamlit run gui.py
"""

import os
import joblib
import streamlit as st

from utils import predict_strength, get_output_dirs


MODEL_DISPLAY = {
    'Random Forest':      'rf_model_v2.pkl',
    'Gradient Boosting':  'gb_model_v2.pkl',
    'SVR':                'svr_model_v2.pkl',
    'Bagging':            'bagging_model_v2.pkl',
    'ANN':                'ann_model_v2.pkl',
    'GEP':                'gep_model_v2.pkl',
    'MARS':               'mars_model_v2.pkl',
    'M5P':                'm5p_model_v2.pkl',
    'Hybrid':             'hybrid_model_v2.pkl',
}

MODEL_DISPLAY_V1 = {
    'Linear Regression':  'linear_regression_model.pkl',
    'Random Forest':      'random_forest_model_tuned.pkl',
    'Gradient Boosting':  'gradient_boosting_model_tuned.pkl',
    'SVR':                'svr_model_tuned.pkl',
    'Bagging':            'bagging_model_tuned.pkl',
    'GEP':                'gep_model.pkl',
    'Hybrid':             'hybrid_model.pkl',
}

DEFAULTS = {
    'cement': 400.0,
    'pofa':   100.0,
    'ca_fa':  1.30,
    'w_b':    0.35,
    'sp':     5.0,
    'age':    28,
}


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_scaler(version):
    dirs = get_output_dirs(version)
    path = os.path.join(dirs['models'], 'pofa_concrete_strength_scaler.pkl')
    if not os.path.exists(path):
        # Fallback to v1 scaler if requested version doesn't have one
        path = os.path.join(
            get_output_dirs('v1')['models'],
            'pofa_concrete_strength_scaler.pkl',
        )
    return joblib.load(path)


@st.cache_resource
def load_model(version, filename):
    return joblib.load(os.path.join(get_output_dirs(version)['models'], filename))


def get_available_models(version):
    mapping = MODEL_DISPLAY if version == 'v2' else MODEL_DISPLAY_V1
    dirs = get_output_dirs(version)
    return {
        name: fname for name, fname in mapping.items()
        if os.path.exists(os.path.join(dirs['models'], fname))
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="POFA Concrete CS Predictor",
    page_icon="🏗️",
    layout="centered",
)

st.title("GUI Model for Predicting Concrete CS")
st.caption("Predict POFA concrete compressive strength using a trained ML model.")

# Sidebar — version + model selection
st.sidebar.header("Model")
version = st.sidebar.radio("Version", options=['v2', 'v1'], index=0,
                           help="v2 = Optuna-tuned. v1 = baseline tuned.")

available = get_available_models(version)

if not available:
    st.warning(
        f"No {version} models found at `out/{version}/models/`. "
        f"Run `python run_v2.py --trials 200` first, "
        f"or switch to the other version in the sidebar."
    )
    other = 'v1' if version == 'v2' else 'v2'
    other_available = get_available_models(other)
    if not other_available:
        st.error("No models found in either v1 or v2. Train the models first.")
        st.stop()
    else:
        st.info(f"Hint: {len(other_available)} model(s) available under `{other}`.")
        st.stop()

selected_name = st.sidebar.selectbox("Select model", list(available.keys()), index=0)
selected_file = available[selected_name]
st.sidebar.caption(f"Loaded: `{selected_file}`")

# Main inputs
st.subheader("Definition of Parameters")

col1, col2 = st.columns(2)
with col1:
    cement = st.number_input(
        "X1: Cement (kg/m³)",
        value=DEFAULTS['cement'], min_value=0.0, step=10.0, format="%.2f",
    )
    pofa = st.number_input(
        "X2: POFA (kg/m³)",
        value=DEFAULTS['pofa'], min_value=0.0, step=5.0, format="%.2f",
    )
    ca_fa = st.number_input(
        "X3: CA/FA ratio",
        value=DEFAULTS['ca_fa'], min_value=0.0, step=0.05, format="%.3f",
    )
with col2:
    w_b = st.number_input(
        "X4: W/B ratio",
        value=DEFAULTS['w_b'], min_value=0.0, step=0.01, format="%.3f",
    )
    sp = st.number_input(
        "X5: Superplasticizer (kg/m³)",
        value=DEFAULTS['sp'], min_value=0.0, step=0.5, format="%.2f",
    )
    age = st.number_input(
        "X6: Age (days)",
        value=DEFAULTS['age'], min_value=1, step=1,
    )

st.divider()

col_predict, col_clear = st.columns([1, 1])
with col_predict:
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear", use_container_width=True):
        st.rerun()

# Prediction output
if predict_clicked:
    try:
        scaler = load_scaler(version)
        model = load_model(version, selected_file)
        prediction = predict_strength(
            model, scaler, cement, pofa, ca_fa, w_b, sp, age,
        )

        st.subheader("Output:")
        st.markdown(
            f"<div style='font-size:28px; color:#cc0000; font-weight:bold; "
            f"border:1px solid #ccc; padding:14px 18px; border-radius:6px; "
            f"display:inline-block; min-width:420px;'>"
            f"Concrete compressive strength (MPa): {prediction:.4f}"
            f"</div>",
            unsafe_allow_html=True,
        )

        with st.expander("Input summary"):
            st.write({
                "Model":              selected_name,
                "Version":            version,
                "Cement (kg/m³)":     cement,
                "POFA (kg/m³)":       pofa,
                "CA/FA ratio":        ca_fa,
                "W/B ratio":          w_b,
                "Superplasticizer (kg/m³)": sp,
                "Age (days)":         int(age),
            })
    except FileNotFoundError as e:
        st.error(f"Required file missing: {e}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
