import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Soft Premium Light Off-White & Deep Wine Aesthetic Setup
st.set_page_config(page_title="Fraud Sentry Dashboard", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #fcfaf7; color: #2b1111; }
    .main-title { color: #4a0f0f; font-family: 'Inter', sans-serif; font-size: 36px; text-align: center; font-weight: 700; margin-bottom: 2px; }
    .sub-title { color: #8c6e6e; font-family: 'Inter', sans-serif; font-size: 14px; text-align: center; margin-top: 0px; margin-bottom: 20px; }
    h3 { color: #4a0f0f !important; font-family: 'Inter', sans-serif; font-weight: 600; font-size: 20px; margin-bottom: 15px; }
    .stNumberInput input, .stSelectbox div, .stSlider div { background-color: #ffffff !important; color: #2b1111 !important; border: 1px solid rgba(74, 15, 15, 0.15) !important; border-radius: 6px !important; }
    .stButton>button { background: #4a0f0f; color: #ffffff !important; border: none; border-radius: 6px; width: 100%; height: 45px; font-size: 16px; font-weight: 500; }
    .stButton>button:hover { background: #6b1d1d; }
    .result-box-safe { background-color: rgba(40, 167, 69, 0.08); border: 1px solid #28a745; padding: 20px; border-radius: 8px; text-align: center; }
    .result-box-fraud { background-color: rgba(220, 53, 69, 0.08); border: 1px solid #dc3545; padding: 20px; border-radius: 8px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>FRAUD SENTRY: FINANCIAL RISK DASHBOARD</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Developed by Aein</div>", unsafe_allow_html=True)
st.markdown("<hr style='border: 0; border-top: 1px solid rgba(74, 15, 15, 0.1); margin-bottom: 30px;'>", unsafe_allow_html=True)

# 2. Single Asset Loader (Unified Pipeline)
@st.cache_resource
def load_pipeline():
    with open('matrix_model.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

try:
    model_pipeline = load_pipeline()
except Exception as e:
    st.error("Model Loading Error: Ensure 'matrix_model.pkl' is uploaded to GitHub.")

# 3. Layout Configuration
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### Input Transaction Metrics")
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0)
    account_age_days = st.number_input("Account Age (Days)", min_value=0, value=365)
    shipping_distance_km = st.number_input("Shipping Distance (KM)", min_value=0.0, value=45.0)
    avg_amount_user = st.number_input("User Average Historical Amount ($)", min_value=0.0, value=120.0)
    total_transactions_user = st.number_input("Total Historical Transactions", min_value=0, value=25)
    transaction_hour = st.slider("Hour of Transaction (0-23)", 0, 23, 14)
    avs_match = st.selectbox("AVS Match Status", [1, 0])
    cvv_result = st.selectbox("CVV Verification Result", [1, 0])
    three_ds_flag = st.selectbox("3D Secure Dynamic Flag", [1, 0])
    promo_used = st.selectbox("Promo Code Applied", [0, 1])
    
    input_data = np.zeros(33)
    input_data[0] = account_age_days
    input_data[1] = total_transactions_user
    input_data[2] = avg_amount_user
    input_data[3] = amount
    input_data[4] = promo_used
    input_data[5] = avs_match
    input_data[6] = cvv_result
    input_data[7] = three_ds_flag
    input_data[8] = shipping_distance_km
    input_data[9] = transaction_hour
    
    st.write("")
    predict_btn = st.button("Run Risk Evaluation")

with col2:
    st.markdown("### Operational Integrity Report")
    if predict_btn:
        # Pipeline automatic scale bhi karega aur predict bhi karega!
        prediction = model_pipeline.predict([input_data])[0]
        prediction_proba = model_pipeline.predict_proba([input_data])[0]
        
        if prediction == 1:
            st.markdown(f"""
                <div class='result-box-fraud'>
                    <h2 style='color: #dc3545; margin: 0; font-size: 22px; font-weight: 600;'>TRANSACTION BLOCKED</h2>
                    <p style='color: #2b1111; font-size: 15px; margin-top: 8px;'>High Probability Fraud Pattern Detected</p>
                    <h4 style='color: #8c6e6e; margin: 5px 0 0 0; font-size: 16px;'>Confidence: {prediction_proba[1] * 100:.1f}%</h4>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='result-box-safe'>
                    <h2 style='color: #28a745; margin: 0; font-size: 22px; font-weight: 600;'>TRANSACTION CLEAN</h2>
                    <p style='color: #2b1111; font-size: 15px; margin-top: 8px;'>Authorized and Cleared for Settlement</p>
                    <h4 style='color: #8c6e6e; margin: 5px 0 0 0; font-size: 16px;'>Confidence: {prediction_proba[0] * 100:.1f}%</h4>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Awaiting input execution. Adjust the system metrics on the left panel.")
        
    st.write("")
    st.markdown("### Core Decision Metrics")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    fig.patch.set_facecolor('#fcfaf7')
    ax.set_facecolor('#ffffff')
    
    features = ['Transaction Hour', 'Shipping Distance', 'Avg Amount', 'Account Age', 'Amount']
    importance = [0.008, 0.024, 0.026, 0.029, 0.033]
    
    ax.barh(features, importance, color='#4a0f0f', edgecolor='#4a0f0f', linewidth=0.5, height=0.5)
    ax.set_title("Top Operational Drivers Behind Risk Assessment", color='#4a0f0f', fontsize=10, weight='medium')
    ax.tick_params(colors='#4a0f0f', labelsize=8)
    ax.xaxis.grid(True, linestyle='--', alpha=0.1, color='#4a0f0f')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#4a0f0f')
    ax.spines['bottom'].set_color('#4a0f0f')
    
    st.pyplot(fig)
