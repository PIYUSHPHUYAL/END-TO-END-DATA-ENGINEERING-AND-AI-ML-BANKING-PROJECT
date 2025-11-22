import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import time
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Transaction Risk Assessment",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    /* Main background - Clean white professional */
    .stApp {
        background: linear-gradient(180deg, #f8f9fb 0%, #ffffff 100%);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container */
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Header styling - Professional banker aesthetic */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: left;
        color: #1a2844;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .sub-header {
        font-size: 1rem;
        text-align: left;
        color: #677487;
        margin-bottom: 2.5rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }

    # /* Card styling - Clean white cards with subtle shadow */
    # .card {
    #     background: #ffffff;
    #     padding: 2.5rem;
    #     border-radius: 12px;
    #     box-shadow: 0 2px 8px rgba(26, 40, 68, 0.08);
    #     border: 1px solid #e8edf5;
    #     margin-bottom: 2rem;
    # }

    /* Result cards - Professional status indicators */
    .result-card-fraud {
        background: linear-gradient(135deg, #fee8e8 0%, #fde0e0 100%);
        padding: 2.5rem;
        border-radius: 12px;
        text-align: center;
        color: #8b2f2f;
        box-shadow: 0 2px 8px rgba(26, 40, 68, 0.08);
        border: 1px solid #f5b8b8;
        margin: 2rem 0;
    }

    .result-card-safe {
        background: linear-gradient(135deg, #e7f7f0 0%, #d9f1e8 100%);
        padding: 2.5rem;
        border-radius: 12px;
        text-align: center;
        color: #1d5f47;
        box-shadow: 0 2px 8px rgba(26, 40, 68, 0.08);
        border: 1px solid #b5dcc8;
        margin: 2rem 0;
    }

    .result-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.3px;
    }

    .result-subtitle {
        font-size: 0.95rem;
        opacity: 0.9;
        font-weight: 500;
    }

    /* Input section styling */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a2844;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        letter-spacing: -0.2px;
    }

    /* Metric cards - Professional dark blue accent */
    .metric-container {
        background: #ffffff;
        padding: 1.75rem;
        border-radius: 12px;
        text-align: center;
        color: #1a2844;
        box-shadow: 0 2px 8px rgba(26, 40, 68, 0.08);
        border: 1px solid #e8edf5;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        color: #0052cc;
    }

    .metric-label {
        font-size: 0.8rem;
        opacity: 0.75;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #677487;
        font-weight: 600;
    }

    /* Button styling - Professional solid design */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #0052cc 0%, #003d99 100%);
        color: #ffffff;
        border: 2px solid #0052cc;
        padding: 1rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 82, 204, 0.12);
        transition: all 0.3s ease;
        letter-spacing: 0.3px;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #003d99 0%, #002966 100%);
        color: #ffffff;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 82, 204, 0.2);
    }

    /* Footer */
    .custom-footer {
        text-align: center;
        color: #677487;
        margin-top: 4rem;
        padding: 2rem 1rem;
        font-size: 0.85rem;
        opacity: 0.85;
        border-top: 1px solid #e8edf5;
    }

    .custom-footer strong {
        color: #1a2844;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def load_models():
    try:
        best_model = joblib.load('best_fraud_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return best_model, scaler, feature_names
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        return None, None, None

model, scaler, feature_names = load_models()

# ============================================
# HEADER
# ============================================
st.markdown('<h1 class="main-header">Transaction Risk Assessment System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered fraud detection for financial institutions</p>', unsafe_allow_html=True)

# ============================================
# INPUT SECTION
# ============================================
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<p class="section-title">Transaction Details</p>', unsafe_allow_html=True)

    amount = st.number_input(
        "Transaction Amount ($)",
        min_value=0.01,
        max_value=10000.0,
        value=250.0,
        step=10.0,
        help="Enter the transaction amount"
    )

    merchant_category = st.selectbox(
        "Merchant Category",
        ['Grocery', 'Restaurant', 'Gas Station', 'Online Retail',
         'Electronics', 'Travel', 'Entertainment', 'Healthcare', 'Utilities'],
        help="Select merchant type"
    )

    card_type = st.selectbox(
        "Card Type",
        ['Debit', 'Credit', 'Prepaid'],
        help="Type of card used"
    )

    card_limit = st.number_input(
        "Card Limit ($)",
        min_value=500,
        max_value=50000,
        value=5000,
        step=500,
        help="Maximum limit on the card"
    )

    has_branch = st.checkbox("Branch Transaction", value=False, help="Was transaction done at a bank branch?")

with col2:
    st.markdown('<p class="section-title">Customer & Time Information</p>', unsafe_allow_html=True)

    transaction_time = st.time_input(
        "Transaction Time",
        value=time(14, 30),
        help="Time when transaction occurred"
    )
    hour = transaction_time.hour

    day_of_week = st.selectbox(
        "Day of Week",
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    )
    day_mapping = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    day_numeric = day_mapping[day_of_week]

    age = st.slider("Customer Age", 18, 80, 35, help="Age of the customer")

    credit_score = st.slider("Credit Score", 300, 850, 650, help="Customer's credit score")

    customer_avg_amount = st.number_input(
        "Average Transaction ($)",
        min_value=0.0,
        value=120.0,
        step=10.0,
        help="Customer's average transaction amount"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# ANALYZE BUTTON
# ============================================
if st.button("Analyze Transaction", type="primary"):
    if model is not None and scaler is not None:

        with st.spinner('Analyzing transaction patterns...'):
            is_weekend = 1 if day_numeric >= 5 else 0
            is_night = 1 if hour >= 22 or hour <= 6 else 0
            is_high_amount = 1 if amount > 500 else 0
            amount_log = np.log1p(amount)
            is_credit_card = 1 if card_type == 'Credit' else 0
            amount_to_limit_ratio = amount / card_limit
            has_branch_flag = 1 if has_branch else 0

            customer_std_amount = customer_avg_amount * 0.3
            amount_deviation = abs(amount - customer_avg_amount)
            amount_deviation_ratio = amount_deviation / (customer_std_amount + 1)

            merchant_fraud_rates = {
                'Grocery': 0.01, 'Restaurant': 0.015, 'Gas Station': 0.012,
                'Online Retail': 0.035, 'Electronics': 0.040, 'Travel': 0.025,
                'Entertainment': 0.020, 'Healthcare': 0.008, 'Utilities': 0.005
            }
            merchant_category_fraud_rate = merchant_fraud_rates.get(merchant_category, 0.02)

            customer_total_transactions = 50

            features = {
                'amount': amount,
                'amount_log': amount_log,
                'is_high_amount': is_high_amount,
                'hour': hour,
                'day_of_week': day_numeric,
                'is_weekend': is_weekend,
                'is_night': is_night,
                'customer_avg_amount': customer_avg_amount,
                'customer_std_amount': customer_std_amount,
                'customer_total_transactions': customer_total_transactions,
                'amount_deviation': amount_deviation,
                'amount_deviation_ratio': amount_deviation_ratio,
                'merchant_category_fraud_rate': merchant_category_fraud_rate,
                'is_credit_card': is_credit_card,
                'amount_to_limit_ratio': amount_to_limit_ratio,
                'card_limit': card_limit,
                'has_branch': has_branch_flag,
                'age': age,
                'credit_score': credit_score
            }

            input_df = pd.DataFrame([features], columns=feature_names)
            input_scaled = scaler.transform(input_df)

            prediction = model.predict(input_scaled)[0]
            fraud_score = -model.decision_function(input_scaled)[0]
            fraud_probability = min(100, max(0, (fraud_score + 1) * 50))

        # ============================================
        # RESULTS
        # ============================================
        st.markdown("---")

        # Result card
        if prediction == -1:
            st.markdown(f'''
                <div class="result-card-fraud">
                    <div class="result-title">FRAUD RISK IDENTIFIED</div>
                    <div class="result-subtitle">Transaction requires immediate review</div>
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
                <div class="result-card-safe">
                    <div class="result-title">APPROVED FOR PROCESSING</div>
                    <div class="result-subtitle">No significant fraud indicators detected</div>
                </div>
            ''', unsafe_allow_html=True)

        # Metrics
        col1, col2, col3, col4 = st.columns(4, gap="medium")

        with col1:
            st.markdown(f'''
                <div class="metric-container">
                    <div class="metric-value">{fraud_probability:.0f}%</div>
                    <div class="metric-label">Risk Score</div>
                </div>
            ''', unsafe_allow_html=True)

        with col2:
            risk_level = "HIGH" if fraud_probability > 70 else "MEDIUM" if fraud_probability > 40 else "LOW"
            risk_color = "🔴" if fraud_probability > 70 else "🟡" if fraud_probability > 40 else "🟢"
            st.markdown(f'''
                <div class="metric-container">
                    <div class="metric-value" style="color: #e8edf5;">{risk_color}</div>
                    <div class="metric-label">{risk_level} RISK</div>
                </div>
            ''', unsafe_allow_html=True)

        with col3:
            st.markdown(f'''
                <div class="metric-container">
                    <div class="metric-value" style="color: #0052cc;">${amount:,.0f}</div>
                    <div class="metric-label">Amount</div>
                </div>
            ''', unsafe_allow_html=True)

        with col4:
            action = "REVIEW" if prediction == -1 else "APPROVE"
            st.markdown(f'''
                <div class="metric-container">
                    <div class="metric-value" style="color: #0052cc;">{action}</div>
                    <div class="metric-label">Action</div>
                </div>
            ''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_probability,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fraud Probability Score", 'font': {'size': 20, 'color': '#677487'}},
            number={'font': {'size': 48, 'color': '#1a2844'}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#cbd5e1"},
                'bar': {'color': "#0052cc", 'thickness': 0.35},
                'bgcolor': "#e8edf5",
                'borderwidth': 2,
                'bordercolor': "#cbd5e1",
                'steps': [
                    {'range': [0, 40], 'color': '#e7f7f0'},
                    {'range': [40, 70], 'color': '#fef3c7'},
                    {'range': [70, 100], 'color': '#fee8e8'}
                ],
                'threshold': {
                    'line': {'color': "#cc0000", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))

        fig.update_layout(
            height=320,
            paper_bgcolor="#ffffff",
            font={'color': "#677487", 'family': "system-ui, -apple-system, sans-serif", 'size': 12},
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Risk factors
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Risk Analysis Report</p>', unsafe_allow_html=True)

        risk_factors = []
        if is_high_amount:
            risk_factors.append("• High transaction amount exceeds typical thresholds")
        if is_night:
            risk_factors.append("• Transaction during unusual hours (10pm-6am)")
        if not has_branch:
            risk_factors.append("• No in-branch verification completed")
        if amount_deviation_ratio > 2:
            risk_factors.append("• Amount significantly deviates from customer profile")
        if merchant_category_fraud_rate > 0.025:
            risk_factors.append(f"• Merchant category flagged: {merchant_category}")
        if amount_to_limit_ratio > 0.8:
            risk_factors.append("• Transaction approaches card limit")
        if credit_score < 600:
            risk_factors.append("• Customer credit profile indicates higher risk")

        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"<p style='color: #8b2f2f; font-size: 0.95rem; margin: 0.7rem 0; font-weight: 500;'>{factor}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: #1d5f47; font-size: 1rem; font-weight: 500;'>✓ No significant risk factors identified in transaction profile</p>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.markdown('''
    <div class="custom-footer">
        <p><strong>Transaction Risk Assessment System</strong> | Enterprise Fraud Detection</p>
        <p style="font-size: 0.8rem; opacity: 0.8;">Powered by Advanced Machine Learning</p>
    </div>
''', unsafe_allow_html=True)