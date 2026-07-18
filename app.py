import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "naive_purchased_model.pkl")

# --------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Purchase Intent Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------
# CUSTOM CSS — clean corporate / business-analytics theme
# --------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(160deg, #0b1220 0%, #101b30 45%, #0c1626 100%);
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f2c94c, #f2994a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #aab8cf;
        font-size: 1.02rem;
        margin-top: 2px;
        margin-bottom: 1.4rem;
    }
    .kpi-card {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(242,201,76,0.18);
        border-radius: 14px;
        padding: 16px 18px;
        text-align: center;
        box-shadow: 0 4px 18px rgba(0,0,0,0.3);
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #f2c94c;
        margin: 0;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #9fb0c9;
        margin: 0;
    }
    .glass-card {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.28);
        margin-bottom: 16px;
    }
    .result-card {
        background: rgba(255,255,255,0.05);
        border-radius: 18px;
        padding: 26px;
        text-align: center;
        border: 1px solid rgba(242,201,76,0.25);
    }
    .badge-buy {
        display:inline-block;
        padding: 8px 22px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
        background: rgba(39, 174, 96, 0.16);
        color: #2ecc71;
        border: 1px solid rgba(46,204,113,0.4);
    }
    .badge-nobuy {
        display:inline-block;
        padding: 8px 22px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
        background: rgba(235, 87, 87, 0.14);
        color: #eb5757;
        border: 1px solid rgba(235,87,87,0.4);
    }
    .insight-item {
        background: rgba(255,255,255,0.04);
        border-left: 3px solid #f2c94c;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.92rem;
        color: #dce6f2;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1220, #101c30);
        border-right: 1px solid rgba(242,201,76,0.12);
    }
    div.stButton > button {
        background: linear-gradient(90deg, #f2c94c, #f2994a);
        color: #12140f;
        font-weight: 800;
        border-radius: 10px;
        border: none;
        padding: 0.65em 1.4em;
        width: 100%;
        transition: 0.2s;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 18px rgba(242,201,76,0.5);
    }
    h1, h2, h3, h4 { color: #eef3fb; }
    .footer-note {
        font-size: 0.78rem;
        color: #6b7c94;
        text-align: center;
        margin-top: 18px;
        border-top: 1px solid rgba(255,255,255,0.08);
        padding-top: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------
# SVG GRAPHICS — clean business icons + gauge
# --------------------------------------------------------------------------------
def cart_svg(size=70):
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
      <circle cx="24" cy="54" r="4" fill="#f2c94c"/>
      <circle cx="46" cy="54" r="4" fill="#f2c94c"/>
      <path d="M6 8 h8 l6 30 h30 l6 -20 H18" fill="none" stroke="#f2994a" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
    </svg>
    """


def growth_svg(size_w=140, size_h=60, color="#2ecc71"):
    return f"""
    <svg width="{size_w}" height="{size_h}" viewBox="0 0 200 70" xmlns="http://www.w3.org/2000/svg">
      <polyline points="0,60 30,45 55,50 80,25 110,32 140,10 200,15"
                fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="140" cy="10" r="4" fill="{color}"/>
    </svg>
    """


def gauge_svg(pct, size=190, color="#2ecc71", track="rgba(255,255,255,0.08)"):
    pct = max(0.0, min(1.0, pct))
    r = 70
    circumference = 2 * np.pi * r
    dash = circumference * pct
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 180 180" xmlns="http://www.w3.org/2000/svg">
      <circle cx="90" cy="90" r="{r}" fill="none" stroke="{track}" stroke-width="14"/>
      <circle cx="90" cy="90" r="{r}" fill="none" stroke="{color}" stroke-width="14"
              stroke-linecap="round"
              stroke-dasharray="{dash:.1f} {circumference:.1f}"
              transform="rotate(-90 90 90)"/>
      <text x="90" y="82" text-anchor="middle" font-size="30" font-weight="800" fill="{color}">{pct*100:.0f}%</text>
      <text x="90" y="106" text-anchor="middle" font-size="11" fill="#aab8cf">Purchase Likelihood</text>
    </svg>
    """


# --------------------------------------------------------------------------------
# LOAD MODEL
# --------------------------------------------------------------------------------
@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.error(
            f"❌ Could not find `naive_purchased_model.pkl` at:\n\n`{path}`\n\n"
            "Make sure the file is committed to the same repo folder as `app.py`."
        )
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)


model = load_model()
FEATURES = list(getattr(model, "feature_names_in_", ["Gender", "Age", "EstimatedSalary"]))

# --------------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------------
st.markdown(
    f"""
    <div style="display:flex; align-items:center; justify-content:center; gap:16px;">
        {cart_svg(64)}
        <div>
            <div class="main-title">🛒 Purchase Intent Predictor</div>
            <div class="sub-title">Gaussian Naive Bayes model — will this customer buy?</div>
        </div>
        {growth_svg(140, 55)}
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------
# KPI STRIP (derived from the model's fitted statistics)
# --------------------------------------------------------------------------------
class_count = getattr(model, "class_count_", np.array([0, 0]))
total_samples = int(class_count.sum())
purchase_rate = class_count[1] / total_samples if total_samples else 0
theta = getattr(model, "theta_", None)

k1, k2, k3, k4 = st.columns(4)
for col, label, value in zip(
    (k1, k2, k3, k4),
    ("Training Samples", "Historical Purchase Rate", "Avg. Age (Buyers)", "Avg. Salary (Buyers)"),
    (
        f"{total_samples}",
        f"{purchase_rate*100:.1f}%",
        f"{theta[1][1]:.0f} yrs" if theta is not None else "N/A",
        f"${theta[1][2]:,.0f}" if theta is not None else "N/A",
    ),
):
    col.markdown(
        f"""<div class="kpi-card"><p class="kpi-value">{value}</p><p class="kpi-label">{label}</p></div>""",
        unsafe_allow_html=True,
    )

st.write("")

# --------------------------------------------------------------------------------
# SIDEBAR — CUSTOMER INPUTS
# --------------------------------------------------------------------------------
st.sidebar.markdown(f"<div style='text-align:center'>{cart_svg(56)}</div>", unsafe_allow_html=True)
st.sidebar.header("👤 Customer Profile")
st.sidebar.caption("Enter the prospect's details")

gender = st.sidebar.selectbox("⚧ Gender", ["Female", "Male"])
age = st.sidebar.slider("🎂 Age", 18, 70, 35, 1)
salary = st.sidebar.slider("💵 Estimated Salary ($)", 10000, 200000, 60000, 1000)

st.sidebar.markdown("---")
predict_clicked = st.sidebar.button("📈 Predict Purchase")
st.sidebar.markdown("---")
st.sidebar.info(
    f"**Model:** Gaussian Naive Bayes\n\n"
    f"**Features:** {', '.join(FEATURES)}\n\n"
    f"**Training samples:** {total_samples}"
)

gender_encoded = 1 if gender == "Male" else 0
raw = {"Gender": gender_encoded, "Age": age, "EstimatedSalary": salary}
input_df = pd.DataFrame([[raw[f] for f in FEATURES]], columns=FEATURES)

# --------------------------------------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------------------------------------
col_left, col_right = st.columns([1.05, 1])

with col_left:
    st.markdown("#### 📋 Customer Snapshot")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    snapshot = pd.DataFrame({
        "Field": ["Gender", "Age", "Estimated Salary"],
        "Value": [gender, f"{age} yrs", f"${salary:,.0f}"],
    })
    st.dataframe(snapshot, hide_index=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### 📊 How This Customer Compares")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if theta is not None:
        compare_df = pd.DataFrame({
            "Segment": ["Non-Buyers (avg)", "This Customer", "Buyers (avg)"],
            "Age": [theta[0][1], age, theta[1][1]],
        }).set_index("Segment")
        st.caption("Age vs. class averages")
        st.bar_chart(compare_df, use_container_width=True)

        compare_salary_df = pd.DataFrame({
            "Segment": ["Non-Buyers (avg)", "This Customer", "Buyers (avg)"],
            "Salary": [theta[0][2], salary, theta[1][2]],
        }).set_index("Segment")
        st.caption("Salary vs. class averages")
        st.bar_chart(compare_salary_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown("#### 🔮 Prediction")

    if predict_clicked or "last_pred" not in st.session_state:
        pred_class = int(model.predict(input_df.values)[0])
        proba = model.predict_proba(input_df.values)[0]
        buy_prob = float(proba[list(model.classes_).index(1)])
        st.session_state["last_pred"] = pred_class
        st.session_state["last_prob"] = buy_prob
    else:
        pred_class = st.session_state["last_pred"]
        buy_prob = st.session_state["last_prob"]

    gauge_color = "#2ecc71" if buy_prob >= 0.5 else "#eb5757"
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    if pred_class == 1:
        st.markdown('<span class="badge-buy">✅ Likely to Purchase</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-nobuy">❌ Unlikely to Purchase</span>', unsafe_allow_html=True)
    st.markdown(f"<div style='margin-top:14px;'>{gauge_svg(buy_prob, color=gauge_color)}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    proba_df = pd.DataFrame({
        "Outcome": ["Won't Buy", "Will Buy"],
        "Probability": [1 - buy_prob, buy_prob],
    }).set_index("Outcome")
    st.bar_chart(proba_df, use_container_width=True)

    st.markdown("##### 💡 Business Recommendation")
    if buy_prob >= 0.7:
        st.success("🟢 High-intent prospect — prioritize for direct outreach, personalized offers, or premium product tiers.")
    elif buy_prob >= 0.4:
        st.warning("🟡 Moderate intent — consider retargeting ads or a limited-time discount to nudge conversion.")
    else:
        st.error("🔴 Low intent — deprioritize for paid outreach; nurture via general awareness campaigns instead.")

# --------------------------------------------------------------------------------
# MODEL INSIGHTS SECTION
# --------------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🧠 Model Insights")
st.caption("Derived from the Naive Bayes model's fitted class statistics (not live data).")

ic1, ic2 = st.columns(2)
with ic1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("**Class Distribution (Training Data)**")
    dist_df = pd.DataFrame({
        "Class": ["Did Not Purchase", "Purchased"],
        "Count": [int(class_count[0]), int(class_count[1])],
    }).set_index("Class")
    st.bar_chart(dist_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with ic2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("**Key Insights**")
    insights = [
        f"Buyers in the training data are on average **{theta[1][1]:.0f} years old** vs **{theta[0][1]:.0f} years** for non-buyers — age is a strong signal.",
        f"Buyers earn **${theta[1][2]:,.0f}** on average vs **${theta[0][2]:,.0f}** for non-buyers — higher income correlates with purchase likelihood.",
        f"The base purchase rate in this dataset is **{purchase_rate*100:.1f}%** — use this as a benchmark when evaluating campaign lift.",
    ]
    for text in insights:
        st.markdown(f'<div class="insight-item">{text}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-note">
    ℹ️ Predictions are statistical estimates based on historical patterns and should be used as one input
    among several in sales/marketing decisions, not as a sole determinant.
    </div>
    <p style='text-align:center; color:#5f6f88; font-size:0.8rem; margin-top:6px;'>
    Built with Streamlit &nbsp;|&nbsp; Model: GaussianNB (scikit-learn) &nbsp;|&nbsp; Purchase Intent Predictor
    </p>
    """,
    unsafe_allow_html=True,
)
