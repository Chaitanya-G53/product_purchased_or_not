import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="Customer Purchase Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD MODEL
# ==========================================================
@st.cache_resource
def load_model():
    return joblib.load("naive_purchased_model.pkl")

model = load_model()

# ==========================================================
# CUSTOM CSS
# ==========================================================
st.markdown("""
<style>

.main{
    background-color:#f7f9fc;
}

.title{
    font-size:42px;
    font-weight:bold;
    color:#0d6efd;
}

.subtitle{
    color:gray;
    font-size:18px;
}

.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 5px 15px rgba(0,0,0,0.08);
}

.result-success{
    background:#d4edda;
    color:#155724;
    padding:20px;
    border-radius:12px;
    font-size:22px;
    font-weight:bold;
}

.result-danger{
    background:#f8d7da;
    color:#721c24;
    padding:20px;
    border-radius:12px;
    font-size:22px;
    font-weight:bold;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:10px;
    background:#0d6efd;
    color:white;
    font-size:18px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================
st.markdown(
"""
<div class='title'>
🛒 Customer Purchase Prediction
</div>

<div class='subtitle'>
Predict whether a customer is likely to purchase a product using a
Gaussian Naive Bayes Machine Learning model.
</div>
""",
unsafe_allow_html=True
)

st.divider()

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.title("📊 Model Information")

st.sidebar.info("""
**Algorithm**

Gaussian Naive Bayes

**Input Features**

- Gender
- Age
- Estimated Salary

**Output**

Purchase Prediction
""")

# ==========================================================
# MAIN LAYOUT
# ==========================================================
left, right = st.columns([1,1])

with left:

    st.subheader("Customer Details")

    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    age = st.slider(
        "Age",
        min_value=18,
        max_value=70,
        value=30
    )

    salary = st.slider(
        "Estimated Salary (₹)",
        min_value=10000,
        max_value=150000,
        value=50000,
        step=1000
    )

    predict = st.button("Predict Purchase")

with right:

    st.subheader("Input Summary")

    st.metric("Gender", gender)
    st.metric("Age", age)
    st.metric("Salary", f"₹{salary:,}")

    chart = pd.DataFrame({
        "Feature":["Age","Salary (×1000)"],
        "Value":[age,salary/1000]
    })

    st.bar_chart(chart.set_index("Feature"))

# ==========================================================
# PREDICTION
# ==========================================================
if predict:

    # ------------------------------------------------------
    # IMPORTANT
    # ------------------------------------------------------
    # Male -> 1
    # Female -> 0
    #
    # If your training used the opposite encoding,
    # simply swap these values.
    # ------------------------------------------------------

    gender_value = 1 if gender == "Male" else 0

    features = np.array([[gender_value, age, salary]])

    prediction = model.predict(features)[0]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.markdown(
        """
        <div class='result-success'>
        ✅ Customer is likely to PURCHASE the product.
        </div>
        """,
        unsafe_allow_html=True
        )

        st.balloons()

    else:

        st.markdown(
        """
        <div class='result-danger'>
        ❌ Customer is NOT likely to PURCHASE the product.
        </div>
        """,
        unsafe_allow_html=True
        )

    # ======================================================
    # Probability
    # ======================================================

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(features)[0]

        purchase_prob = probability[1] * 100
        no_purchase_prob = probability[0] * 100

        st.subheader("Prediction Confidence")

        st.progress(float(purchase_prob/100))

        col1,col2 = st.columns(2)

        with col1:
            st.metric(
                "Purchase Probability",
                f"{purchase_prob:.2f}%"
            )

        with col2:
            st.metric(
                "No Purchase Probability",
                f"{no_purchase_prob:.2f}%"
            )

# ==========================================================
# FOOTER
# ==========================================================
st.divider()

st.caption(
"""
Developed with ❤️ using

• Streamlit

• Scikit-Learn

• Gaussian Naive Bayes
"""
)
