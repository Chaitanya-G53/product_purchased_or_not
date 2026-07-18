import streamlit as st
import joblib
import numpy as np

# -------------------------
# PAGE CONFIGURATION
# -------------------------
st.set_page_config(
    page_title="Customer Purchase Prediction",
    page_icon="🛒",
    layout="wide",
)

# -------------------------
# LOAD MODEL
# -------------------------
model = joblib.load("naive_purchased_model.pkl")


# -------------------------
# CUSTOM CSS
# -------------------------
st.markdown("""
<style>

.main{
background-color:#f7f9fc;
}

.big-title{
font-size:45px;
font-weight:bold;
color:#0E4C92;
text-align:center;
}

.sub-title{
font-size:18px;
text-align:center;
color:gray;
}

.metric-box{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 3px 10px rgba(0,0,0,0.1);
text-align:center;
}

.stButton>button{
background:#0E4C92;
color:white;
height:55px;
width:100%;
border-radius:10px;
font-size:20px;
font-weight:bold;
}

.result-success{
background:#d4edda;
padding:20px;
border-radius:15px;
font-size:24px;
color:#155724;
font-weight:bold;
}

.result-danger{
background:#f8d7da;
padding:20px;
border-radius:15px;
font-size:24px;
color:#721c24;
font-weight:bold;
}

</style>
""", unsafe_allow_html=True)


# -------------------------
# HEADER
# -------------------------

st.markdown("<div class='big-title'>🛍 Customer Purchase Prediction</div>", unsafe_allow_html=True)

st.markdown("<div class='sub-title'>Machine Learning Web Application | Gaussian Naive Bayes</div>", unsafe_allow_html=True)

st.divider()

# -------------------------
# DASHBOARD
# -------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Algorithm", "Gaussian NB")

with col2:
    st.metric("Input Features", "Age, Salary")

with col3:
    st.metric("Prediction", "Purchase / Not Purchase")

st.divider()

# -------------------------
# MAIN LAYOUT
# -------------------------

left, right = st.columns([1,1])

with left:

    st.subheader("📋 Customer Information")

    age = st.slider(
        "Age",
        18,
        70,
        30
    )

    salary = st.slider(
        "Estimated Salary",
        10000,
        150000,
        50000,
        step=1000
    )

    predict = st.button("🔍 Predict Purchase")

with right:

    st.subheader("📊 Input Summary")

    st.info(f"""
    **Age:** {age}

    **Estimated Salary:** ₹{salary:,}
    """)

    st.progress((age-18)/(70-18))

    st.bar_chart({
        "Value":[age,salary/1000]
    })

# -------------------------
# PREDICTION
# -------------------------

if predict:

    features = np.array([[age, salary]])

    prediction = model.predict(features)[0]

    if prediction == 1:

        st.markdown(
        """
        <div class="result-success">
        ✅ Customer is likely to PURCHASE the product.
        </div>
        """,
        unsafe_allow_html=True)

        st.balloons()

    else:

        st.markdown(
        """
        <div class="result-danger">
        ❌ Customer is NOT likely to purchase the product.
        </div>
        """,
        unsafe_allow_html=True)

# -------------------------
# FOOTER
# -------------------------

st.divider()

st.caption(
"Developed using Streamlit • Scikit-learn • Gaussian Naive Bayes"
)
