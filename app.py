import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="🏦",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("loan_model.pkl")

model = load_model()

# ---------------------------------------------------------
# CSS - NO HTML REQUIRED
# ---------------------------------------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #07111f 0%,
        #0b1b32 50%,
        #102a43 100%
    );
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

/* Headings */
h1, h2, h3 {
    color: white !important;
}

/* Normal text */
p, label {
    color: #dbeafe !important;
}

/* Header */
.header-box {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1d4ed8,
        #0284c7
    );
    padding: 35px;
    border-radius: 25px;
    text-align: center;
    margin-bottom: 30px;
}

.header-title {
    font-size: 42px;
    font-weight: 800;
    color: white;
}

.header-subtitle {
    font-size: 17px;
    color: #dbeafe;
}

/* Cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(15, 23, 42, 0.85);
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.2);
}

/* Inputs */
div[data-baseweb="select"] > div {
    background-color: #111f35;
    border-radius: 10px;
}

/* Number inputs */
div[data-baseweb="input"] {
    background-color: #111f35;
    border-radius: 10px;
}

input {
    color: white !important;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 60px;
    border-radius: 15px;
    border: none;
    background: linear-gradient(
        135deg,
        #2563eb,
        #0284c7
    );
    color: white;
    font-size: 19px;
    font-weight: bold;
}

.stButton > button:hover {
    background: linear-gradient(
        135deg,
        #1d4ed8,
        #0369a1
    );
}

/* Success */
.success-box {
    background: #064e3b;
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    border: 2px solid #22c55e;
}

/* Rejected */
.reject-box {
    background: #7f1d1d;
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    border: 2px solid #ef4444;
}

/* Footer */
.footer {
    text-align: center;
    padding: 30px;
    margin-top: 40px;
    color: #94a3b8;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="header-box">
        <div style="font-size:50px;">🏦</div>
        <div class="header-title">
            Loan Eligibility Predictor
        </div>
        <div class="header-subtitle">
            Machine Learning Based Loan Application Assessment
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# APPLICANT INFORMATION
# ---------------------------------------------------------
st.header("👤 Applicant Information")

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    with col2:
        married = st.selectbox(
            "Married",
            ["Yes", "No"]
        )

    with col3:
        dependents = st.selectbox(
            "Dependents",
            ["0", "1", "2", "3+"]
        )


# ---------------------------------------------------------
# EDUCATION & EMPLOYMENT
# ---------------------------------------------------------
st.header("💼 Education & Employment")

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:
        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

    with col2:
        self_employed = st.selectbox(
            "Self Employed",
            ["No", "Yes"]
        )

    with col3:
        credit_history = st.selectbox(
            "Credit History",
            ["Good", "Poor"]
        )


# ---------------------------------------------------------
# PROPERTY
# ---------------------------------------------------------
st.header("🏠 Property Information")

with st.container(border=True):

    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )


# ---------------------------------------------------------
# FINANCIAL INFORMATION
# ---------------------------------------------------------
st.header("💰 Financial Information")

with st.container(border=True):

    col1, col2 = st.columns(2)

    with col1:
        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

    with col2:
        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0.0,
            value=0.0,
            step=500.0
        )

    col1, col2 = st.columns(2)

    with col1:
        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=150.0,
            step=10.0
        )

    with col2:
        loan_amount_term = st.number_input(
            "Loan Term (Months)",
            min_value=1.0,
            value=360.0,
            step=12.0
        )


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
st.write("")

if st.button("🔍 CHECK LOAN ELIGIBILITY"):

    # Encoding

    gender_value = 1 if gender == "Male" else 0

    married_value = 1 if married == "Yes" else 0

    if dependents == "0":
        dependents_value = 0
    elif dependents == "1":
        dependents_value = 1
    elif dependents == "2":
        dependents_value = 2
    else:
        dependents_value = 3

    education_value = 0 if education == "Graduate" else 1

    self_employed_value = 1 if self_employed == "Yes" else 0

    property_area_value = {
        "Rural": 0,
        "Semiurban": 1,
        "Urban": 2
    }[property_area]

    credit_history_value = (
        1.0 if credit_history == "Good" else 0.0
    )

    # Feature engineering

    total_income = (
        applicant_income +
        coapplicant_income
    )

    applicant_income_log = np.log(
        applicant_income + 1
    )

    loan_amount_log = np.log(
        loan_amount + 1
    )

    loan_amount_term_log = np.log(
        loan_amount_term + 1
    )

    total_income_log = np.log(
        total_income + 1
    )

    # Input data

    input_data = pd.DataFrame([[
        gender_value,
        married_value,
        dependents_value,
        education_value,
        self_employed_value,
        credit_history_value,
        property_area_value,
        applicant_income_log,
        loan_amount_log,
        loan_amount_term_log,
        total_income_log
    ]], columns=[
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Credit_History",
        "Property_Area",
        "ApplicantIncomelog",
        "LoanAmountlog",
        "Loan_Amount_Termlog",
        "Total_Incomelog"
    ])

    # Prediction

    prediction = model.predict(input_data)[0]

    # Confidence

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        confidence = max(probabilities) * 100

    # Result

    if prediction == 1:

        st.markdown(
            """
            <div class="success-box">
                <div style="font-size:55px;">✅</div>
                <h1>Loan Approved</h1>
                <p>
                    The machine learning model predicts that
                    this application is eligible for approval.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="reject-box">
                <div style="font-size:55px;">❌</div>
                <h1>Loan Rejected</h1>
                <p>
                    The machine learning model predicts that
                    this application may not be eligible for approval.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    if confidence is not None:
        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )


# ---------------------------------------------------------
# DISCLAIMER
# ---------------------------------------------------------
st.info(
    "This application is a machine learning project developed "
    "for demonstration purposes. The prediction should not be "
    "considered an actual financial or lending decision."
)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        <b>Loan Eligibility Predictor</b><br><br>
        Developed by <b>Krishna Vemula</b><br>
        Contact: krishnavemula7788@gmail.com<br><br>
        Python • Machine Learning • Random Forest
    </div>
    """,
    unsafe_allow_html=True
)