import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Load Model
# --------------------------------------------------
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Insurance Charges Predictor 💰",
    page_icon="💰",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* ---------- MAIN BACKGROUND ---------- */

.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(40,127,232,0.10) 0%, transparent 40%),
        radial-gradient(circle at 90% 10%, rgba(23,107,135,0.10) 0%, transparent 40%),
        linear-gradient(135deg, #f4f9ff 0%, #eef6ff 50%, #e8f2ff 100%) !important;
}

.main {
    background: transparent !important;
}

.block-container {
    padding-top: 4rem !important;
    padding-bottom: 2rem !important;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 3rem !important;
}

/* ---------- ALL TEXT ---------- */

p, label, span {
    color: #172b4d !important;
}

/* ---------- HEADER ---------- */

.hero {
    background: linear-gradient(120deg, #12355b 0%, #176b87 55%, #287fe8 100%);
    border-radius: 22px;
    padding: 38px 30px;
    text-align: center;
    box-shadow: 0 15px 40px rgba(18, 53, 91, 0.25);
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: "";
    position: absolute;
    top: -60px;
    right: -60px;
    width: 220px;
    height: 220px;
    background: rgba(255,255,255,0.08);
    border-radius: 50%;
}

.hero::after {
    content: "";
    position: absolute;
    bottom: -80px;
    left: -40px;
    width: 180px;
    height: 180px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
}

.main-title {
    color: #ffffff !important;
    font-family: 'Poppins', sans-serif;
    font-size: 44px;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
    position: relative;
}

.subtitle {
    color: #dceeff !important;
    font-size: 17px;
    font-weight: 500;
    margin-top: 8px;
    position: relative;
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    color: #ffffff !important;
    font-size: 13px;
    font-weight: 600;
    padding: 6px 16px;
    border-radius: 999px;
    margin-bottom: 14px;
    border: 1px solid rgba(255,255,255,0.3);
    position: relative;
}

/* ---------- SIDEBAR ---------- */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%) !important;
    border-right: 1px solid #dce6ef;
}

[data-testid="stSidebar"] * {
    color: #172b4d !important;
}

[data-testid="stSidebar"] h1 {
    color: #12355b !important;
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 22px;
}

.sidebar-tag {
    display: inline-block;
    background: #eaf4ff;
    color: #176b87 !important;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 999px;
    margin-bottom: 6px;
}

/* ---------- SECTION TITLES ---------- */

.section-title {
    color: #12355b !important;
    font-family: 'Poppins', sans-serif;
    font-size: 24px;
    font-weight: 800;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ---------- SLIDERS ---------- */

[data-testid="stSlider"] {
    color: #287fe8 !important;
}

[data-testid="stTickBar"] {
    display: none;
}

/* ---------- RADIO BUTTONS ---------- */

div[role="radiogroup"] label {
    background: #ffffff;
    border: 1.5px solid #dce6ef;
    border-radius: 10px;
    padding: 6px 14px !important;
    margin-right: 6px !important;
    transition: all 0.2s ease;
}

div[role="radiogroup"] label:hover {
    border-color: #287fe8;
    background: #f2f8ff;
}

/* ---------- REGION DROPDOWN ---------- */

[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #eaf4ff !important;
    border: 2px solid #287fe8 !important;
    border-radius: 10px !important;
    color: #12355b !important;
    min-height: 48px;
    box-shadow: 0 2px 8px rgba(40, 127, 232, 0.08);
}

[data-testid="stSelectbox"] div[data-baseweb="select"] span,
[data-testid="stSelectbox"] div[data-baseweb="select"] div {
    color: #12355b !important;
    font-weight: 600 !important;
    -webkit-text-fill-color: #12355b !important;
}

div[data-baseweb="select"] svg {
    fill: #287fe8 !important;
}

/* Dropdown menu */

ul[role="listbox"] {
    background-color: #ffffff !important;
    border: 1px solid #cbdff3 !important;
    border-radius: 10px !important;
}

li[role="option"] {
    background-color: #ffffff !important;
    color: #172b4d !important;
}

li[role="option"]:hover {
    background-color: #eef6ff !important;
    color: #287fe8 !important;
}

/* ---------- SUMMARY CARD ---------- */

.input-card {
    background: transparent !important;
    padding: 0;
    border-radius: 0;
    border: none;
    box-shadow: none;
}

/* ---------- BMI CARD ---------- */

.info-card {
    background: linear-gradient(135deg, #ffffff, #f0f8ff) !important;
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #cfe3f7;
    border-left: 5px solid #287fe8;
    box-shadow: 0 8px 24px rgba(18, 53, 91, 0.08);
    margin-top: 22px;
    color: #172b4d !important;
}

.info-card b {
    color: #12355b !important;
}

.bmi-pill {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 13px;
    margin-left: 6px;
}

/* ---------- PREDICTION CARD ---------- */

.prediction-card {
    background: linear-gradient(135deg, #12355b 0%, #176b87 55%, #287fe8 100%) !important;
    padding: 42px 30px;
    border-radius: 24px;
    text-align: center;
    box-shadow: 0 15px 40px rgba(40, 127, 232, 0.25);
    margin-top: 22px;
    position: relative;
    overflow: hidden;
}

.prediction-card::before {
    content: "";
    position: absolute;
    top: -50px;
    right: -50px;
    width: 160px;
    height: 160px;
    background: rgba(255,255,255,0.08);
    border-radius: 50%;
}

.prediction-label {
    color: #dceeff !important;
    font-size: 16px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
    position: relative;
}

.prediction-value {
    color: #ffffff !important;
    font-family: 'Poppins', sans-serif;
    font-size: 50px;
    font-weight: 800;
    margin-bottom: 14px;
    position: relative;
}

.prediction-risk {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    color: #ffffff !important;
    font-size: 15px;
    font-weight: 700;
    padding: 7px 20px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.35);
    position: relative;
}

/* ---------- BUTTON ---------- */

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #287fe8, #176b87) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px;
    padding: 14px;
    font-size: 17px;
    font-weight: 700;
    box-shadow: 0 8px 20px rgba(40, 127, 232, 0.28);
    transition: all 0.25s ease;
    letter-spacing: 0.3px;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #176b87, #12355b) !important;
    box-shadow: 0 10px 26px rgba(40, 127, 232, 0.38);
    transform: translateY(-2px);
}

.stButton > button:active {
    transform: translateY(0px);
}

/* ---------- METRIC CARDS ---------- */

[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #d7e5f2;
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 6px 20px rgba(18, 53, 91, 0.07);
    transition: transform 0.2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
}

[data-testid="stMetricLabel"] {
    color: #5b7088 !important;
}

[data-testid="stMetricValue"] {
    color: #12355b !important;
    font-family: 'Poppins', sans-serif;
}

/* ---------- INFO MESSAGE ---------- */

[data-testid="stAlert"] {
    background: #eef7ff !important;
    border: 1px solid #b9d9f7 !important;
    border-radius: 14px;
}

[data-testid="stAlert"] * {
    color: #12355b !important;
}

/* ---------- DATAFRAME ---------- */

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ---------- FOOTER ---------- */

.footer {
    text-align: center;
    color: #60758a !important;
    font-size: 13px;
    margin-top: 40px;
    padding: 18px 0 10px 0;
    border-top: 1px solid #dce6ef;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">⚡ Powered by XGBoost</div>
        <div class="main-title">💰 Insurance Charges Predictor</div>
        <div class="subtitle">AI-powered prediction of medical insurance charges based on patient profile</div>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.markdown('<div class="sidebar-tag">Patient Details</div>', unsafe_allow_html=True)
st.sidebar.markdown("<h1>🧾 Patient Information</h1>", unsafe_allow_html=True)

st.sidebar.markdown("---")

age = st.sidebar.slider(
    "🗓️ Age",
    18,
    100,
    30
)

sex = st.sidebar.radio(
    "⚧ Sex",
    ["Male", "Female"],
    horizontal=True
)

bmi = st.sidebar.slider(
    "⚖️ BMI",
    10.0,
    60.0,
    25.0,
    0.1
)

children = st.sidebar.slider(
    "👶 Number of Children",
    0,
    10,
    0
)

smoker = st.sidebar.radio(
    "🚬 Smoker",
    ["No", "Yes"],
    horizontal=True
)

region = st.sidebar.selectbox(
    "📍 Region",
    [
        "Northeast",
        "Northwest",
        "Southeast",
        "Southwest"
    ]
)


# --------------------------------------------------
# BMI CATEGORY
# --------------------------------------------------

if bmi <= 18.5:
    bmi_category = "Underweight"
    bmi_color = "#287fe8"
elif bmi <= 24.9:
    bmi_category = "Normal"
    bmi_color = "#2fb380"
elif bmi <= 29.9:
    bmi_category = "Overweight"
    bmi_color = "#e8a628"
else:
    bmi_category = "Obese"
    bmi_color = "#e5484d"


# --------------------------------------------------
# ENCODING
# --------------------------------------------------

streis_female = 1 if sex == "Female" else 0
is_smoker = 1 if smoker == "Yes" else 0

region_southeast = 1 if region == "Southeast" else 0
region_northwest = 1 if region == "Northwest" else 0

bmi_category_obese = 1 if bmi_category == "Obese" else 0


# --------------------------------------------------
# SCALE NUMERICAL VALUES
# --------------------------------------------------

scaled_values = scaler.transform(
    [[age, bmi, children]]
)

scaled_age = scaled_values[0][0]
scaled_bmi = scaled_values[0][1]
scaled_children = scaled_values[0][2]


# --------------------------------------------------
# MODEL INPUT
# --------------------------------------------------

input_data = pd.DataFrame({
    "age": [scaled_age],
    "is_female": [streis_female],
    "bmi": [scaled_bmi],
    "children": [scaled_children],
    "is_smoker": [is_smoker],
    "region_southeast": [region_southeast],
    "bmi_category_Obese": [bmi_category_obese],
    "region_northwest": [region_northwest]
})


# --------------------------------------------------
# MAIN CONTENT
# --------------------------------------------------

left, right = st.columns(
    [1, 1],
    gap="large"
)


# --------------------------------------------------
# PATIENT SUMMARY
# --------------------------------------------------

with left:

    st.markdown(
        '<div class="section-title">📋 Patient Summary</div>',
        unsafe_allow_html=True
    )

    summary = pd.DataFrame({
        "Parameter": [
            "Age",
            "Sex",
            "BMI",
            "BMI Category",
            "Children",
            "Smoker",
            "Region"
        ],
        "Value": [
            f"{age} years",
            sex.capitalize(),
            f"{bmi:.1f}",
            bmi_category,
            children,
            smoker.capitalize(),
            region.capitalize()
        ]
    })

    st.markdown(
        '<div class="input-card">',
        unsafe_allow_html=True
    )

    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="info-card">
            <b>BMI Category:</b>
            <span class="bmi-pill" style="background:{bmi_color}22; color:{bmi_color} !important;">{bmi_category}</span>
            <br><br>
            <b>BMI Value:</b> {bmi:.1f}
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

with right:

    st.markdown(
        '<div class="section-title">🔮 Prediction</div>',
        unsafe_allow_html=True
    )

    predict = st.button("🚀 Predict Insurance Charges")

    if predict:

        prediction = model.predict(input_data)[0]

        if prediction < 10000:
            risk = "🟢 Lower Cost Range"
        elif prediction < 20000:
            risk = "🟡 Moderate Cost Range"
        else:
            risk = "🔴 Higher Cost Range"

        # Prediction Card
        st.markdown(
            f"""
            <div class="prediction-card">
                <p class="prediction-label">Estimated Insurance Charges</p>
                <p class="prediction-value">${prediction:,.2f}</p>
                <span class="prediction-risk">{risk}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        # Metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Age", age)

        with col2:
            st.metric("BMI", f"{bmi:.1f}")

        with col3:
            st.metric("Children", children)

        st.info(
            "✨ Prediction generated using the trained XGBoost model."
        )

    else:

        st.markdown(
            """
            <div class="prediction-card">
                <p class="prediction-label">Estimated Insurance Charges</p>
                <p class="prediction-value">$ ———</p>
                <span class="prediction-risk" style="background:rgba(255,255,255,0.12);">
                    Enter patient information and predict
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    '<div class="footer">'
    '💰 Insurance Charges Prediction • XGBoost Machine Learning Model'
    '</div>',
    unsafe_allow_html=True
)
