# ── FairLoan Web App ───────────────────────────────────────────
# A Gender-Aware Credit Risk Prediction System
# Built with Streamlit | Phase 1
# PES University | SDG 1 & SDG 5

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Page Configuration (must be first Streamlit call) ──────────
st.set_page_config(
    page_title="FairLoan — Credit Risk Predictor",
    page_icon="⚖️",
    layout="centered"
)

# ── Custom CSS (Modern, Clear Fintech Theme) ───────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Background Reset */
    .stApp {
        background-color: #F8FAFC !important;
        font-family: 'Inter', sans-serif;
    }

    /* Clean container spacing */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 5rem;
        max-width: 800px;
    }

    /* Hide Streamlit default headers/footers */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── App Header ── */
    .app-header {
        text-align: left;
        padding-bottom: 2rem;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 2rem;
    }
    .app-title {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.2rem;
        color: #0F172A;
        letter-spacing: -0.03em;
        margin: 0;
    }
    .app-subtitle {
        font-size: 1rem;
        color: #64748B;
        margin-top: 0.2rem;
    }
    .sdg-container {
        margin-top: 0.75rem;
    }
    .sdg-badge {
        display: inline-block;
        background: #F1F5F9;
        color: #475569;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        border: 1px solid #E2E8F0;
        margin-right: 0.5rem;
    }

    /* ── Form Section Cards ── */
    .section-card {
        background: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .section-title {
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 0.5rem;
    }

    /* Input Field & Radio Selection Labels Fix */
    label, .stSelectbox label, .stNumberInput label, .stRadio label {
        color: #1E293B !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }
    
    /* Crucial Fix: Forces the Model options text to be completely visible */
    [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
        color: #1E293B !important;
        font-weight: 400 !important;
    }

    /* ── Fix for Indian Context Expander Tables ── */
    .stExpander div {
        color: #1E293B !important;
    }
    .stExpander table {
        width: 100%;
        color: #1E293B !important;
        border-collapse: collapse;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .stExpander th {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        font-weight: 600 !important;
        padding: 8px 12px !important;
        text-align: left;
        border: 1px solid #E2E8F0 !important;
    }
    .stExpander td {
        padding: 8px 12px !important;
        border: 1px solid #E2E8F0 !important;
        color: #334155 !important;
        background-color: #FFFFFF !important;
    }

    /* ── Predict Button ── */
    .stButton > button {
        background-color: #0F172A !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);
        margin-top: 1rem;
    }
    .stButton > button:hover {
        background-color: #1E293B !important;
        transform: translateY(-0.5px);
    }

    /* ── Result Cards ── */
    .result-eligible {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #10B981;
        border-radius: 12px;
        padding: 2rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .result-not-eligible {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #EF4444;
        border-radius: 12px;
        padding: 2rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .result-verdict {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-top: 0.25rem;
    }
    .result-confidence {
        font-size: 0.88rem;
        color: #64748B;
        margin-top: 0.6rem;
    }

    /* ── Information Boxes ── */
    .fairness-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1.5rem;
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.6;
    }
    .bias-alert {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-radius: 8px;
        padding: 1.2rem;
        font-size: 0.88rem;
        color: #78350F;
        line-height: 1.6;
        margin-bottom: 2rem;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        font-size: 0.78rem;
        color: #94A3B8;
        margin-top: 4rem;
        padding-top: 1.5rem;
        border-top: 1px solid #E2E8F0;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Models ────────────────────────────────────────────────
with open('models/lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)
with open('models/dt_model.pkl', 'rb') as f:
    dt_model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ── Header Section with SVG Logo ───────────────────────────────
st.markdown("""
<div class="app-header">
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 0.4rem;">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="3" y="14" width="4" height="6" rx="1" fill="#4F46E5" />
            <rect x="10" y="8" width="4" height="12" rx="1" fill="#0F172A" />
            <rect x="17" y="4" width="4" height="16" rx="1" fill="#10B981" />
            <path d="M3 20H21" stroke="#E2E8F0" stroke-width="2" stroke-linecap="round"/>
        </svg>
        <div class="app-title">
            <span style="font-weight: 800; color: #0F172A;">Fair</span><span style="font-weight: 300; color: #4F46E5;">Loan</span>
        </div>
    </div>
    <div class="app-subtitle">Gender-Aware Credit Risk Prediction System</div>
    <div class="sdg-container">
        <span class="sdg-badge">SDG 1 — No Poverty</span>
        <span class="sdg-badge">SDG 5 — Gender Equality</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Bias Alert ─────────────────────────────────────────────────
st.markdown("""
<div class="bias-alert">
    📊 <strong>Gender Disparity in Data:</strong> Only <strong>10.9% of women</strong> 
    vs <strong>30.4% of men</strong> fall in the high-income group in this dataset. 
    FairLoan surfaces this disparity to promote transparent and equitable lending decisions.
</div>
""", unsafe_allow_html=True)

# ── Indian Context Guide ───────────────────────────────────────
with st.expander("🔴If not sure what these fields mean? Click for Indian equivalents"):
    st.markdown("""
    ### 🏢 Work Class
    | Field Value | Indian Equivalent |
    | :--- | :--- |
    | Private | Private company — TCS, Infosys, any pvt firm |
    | Self-emp-not-inc | Freelancer, street vendor, unregistered business |
    | Self-emp-inc | Registered business owner (Pvt Ltd) |
    | Federal-gov | Central Government — IAS, IPS, Central PSU |
    | Local-gov | Municipal body — BBMP, local PSU |
    | State-gov | State Government department employee |
    | Without-pay | Unpaid family worker / volunteer |

    ### 💼 Occupation
    | Field Value | Indian Equivalent |
    | :--- | :--- |
    | Exec-managerial | Manager / Director / CEO level |
    | Prof-specialty | Doctor / Engineer / CA / Lawyer |
    | Tech-support | IT helpdesk / Technical support |
    | Adm-clerical | Office clerk / Admin / Data entry |
    | Sales | Sales executive / Shop owner |
    | Craft-repair | Electrician / Mechanic / Carpenter |
    | Transport-moving | Driver / Delivery / Logistics |
    | Farming-fishing | Farmer / Agricultural worker |
    | Machine-op-inspct | Factory worker / Quality inspector |
    | Handlers-cleaners | Manual labour / Sanitation |
    | Priv-house-serv | Domestic help |
    | Protective-serv | Police / Security guard |
    | Armed-Forces | Army / Navy / Air Force |

    ### 🎓 Education
    | Field Value | Indian Equivalent |
    | :--- | :--- |
    | Preschool / 1st-4th | Primary school (incomplete) |
    | 5th–8th | Middle school |
    | 9th–12th | Secondary / PUC level |
    | HS-grad | 12th pass / PUC completed |
    | Some-college | Dropped out of degree |
    | Assoc-voc / Assoc-acdm | Diploma / ITI |
    | Bachelors | B.E. / BCA / B.Com / BA |
    | Masters | M.Tech / MCA / MBA |
    | Prof-school | CA / MBBS / Law |
    | Doctorate | PhD |

    ### 💰 Capital Gain / Capital Loss
    Money earned or lost from investments — like profit from selling shares on NSE/BSE or selling property. 
    Most salaried employees enter 0 here. A high capital gain strongly signals a wealthy applicant.
    """)

# ── SECTION: Personal Info ─────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title">Personal Information</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=17, max_value=90, value=30)
with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION: Employment ────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title">Employment Details</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    workclass = st.selectbox("Work Class", [
        "Private", "Self-emp-not-inc", "Self-emp-inc",
        "Federal-gov", "Local-gov", "State-gov", "Without-pay"
    ])
with col4:
    occupation = st.selectbox("Occupation", [
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
    ])
hours_per_week = st.number_input("Hours Worked Per Week  (Standard = 40 hrs)", min_value=1, max_value=99, value=40)
st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION: Education & Background ───────────────────────────
st.markdown('<div class="section-card"><div class="section-title">Education & Background</div>', unsafe_allow_html=True)
col5, col6 = st.columns(2)
with col5:
    education = st.selectbox("Education Level", [
        "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th",
        "11th", "12th", "HS-grad", "Some-college", "Assoc-voc",
        "Assoc-acdm", "Bachelors", "Prof-school", "Masters", "Doctorate"
    ])
with col6:
    marital_status = st.selectbox("Marital Status", [
        "Married-civ-spouse", "Divorced", "Never-married",
        "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ])
relationship = st.selectbox("Relationship Role in Household", [
    "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
])
st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION: Financial ─────────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title">Financial Information</div>', unsafe_allow_html=True)
col7, col8 = st.columns(2)
with col7:
    capital_gain = st.number_input("Capital Gain ($)  — 0 if none", min_value=0, max_value=99999, value=0)
with col8:
    capital_loss = st.number_input("Capital Loss ($)  — 0 if none", min_value=0, max_value=99999, value=0)
st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION: Model Selector ────────────────────────────────────
st.markdown('<div class="section-card"><div class="section-title">Prediction Model</div>', unsafe_allow_html=True)
model_choice = st.radio(
    "Choose which ML model to use for prediction:",
    ["Decision Tree (85.96% accuracy)", "Logistic Regression (85.18% accuracy)"],
    horizontal=True
)
st.markdown('</div>', unsafe_allow_html=True)

# ── Education num map ──────────────────────────────────────────
education_num_map = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4,
    "9th": 5, "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9,
    "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12,
    "Bachelors": 13, "Prof-school": 14, "Masters": 15, "Doctorate": 16
}

# ── Predict Button ─────────────────────────────────────────────
predict_clicked = st.button("🔍  Predict Loan Eligibility")

if predict_clicked:

    input_data = {
        'age': [age],
        'educational-num': [education_num_map[education]],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'workclass_Federal-gov': [1 if workclass == 'Federal-gov' else 0],
        'workclass_Local-gov': [1 if workclass == 'Local-gov' else 0],
        'workclass_Never-worked': [0],
        'workclass_Private': [1 if workclass == 'Private' else 0],
        'workclass_Self-emp-inc': [1 if workclass == 'Self-emp-inc' else 0],
        'workclass_Self-emp-not-inc': [1 if workclass == 'Self-emp-not-inc' else 0],
        'workclass_State-gov': [1 if workclass == 'State-gov' else 0],
        'workclass_Without-pay': [1 if workclass == 'Without-pay' else 0],
        # Education
        'education_11th': [1 if education == '11th' else 0],
        'education_12th': [1 if education == '12th' else 0],
        'education_1st-4th': [1 if education == '1st-4th' else 0],
        'education_5th-6th': [1 if education == '5th-6th' else 0],
        'education_7th-8th': [1 if education == '7th-8th' else 0],
        'education_9th': [1 if education == '9th' else 0],
        'education_Assoc-acdm': [1 if education == 'Assoc-acdm' else 0],
        'education_Assoc-voc': [1 if education == 'Assoc-voc' else 0],
        'education_Bachelors': [1 if education == 'Bachelors' else 0],
        'education_Doctorate': [1 if education == 'Doctorate' else 0],
        'education_HS-grad': [1 if education == 'HS-grad' else 0],
        'education_Masters': [1 if education == 'Masters' else 0],
        'education_Preschool': [1 if education == 'Preschool' else 0],
        'education_Prof-school': [1 if education == 'Prof-school' else 0],
        'education_Some-college': [1 if education == 'Some-college' else 0],
        # Marital status
        'marital-status_Married-AF-spouse': [1 if marital_status == 'Married-AF-spouse' else 0],
        'marital-status_Married-civ-spouse': [1 if marital_status == 'Married-civ-spouse' else 0],
        'marital-status_Married-spouse-absent': [1 if marital_status == 'Married-spouse-absent' else 0],
        'marital-status_Never-married': [1 if marital_status == 'Never-married' else 0],
        'marital-status_Separated': [1 if marital_status == 'Separated' else 0],
        'marital-status_Widowed': [1 if marital_status == 'Widowed' else 0],
        # Occupation
        'occupation_Armed-Forces': [1 if occupation == 'Armed-Forces' else 0],
        'occupation_Craft-repair': [1 if occupation == 'Craft-repair' else 0],
        'occupation_Exec-managerial': [1 if occupation == 'Exec-managerial' else 0],
        'occupation_Farming-fishing': [1 if occupation == 'Farming-fishing' else 0],
        'occupation_Handlers-cleaners': [1 if occupation == 'Handlers-cleaners' else 0],
        'occupation_Machine-op-inspct': [1 if occupation == 'Machine-op-inspct' else 0],
        'occupation_Other-service': [1 if occupation == 'Other-service' else 0],
        'occupation_Priv-house-serv': [1 if occupation == 'Priv-house-serv' else 0],
        'occupation_Prof-specialty': [1 if occupation == 'Prof-specialty' else 0],
        'occupation_Protective-serv': [1 if occupation == 'Protective-serv' else 0],
        'occupation_Sales': [1 if occupation == 'Sales' else 0],
        'occupation_Tech-support': [1 if occupation == 'Tech-support' else 0],
        'occupation_Transport-moving': [1 if occupation == 'Transport-moving' else 0],
        # Relationship
        'relationship_Not-in-family': [1 if relationship == 'Not-in-family' else 0],
        'relationship_Other-relative': [1 if relationship == 'Other-relative' else 0],
        'relationship_Own-child': [1 if relationship == 'Own-child' else 0],
        'relationship_Unmarried': [1 if relationship == 'Unmarried' else 0],
        'relationship_Wife': [1 if relationship == 'Wife' else 0],
        # Race
        'race_Asian-Pac-Islander': [0],
        'race_Black': [0],
        'race_Other': [0],
        'race_White': [0],
        # Gender
        'gender_Male': [1 if gender == 'Male' else 0],
    }

    input_df = pd.DataFrame(input_data)
    input_df = input_df[dt_model.feature_names_in_]

    if "Decision Tree" in model_choice:
        prediction = dt_model.predict(input_df)[0]
        probability = dt_model.predict_proba(input_df)[0]
        model_used = "Decision Tree"
    else:
        input_scaled = scaler.transform(input_df)
        prediction = lr_model.predict(input_scaled)[0]
        probability = lr_model.predict_proba(input_scaled)[0]
        model_used = "Logistic Regression"

    # ── Result ─────────────────────────────────────────────────
    if prediction == 1:
        confidence = round(probability[1] * 100, 1)
        st.markdown(f"""
        <div class="result-eligible">
            <div class="result-verdict" style="color:#10B981 !important;">Eligible</div>
            <div style="font-size:0.92rem; color:#475569; margin-top:0.3rem;">
                This applicant is likely to earn <strong>&gt;$50K</strong>
            </div>
            <div class="result-confidence">
                Confidence: <strong>{confidence}%</strong> &nbsp;|&nbsp; Model: {model_used}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        confidence = round(probability[0] * 100, 1)
        st.markdown(f"""
        <div class="result-not-eligible">
            <div class="result-verdict" style="color:#EF4444 !important;">Not Eligible</div>
            <div style="font-size:0.92rem; color:#475569; margin-top:0.3rem;">
                This applicant is likely to earn <strong>≤$50K</strong>
            </div>
            <div class="result-confidence">
                Confidence: <strong>{confidence}%</strong> &nbsp;|&nbsp; Model: {model_used}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Gender Fairness Note ───────────────────────────────────
    if gender == "Female":
        st.markdown("""
        <div class="fairness-box">
            👩 <strong>Gender Fairness Note:</strong> This applicant is female. 
            Our dataset shows only <strong>10.9% of women</strong> fall in the 
            high-income group vs <strong>30.4% of men</strong>. This prediction 
            has been made purely on financial and occupational merit — 
            gender has not been used to disadvantage this applicant.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="fairness-box">
            👨 <strong>Gender Fairness Note:</strong> Historical data shows men are 
            statistically overrepresented in the high-income group (30.4% vs 10.9% for women). 
            FairLoan is designed to ensure women applicants receive equally fair evaluation.
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    FairLoan &nbsp;|&nbsp; PES University MCA Project &nbsp;|&nbsp; 
    SDG 1 — No Poverty &nbsp;|&nbsp; SDG 5 — Gender Equality<br>
    S D Vachan Kariappa &nbsp;|&nbsp; PES1PG25CA326
</div>
""", unsafe_allow_html=True)