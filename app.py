import streamlit as st
import pandas as pd
import pickle
import random
from sample_data import SAMPLE_CASES

st.set_page_config(page_title="Insurance Fraud Detection", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.main-header{font-size:3rem;font-weight:bold;text-align:center;color:#1f77b4}
.sub-header{text-align:center;color:gray;margin-bottom:30px}
.fraud-alert{background:linear-gradient(135deg,#ffe2e2,#ffc4c4);border:2px solid #ef4444;padding:20px;border-radius:14px;font-size:28px;text-align:center;color:#991b1b;font-weight:bold;box-shadow:0 8px 20px rgba(239,68,68,.18)}
.legitimate-alert{background:linear-gradient(135deg,#ddffe6,#bbf7d0);border:2px solid #22c55e;padding:20px;border-radius:14px;font-size:28px;text-align:center;color:#166534;font-weight:bold;box-shadow:0 8px 20px rgba(34,197,94,.18)}
.suspicious-alert{background:linear-gradient(135deg,#fff4dd,#ffe4b5);border:2px solid #f59e0b;padding:20px;border-radius:14px;font-size:28px;text-align:center;color:#92400e;font-weight:bold;box-shadow:0 8px 20px rgba(245,158,11,.18)}
.prob-card{padding:18px;border-radius:12px;text-align:center;font-weight:600}
.prob-title{font-size:16px;opacity:.9;margin-bottom:8px}
.prob-value{font-size:30px;font-weight:800}
.prob-legit{background:#ecfdf3;border:1px solid #86efac;color:#166534}
.prob-fraud{background:#fef2f2;border:1px solid #fca5a5;color:#991b1b}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛡️ Insurance Fraud Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning Powered Claim Analysis</div>', unsafe_allow_html=True)


@st.cache_resource
def load_model():
    with open("fraud_model.pkl","rb") as f:
        return pickle.load(f)

model_data = load_model()
model = model_data["model"]
scaler = model_data["scaler"]
selector = model_data.get("selector")
feature_names = model_data["feature_names"]
label_encoders = model_data["label_encoders"]

if "sample_values" not in st.session_state:
    st.session_state.sample_values = {}


def get_sample_value(key, default):
    if key in {"incident_day", "incident_month"} and "incident_date" in st.session_state.sample_values:
        dt = pd.to_datetime(st.session_state.sample_values.get("incident_date"), errors="coerce")
        if not pd.isna(dt):
            if key == "incident_day":
                return int(dt.day)
            return int(dt.month)
    return st.session_state.sample_values.get(key, default)

c1, c2 = st.columns(2)
with c1:
    if st.button("Load Sample Case"):
        case = random.choice(SAMPLE_CASES)
        st.session_state.sample_values = case["data"]
        st.success(f"Loaded Sample: {case['name']}")
        st.rerun()
with c2:
    if st.button("Clear Form"):
        st.session_state.sample_values = {}
        st.rerun()

st.markdown("## Enter Claim Details")

with st.form("fraud_form"):

    d = {}

    st.subheader("Section 1 - Policy Details")
    a, b = st.columns(2)

    with a:
        csl_values = ["100/300", "250/500", "500/1000"]
        selected = get_sample_value("policy_csl", "250/500")
        d["policy_csl"] = st.selectbox("Policy CSL", csl_values, index=csl_values.index(selected) if selected in csl_values else 1)
    with b:
        umbrella_values = [0, 2000000, 5000000, 10000000]
        selected = int(get_sample_value("umbrella_limit", 0))
        d["umbrella_limit"] = st.selectbox("Umbrella Limit ($)", umbrella_values, index=umbrella_values.index(selected) if selected in umbrella_values else 0)

    st.subheader("Section 2 - Incident Details")
    a, b, c = st.columns(3)

    with a:
        severity_values = ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"]
        selected = get_sample_value("incident_severity", "Minor Damage")
        d["incident_severity"] = st.selectbox("Severity", severity_values, index=severity_values.index(selected) if selected in severity_values else 1)
    with b:
        incident_states = ["OH", "IN", "IL", "CA", "TX", "NY", "FL", "WA", "NC", "SC", "VA", "WV", "PA"]
        selected = get_sample_value("incident_state", "OH")
        d["incident_state"] = st.selectbox("Incident State", incident_states, index=incident_states.index(selected) if selected in incident_states else 0)
    with c:
        authority_values = ["Police", "Fire", "Ambulance", "None", "Other"]
        selected = get_sample_value("authorities_contacted", "Police")
        d["authorities_contacted"] = st.selectbox("Authorities Contacted", authority_values, index=authority_values.index(selected) if selected in authority_values else 0)

    a, b = st.columns(2)
    with a:
        d["incident_day"] = st.number_input("Incident Day", min_value=1, max_value=31, value=int(get_sample_value("incident_day", 15)))
    with b:
        d["incident_month"] = st.number_input("Incident Month", min_value=1, max_value=12, value=int(get_sample_value("incident_month", 6)))

    a, b = st.columns(2)
    with a:
        incident_types = ["Single Vehicle Collision", "Multi-vehicle Collision", "Vehicle Theft", "Parked Car"]
        selected = get_sample_value("incident_type", "Single Vehicle Collision")
        d["incident_type"] = st.selectbox("Incident Type", incident_types, index=incident_types.index(selected) if selected in incident_types else 0)
    with b:
        d["witnesses"] = st.number_input("Witnesses", 0, 10, int(get_sample_value("witnesses", 2)))

    st.subheader("Section 3 - Claim Details")
    a, b, c = st.columns(3)

    with a:
        d["total_claim_amount"] = st.number_input("Total Claim Amount", 0, 150000, int(get_sample_value("total_claim_amount", 10000)))
    with b:
        d["property_claim"] = st.number_input("Property Claim", 0, 100000, int(get_sample_value("property_claim", 2000)))
    with c:
        d["vehicle_claim"] = st.number_input("Vehicle Claim", 0, 150000, int(get_sample_value("vehicle_claim", 8000)))

    a, b, c = st.columns(3)
    with a:
        d["injury_claim"] = st.number_input("Injury Claim", 0, 100000, int(get_sample_value("injury_claim", 0)))
    with b:
        d["claim_ratio"] = st.number_input("Claim Ratio", min_value=0.0, max_value=500.0, value=float(get_sample_value("claim_ratio", 8.0)), step=0.1)
    with c:
        d["claim_per_vehicle"] = st.number_input("Claim Per Vehicle", min_value=0.0, max_value=150000.0, value=float(get_sample_value("claim_per_vehicle", 10000.0)), step=100.0)

    st.subheader("Section 4 - Investigation Info")
    a, b = st.columns(2)
    with a:
        pr_values = ["YES", "NO"]
        selected = str(get_sample_value("police_report_available", "YES")).upper()
        d["police_report_available"] = st.selectbox("Police Report Available", pr_values, index=pr_values.index(selected) if selected in pr_values else 0)
    with b:
        pd_values = ["YES", "NO"]
        selected = str(get_sample_value("property_damage", "NO")).upper()
        d["property_damage"] = st.selectbox("Property Damage", pd_values, index=pd_values.index(selected) if selected in pd_values else 1)

    st.subheader("Section 5 - Customer Info")
    a, b = st.columns(2)

    with a:
        hobby_values = [
            "chess", "reading", "board-games", "bungie-jumping", "base-jumping",
            "golf", "camping", "dancing", "skydiving", "movies", "hiking",
            "yachting", "paintball", "kayaking", "polo", "basketball",
            "video-games", "sleeping", "cross-fit", "exercise"
        ]
        selected = get_sample_value("insured_hobbies", "reading")
        d["insured_hobbies"] = st.selectbox("Insured Hobbies", hobby_values, index=hobby_values.index(selected) if selected in hobby_values else 1)
    with b:
        policy_states = ["OH", "IN", "IL", "CA", "TX", "NY", "FL", "WA"]
        selected = get_sample_value("policy_state", "OH")
        d["policy_state"] = st.selectbox("Policy State", policy_states, index=policy_states.index(selected) if selected in policy_states else 0)

    submit = st.form_submit_button("Analyze Claim")

if submit:

    try:

        input_data = st.session_state.sample_values.copy() if st.session_state.sample_values else {}
        input_data.update(d)

        auto = {
        "age": 35,
        "months_as_customer": 150,
        "policy_number": 500000,
        "policy_bind_date": "2020-01-01",
        "policy_deductable": 1000,
        "policy_annual_premium": 1200,
        "insured_sex": "MALE",
        "insured_education_level": "College",
        "insured_occupation": "prof-specialty",
        "insured_relationship": "husband",
        "capital-gains": 0,
        "capital-loss": 0,
        "collision_type": "Front Collision",
        "incident_hour_of_the_day": 14,
        "incident_city": "Columbus",
        "incident_location": "Street",
        "number_of_vehicles_involved": 1,
        "bodily_injuries": 0,
        "auto_make": "Toyota",
        "auto_model": "Camry",
        "auto_year": 2020,
        "injury_claim": 0,
        "vehicle_claim": max(input_data["total_claim_amount"] - input_data["property_claim"], 0)
        }

        for f in feature_names:
            input_data.setdefault(f, auto.get(f, 0))

        input_data.setdefault("claim_ratio", input_data["total_claim_amount"] / (input_data["policy_annual_premium"] + 1))
        input_data.setdefault("claim_per_vehicle", input_data["total_claim_amount"] / max(input_data["number_of_vehicles_involved"], 1))
        input_data["claim_per_injury"] = input_data["injury_claim"] / (input_data["bodily_injuries"] + 1)
        input_data["coverage_ratio"] = input_data["total_claim_amount"] / (input_data["umbrella_limit"] + 1)
        input_data["injury_severity_score"] = input_data["bodily_injuries"] * input_data["injury_claim"]
        input_data["night_incident"] = int(input_data["incident_hour_of_the_day"] < 6)

        input_data["incident_date"] = f"2025-{int(input_data['incident_month']):02d}-{int(input_data['incident_day']):02d}"

        incident_dt = pd.to_datetime(input_data.get("incident_date", "2025-01-01"), errors="coerce")
        if pd.isna(incident_dt):
            incident_dt = pd.to_datetime("2025-01-01")
        input_data["incident_month"] = int(incident_dt.month)
        input_data["incident_day"] = int(incident_dt.day)
        input_data["incident_weekday"] = int(incident_dt.weekday())

        df = pd.DataFrame([input_data])

        for col, le in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = le.transform(df[col].astype(str))
                except:
                    df[col] = 0

        df = df[feature_names]

        if selector is not None:
            selected = selector.transform(df)
            scaled = scaler.transform(selected)
        else:
            scaled = scaler.transform(df)

        prob = model.predict_proba(scaled)[0]
        fraud_prob = float(prob[1])
        legit_prob = float(prob[0])

        if fraud_prob > 0.6:
            result = "🚨 Fraud"
            risk_level = "High"
        elif fraud_prob > 0.3:
            result = "⚠ Suspicious"
            risk_level = "Medium"
        else:
            result = "✅ Legitimate"
            risk_level = "Low"

        st.markdown("---")
        st.subheader("Prediction Result")

        if risk_level == "High":
            st.markdown(f'<div class="fraud-alert">{result}</div>', unsafe_allow_html=True)
            st.error(f"Risk Level: {risk_level}")
        elif risk_level == "Medium":
            st.markdown(f'<div class="suspicious-alert">{result}</div>', unsafe_allow_html=True)
            st.warning(f"Risk Level: {risk_level}")
        else:
            st.markdown(f'<div class="legitimate-alert">{result}</div>', unsafe_allow_html=True)
            st.success(f"Risk Level: {risk_level}")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                f'''<div class="prob-card prob-fraud">
                <div class="prob-title">Fraud Probability</div>
                <div class="prob-value">{fraud_prob*100:.2f}%</div>
                </div>''',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'''<div class="prob-card prob-legit">
                <div class="prob-title">Risk Level</div>
                <div class="prob-value">{risk_level}</div>
                </div>''',
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(f"Prediction error: {e}")