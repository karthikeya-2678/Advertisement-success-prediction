import streamlit as st
import pandas as pd
import pickle

# ---------------------------------
# Page Config
# ---------------------------------

st.set_page_config(page_title="Ad Success Predictor", layout="wide")

st.title("📈 Advertisement Success Prediction & AI Analysis")

# ---------------------------------
# Load Models
# ---------------------------------

@st.cache_resource
def load_models():
    with open("model/model.pkl", "rb") as f:
        models = pickle.load(f)

    return models["rating_model"], models["success_model"], models["money_model"]

rating_model, success_model, money_model = load_models()

# ---------------------------------
# Load Feature Metadata
# ---------------------------------

@st.cache_data
def load_metadata():
    df = pd.read_csv("data/train.csv")

    return df.drop(
        ["UserID", "netgain", "ratings", "money_back_guarantee"],
        axis=1,
        errors="ignore"
    )

df = load_metadata()

# ---------------------------------
# ML INPUT SECTION
# ---------------------------------

st.header("📊 Advertisement Parameters")

input_data = {}

for col in df.columns:

    if df[col].dtype == "object":

        options = df[col].dropna().unique().tolist()

        input_data[col] = st.selectbox(col, options)

    else:

        input_data[col] = st.number_input(
            col,
            value=float(df[col].mean())
        )

# ---------------------------------
# PREDICTION BUTTON
# ---------------------------------

if st.button("🚀 Analyze Advertisement", use_container_width=True):

    features = pd.DataFrame([input_data])

    st.subheader("🤖 ML Prediction")

    try:

        rating = rating_model.predict(features)[0]

        pred = success_model.predict(features)[0]

        prob = success_model.predict_proba(features)[0][1] * 100
        
        money_pred = money_model.predict(features)[0]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Predicted Rating", f"{rating:.3f}")

        c2.metric(
            "Prediction",
            "Successful 🟢" if pred == 1 else "May Fail 🔴"
        )

        c3.metric("Success Probability", f"{prob:.1f}%")
        
        c4.metric(
            "Money Back Guarantee",
            "Will Offer 💸" if money_pred == "Yes" else "No Guarantee 🚫" 
        )

        st.progress(prob / 100)

    except Exception as e:
        st.error(f"Prediction error: {e}")
        rating = 0
        prob = 0
        money_pred = "Unknown"