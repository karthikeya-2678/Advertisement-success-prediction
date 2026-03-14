import streamlit as st
import pandas as pd
import pickle
import tempfile
import os
from video_analyzer import VideoAnalyzer

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
# Layout
# ---------------------------------

col1, col2 = st.columns(2)

# ---------------------------------
# ML INPUT SECTION
# ---------------------------------

with col1:
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
# VIDEO SECTION
# ---------------------------------

with col2:
    st.header("🎬 Upload Advertisement Video")

    uploaded_video = st.file_uploader(
        "Upload video for CV Analysis",
        type=["mp4", "mov", "avi"]
    )

    if uploaded_video:
        st.video(uploaded_video)

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

# ---------------------------------
# COMPUTER VISION VIDEO ANALYSIS
# ---------------------------------

    if uploaded_video:

        st.subheader("👁️ Local Computer Vision Analysis")

        with st.spinner("Analyzing video frames safely and locally..."):
            try:
                # Save video temporarily for OpenCV to read
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uploaded_video.read())
                    video_path = tmp.name

                # Run CV Analyzer
                analyzer = VideoAnalyzer()
                report = analyzer.analyze_ad_video(
                    video_path=video_path,
                    ml_rating=rating,
                    ml_success_prob=prob,
                    ml_money_pred=money_pred
                )
                
                st.markdown(report)

                # Cleanup
                os.remove(video_path)

            except Exception as e:
                st.error(f"Computer Vision analysis failed: {e}")