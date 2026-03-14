import streamlit as st
import pandas as pd
import pickle
import os
import tempfile
import time
import google.generativeai as genai

# ---------------------------------
# Page Config
# ---------------------------------

st.set_page_config(page_title="Ad Success Predictor", layout="wide")

st.title("📈 Advertisement Success Prediction & AI Analysis")

# ---------------------------------
# Sidebar
# ---------------------------------

st.sidebar.title("⚙️ AI Configuration")

api_key = st.sidebar.text_input(
    "Enter Google Gemini API Key",
    type="password"
)

st.sidebar.info(
    "Gemini AI analyzes your uploaded advertisement video "
    "and gives marketing insights."
)

# ---------------------------------
# Load Models
# ---------------------------------

@st.cache_resource
def load_models():
    with open("model/model.pkl", "rb") as f:
        models = pickle.load(f)

    return models["rating_model"], models["success_model"]

rating_model, success_model = load_models()

# ---------------------------------
# Load Feature Metadata
# ---------------------------------

@st.cache_data
def load_metadata():
    df = pd.read_csv("data/train.csv")

    return df.drop(
        ["UserID", "netgain", "ratings"],
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
        "Upload video",
        type=["mp4", "mov", "avi"]
    )

    ad_context = st.text_area(
        "Optional: Describe the ad or campaign"
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

        c1, c2, c3 = st.columns(3)

        c1.metric("Predicted Rating", f"{rating:.3f}")

        c2.metric(
            "Prediction",
            "Successful 🟢" if pred == 1 else "May Fail 🔴"
        )

        c3.metric("Success Probability", f"{prob:.1f}%")

        st.progress(prob / 100)

    except Exception as e:
        st.error(f"Prediction error: {e}")
        rating = 0
        prob = 0

# ---------------------------------
# GEMINI VIDEO ANALYSIS
# ---------------------------------

    if uploaded_video:

        st.subheader("🧠 Gemini AI Video Analysis")

        if not api_key:
            st.warning("Enter Gemini API key in sidebar.")
        else:

            with st.spinner("Analyzing advertisement..."):

                try:

                    genai.configure(api_key=api_key)

                    # Save video temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:

                        tmp.write(uploaded_video.read())

                        video_path = tmp.name

                    video_file = genai.upload_file(path=video_path)

                    while video_file.state.name == "PROCESSING":
                        time.sleep(5)
                        video_file = genai.get_file(video_file.name)

                    model = genai.GenerativeModel("gemini-2.5-flash")

                    prompt = f"""
You are an expert advertising strategist.

Analyze the advertisement video.

Context: {ad_context}

ML predicted rating: {rating}
ML success probability: {prob:.1f}%

Give short marketing insights:

• Product focus
• Strengths
• Weaknesses
• ROI expectation
• Key improvements
"""

                    response = model.generate_content([video_file, prompt])

                    st.markdown(response.text)

                    genai.delete_file(video_file.name)

                    os.remove(video_path)

                except Exception as e:
                    st.error(f"Gemini analysis failed: {e}")