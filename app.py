import streamlit as st
import pandas as pd
import pickle
import os
import tempfile
import time
from google import genai

# ---------------------------------
# Page Config
# ---------------------------------

st.set_page_config(page_title="Ad Success Predictor", layout="wide")

st.title("📈 Advertisement Success Prediction & AI Analysis")

# ---------------------------------
# Sidebar
# ---------------------------------

st.sidebar.title("⚙️ AI Configuration")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.sidebar.error("⚠️ GEMINI_API_KEY environment variable is not set. Video analysis will not work.")
else:
    st.sidebar.success("✅ Connected to Gemini API")

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
# GEMINI VIDEO ANALYSIS
# ---------------------------------

    if uploaded_video:

        st.subheader("🧠 Gemini AI Video Analysis")

        if not api_key:
            st.warning("GEMINI_API_KEY was not found in environment variables.")
        else:

            with st.spinner("Analyzing advertisement..."):

                try:

                    # Initialize the new GenAI client
                    client = genai.Client(api_key=api_key)

                    # Save video temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:

                        tmp.write(uploaded_video.read())

                        video_path = tmp.name

                    # Upload using the new client API
                    video_file = client.files.upload(file=video_path)

                    while video_file.state.name == "PROCESSING":
                        time.sleep(5)
                        video_file = client.files.get(name=video_file.name)

                    prompt = f"""
You are an expert advertising strategist.

Analyze the advertisement video.

Context: {ad_context}

ML predicted rating: {rating}
ML success probability: {prob:.1f}%
ML predicted Money Back Guarantee: {money_pred}

Give short marketing insights:

• Product focus
• Strengths
• Weaknesses
• ROI expectation
• Key improvements
"""

                    # Generate content using the new client API
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[video_file, prompt]
                    )

                    st.markdown(response.text)

                    # Cleanup
                    client.files.delete(name=video_file.name)

                    os.remove(video_path)

                except Exception as e:
                    st.error(f"Gemini analysis failed: {e}")