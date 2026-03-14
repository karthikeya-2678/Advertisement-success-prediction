import pickle
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
import tempfile
import os
from video_analyzer import VideoAnalyzer

app = FastAPI(
    title="Advertisement Success Prediction API",
    description="API for predicting ad success and generating AI marketing insights.",
    version="1.0.0"
)

# ---------------------------------
# Load Models on Startup
# ---------------------------------
def load_models():
    with open("model/model.pkl", "rb") as f:
        models = pickle.load(f)
    return models["rating_model"], models["success_model"], models["money_model"]

rating_model, success_model, money_model = load_models()

# ---------------------------------
# Defining Data Schemas
# ---------------------------------
class AdFeatures(BaseModel):
    realtionship_status: str
    industry: str
    genre: str
    targeted_sex: str
    average_runtime_minutes_per_week: int = Field(alias="average_runtime(minutes_per_week)")
    airtime: str
    airlocation: str
    expensive: str

class MLPredictionResponse(BaseModel):
    predicted_rating: float
    success_probability_percentage: float
    will_succeed: bool
    will_offer_money_back_guarantee: bool

# ---------------------------------
# Endpoint: ML Predictions
# ---------------------------------
@app.post("/predict/metrics", response_model=MLPredictionResponse)
def predict_metrics(features: AdFeatures):
    try:
        # Convert incoming JSON into a pandas DataFrame using exact feature names
        input_data = features.dict(by_alias=True)
        df = pd.DataFrame([input_data])
        
        # Predictions
        rating = rating_model.predict(df)[0]
        pred = success_model.predict(df)[0]
        prob = success_model.predict_proba(df)[0][1] * 100
        money_pred = money_model.predict(df)[0]

        return MLPredictionResponse(
            predicted_rating=float(rating),
            success_probability_percentage=float(prob),
            will_succeed=bool(pred == 1),
            will_offer_money_back_guarantee=bool(money_pred == "Yes")
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# ---------------------------------
# Endpoint: CV Video Analysis
# ---------------------------------
@app.post("/analyze/video")
async def analyze_video(
    video: UploadFile = File(...),
    rating: float = Form(0.0),
    success_prob: float = Form(0.0),
    money_pred: str = Form("Unknown")
):
    if not video.filename.endswith((".mp4", ".mov", ".avi")):
        raise HTTPException(status_code=400, detail="Invalid video format. Use .mp4, .mov, or .avi")
        
    try:
        # Save uploaded file temporarily for OpenCV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await video.read()
            tmp.write(content)
            video_path = tmp.name
            
        analyzer = VideoAnalyzer()
        report = analyzer.analyze_ad_video(
            video_path=video_path,
            ml_rating=rating,
            ml_success_prob=success_prob,
            ml_money_pred=money_pred
        )
        
        # Cleanup
        os.remove(video_path)
        
        return {"analysis": report}

    except Exception as e:
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)
        raise HTTPException(status_code=500, detail=f"CV analysis failed: {str(e)}")

# To run the server:
# uvicorn api:app --reload
