# fastapi_app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# -----------------------
# Load model
# -----------------------
model_path = "catboost_model.pkl"
model = joblib.load(model_path)

# -----------------------
# Define input schema
# -----------------------
class DiabetesInput(BaseModel):
    HighBP: int
    HighChol: int
    BMI: float
    Smoker: int
    Stroke: int
    HeartDiseaseorAttack: int
    PhysActivity: int
    Fruits: int
    Veggies: int
    HvyAlcoholConsump: int
    AnyHealthcare: int
    GenHlth: int
    MentHlth: int
    PhysHlth: int
    DiffWalk: int
    Sex: int
    Age: int
    CardioIssue: int
    HealthyLifestyleScore: int
    ComorbidityCount: int

# -----------------------
# Initialize app
# -----------------------
app = FastAPI(title="Diabetes Prediction API")

# -----------------------
# Create endpoint
# -----------------------
@app.post("/predict")
def predict_diabetes(input_data: DiabetesInput):
    # Convert input to DataFrame
    df = pd.DataFrame([input_data.dict()])
    
    # Make prediction
    pred = model.predict(df)[0]
    pred_proba = model.predict_proba(df)[0]
    
    return {
        "Predicted_Class": int(pred),
        "Prob_Class_0": float(pred_proba[0]),
        "Prob_Class_1": float(pred_proba[1]),
        "Prob_Class_2": float(pred_proba[2])
    }
