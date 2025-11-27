import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostClassifier

# --------------------------
# Load the trained model
# --------------------------
model_path = "catboost_model.pkl"
loaded_model = joblib.load(model_path)

# --------------------------
# Streamlit App
# --------------------------
st.title("Diabetes Risk Prediction")
st.write("Enter the health indicators below to predict diabetes risk class (0, 1, 2)")

# Input fields
def user_input_features():
    HighBP = st.selectbox("HighBP", [0,1])
    HighChol = st.selectbox("HighChol", [0,1])
    BMI = st.number_input("BMI", -2.47, 10.534, step=0.1)
    Smoker = st.selectbox("Smoker", [0,1])
    Stroke = st.selectbox("Stroke", [0,1])
    HeartDiseaseorAttack = st.selectbox("HeartDiseaseorAttack", [0,1])
    PhysActivity = st.selectbox("PhysActivity", [0,1])
    Fruits = st.selectbox("Fruits", [0,1])
    Veggies = st.selectbox("Veggies", [0,1])
    HvyAlcoholConsump = st.selectbox("HvyAlcoholConsump", [0,1])
    AnyHealthcare = st.selectbox("AnyHealthcare", [0,1])
    GenHlth = st.slider("GenHlth", 1, 5)
    MentHlth = st.slider("MentHlth", 0, 30)
    PhysHlth = st.slider("PhysHlth", 0, 30)
    DiffWalk = st.selectbox("DiffWalk", [0,1])
    Sex = st.selectbox("Sex", [0,1])
    Age = st.number_input("Age", 1, 11, step=1)
    CardioIssue = st.selectbox("CardioIssue", [0,1])
    HealthyLifestyleScore = st.slider("HealthyLifestyleScore", 0, 10)
    ComorbidityCount = st.slider("ComorbidityCount", 0, 10)

    data = {
        'HighBP': HighBP,
        'HighChol': HighChol,
        'BMI': BMI,
        'Smoker': Smoker,
        'Stroke': Stroke,
        'HeartDiseaseorAttack': HeartDiseaseorAttack,
        'PhysActivity': PhysActivity,
        'Fruits': Fruits,
        'Veggies': Veggies,
        'HvyAlcoholConsump': HvyAlcoholConsump,
        'AnyHealthcare': AnyHealthcare,
        'GenHlth': GenHlth,
        'MentHlth': MentHlth,
        'PhysHlth': PhysHlth,
        'DiffWalk': DiffWalk,
        'Sex': Sex,
        'Age': Age,
        'CardioIssue': CardioIssue,
        'HealthyLifestyleScore': HealthyLifestyleScore,
        'ComorbidityCount': ComorbidityCount
    }
    features = pd.DataFrame([data])
    return features

input_df = user_input_features()

# Prediction
if st.button("Predict"):
    prediction = loaded_model.predict(input_df)
    prediction_proba = loaded_model.predict_proba(input_df)
    st.write(f"Predicted Diabetes Class: {int(prediction[0])}")
    st.write("Prediction Probabilities:")
    st.write(pd.DataFrame(prediction_proba, columns=["Class 0", "Class 1", "Class 2"]))
