import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostClassifier

# --------------------------
# Load Model + Scaler
# --------------------------
model = joblib.load("catboost_model.pkl")
bmi_scaler = joblib.load("bmi_scaler.pkl")   # you saved this already

# --------------------------
# Age Encoding (1–11)
# --------------------------
def encode_age(age):
    if age < 25: return 1
    elif age < 30: return 2
    elif age < 35: return 3
    elif age < 40: return 4
    elif age < 45: return 5
    elif age < 50: return 6
    elif age < 55: return 7
    elif age < 60: return 8
    elif age < 65: return 9
    elif age < 70: return 10
    else: return 11

# --------------------------
# Streamlit UI
# --------------------------
st.title("Diabetes Risk Prediction")
st.write("Enter real health indicators. System converts them into model-ready values.")

def user_input_features():

    HighBP = st.selectbox("High Blood Pressure (0/1)", [0,1])
    HighChol = st.selectbox("High Cholesterol (0/1)", [0,1])

    # REAL BMI from user
    BMI_real = st.number_input("BMI (real value)", 10.0, 60.0, step=0.1)

    # Convert to scaled BMI
    BMI_scaled = float(bmi_scaler.transform([[BMI_real]])[0][0])

    Smoker = st.selectbox("Smoker (0/1)", [0,1])
    Stroke = st.selectbox("Stroke (0/1)", [0,1])
    HeartDiseaseorAttack = st.selectbox("Heart Disease or Attack (0/1)", [0,1])
    PhysActivity = st.selectbox("Physical Activity (0/1)", [0,1])
    Fruits = st.selectbox("Fruits Consumption (0/1)", [0,1])
    Veggies = st.selectbox("Vegetables Consumption (0/1)", [0,1])
    HvyAlcoholConsump = st.selectbox("Heavy Alcohol Consumption (0/1)", [0,1])
    AnyHealthcare = st.selectbox("Any Healthcare Access (0/1)", [0,1])
    GenHlth = st.slider("General Health (1–5)", 1, 5)
    MentHlth = st.slider("Mental Health days (0–30)", 0, 30)
    PhysHlth = st.slider("Physical Health days (0–30)", 0, 30)
    DiffWalk = st.selectbox("Difficulty Walking (0/1)", [0,1])
    Sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0,1])

    # REAL AGE from user
    Age_real = st.number_input("Age (years)", 1, 100, step=1)

    # Convert to encoded 1–11 class
    Age_encoded = encode_age(Age_real)

    CardioIssue = st.selectbox("Cardio Issue (0/1)", [0,1])
    HealthyLifestyleScore = st.slider("Healthy Lifestyle Score (0–10)", 0, 10)
    ComorbidityCount = st.slider("Comorbidity Count (0–10)", 0, 10)

    data = {
        "HighBP": HighBP,
        "HighChol": HighChol,
        "BMI": BMI_scaled,
        "Smoker": Smoker,
        "Stroke": Stroke,
        "HeartDiseaseorAttack": HeartDiseaseorAttack,
        "PhysActivity": PhysActivity,
        "Fruits": Fruits,
        "Veggies": Veggies,
        "HvyAlcoholConsump": HvyAlcoholConsump,
        "AnyHealthcare": AnyHealthcare,
        "GenHlth": GenHlth,
        "MentHlth": MentHlth,
        "PhysHlth": PhysHlth,
        "DiffWalk": DiffWalk,
        "Sex": Sex,
        "Age": Age_encoded,
        "CardioIssue": CardioIssue,
        "HealthyLifestyleScore": HealthyLifestyleScore,
        "ComorbidityCount": ComorbidityCount
    }

    return pd.DataFrame([data])


input_df = user_input_features()

# --------------------------
# Prediction
# --------------------------
if st.button("Predict"):
    pred = model.predict(input_df)
    pred_proba = model.predict_proba(input_df)

    st.subheader("Prediction Result")
    st.write(f"Predicted Diabetes Class: {int(pred[0])}")

    st.subheader("Prediction Probabilities")
    st.write(pd.DataFrame(pred_proba, columns=["Class 0", "Class 1", "Class 2"]))
