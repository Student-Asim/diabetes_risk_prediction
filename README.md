Diabetes Health Indicator – ML Classification Project

End-to-end machine-learning system predicting diabetes categories (0, 1, 2) using structured health-indicator features.

Project Summary

Built a multi-class classifier using CatBoost on the Diabetes 012 Health Indicators Dataset. Implemented preprocessing, feature engineering, model training, hyperparameter tuning, cross-validation, model persistence, Streamlit UI, FastAPI backend, and Docker containerization. Repository contains complete production-ready workflow.

Features

Data cleaning, scaling, and feature derivation

Class balancing (upsampling)

CatBoost model optimized with RandomizedSearchCV

Cross-validation using StratifiedKFold

Model export using joblib

REST prediction endpoint (FastAPI)

Interactive local UI using Streamlit

Dockerized deployment

Git-based project versioning and repo structure

Tech Stack

Python

Pandas, NumPy

Scikit-Learn

CatBoost

FastAPI

Uvicorn

Streamlit

Docker

Git/GitHub

How to Run Locally
1. Create environment and install dependencies
pip install -r requirements.txt

2. Train model
python train_model.py

3. Run Streamlit UI
streamlit run app.py

4. Run FastAPI backend
uvicorn api:app --reload

5. Docker build and run
docker build -t diabetes-app .
docker run -p 8000:8000 diabetes-app

Repository Structure
.
├── data/                        # Dataset source
├── models/                      # Saved model .pkl
├── train_model.py               # Training pipeline
├── api.py                       # FastAPI backend
├── app.py                       # Streamlit UI
├── Dockerfile                   # Container build file
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation

Prediction Workflow

User inputs health-indicator parameters

Input validated and sent to FastAPI endpoint

Model returns class prediction and class probabilities

Streamlit displays prediction output

Deployment Targets

Local testing via Streamlit

FastAPI microservice behind Docker

Optional cloud deployment (any server running Docker)

Purpose

Provides a complete template for ML deployment pipelines: data → model → API → UI → containerization → version control. Suitable for production-style demonstration projects and portfolio work.

ChatGPT can ma