import streamlit as st
import pickle
import numpy as np

# Load models
diabetes_model = pickle.load(open("diabetes_model.pkl", "rb"))
cancer_model = pickle.load(open("cancer_model.pkl", "rb"))

st.title("Disease Prediction System")

disease = st.sidebar.selectbox(
    "Select Disease",
    ("Diabetes", "Cancer")
)

# ---------------- DIABETES ----------------
if disease == "Diabetes":

    st.header("Diabetes Prediction")

    preg = st.number_input("Pregnancies")
    glucose = st.number_input("Glucose Level")
    bp = st.number_input("Blood Pressure")
    skin = st.number_input("Skin Thickness")
    insulin = st.number_input("Insulin Level")
    bmi = st.number_input("BMI")
    dpf = st.number_input("Diabetes Pedigree Function")
    age = st.number_input("Age")

    if st.button("Diabetes Result"):
        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        prediction = diabetes_model.predict(input_data)

        if prediction[0] == 1:
            st.error("Diabetic")
        else:
            st.success("Not Diabetic")

# ---------------- CANCER ----------------
if disease == "Cancer":

    st.header("Breast Cancer Prediction")

    features = []
    for i in range(30):
        val = st.number_input(f"Feature {i+1}")
        features.append(val)

    if st.button("Cancer Result"):
        input_data = np.array([features])
        prediction = cancer_model.predict(input_data)

        if prediction[0] == 0:
            st.error("Malignant Cancer Detected")
        else:
            st.success("Benign (No Cancer)")
