import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(page_title="Medical Insurance Prediction App", layout="centered")

st.title("Medical Insurance Prediction App")
st.write("Enter the input values to make a prediction.")

# Load trained model
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    return model

model = load_model()

st.subheader("Input Features")

# Example input fields (modify based on your dataset)
age = st.number_input("Age", min_value=0, step=1, format="%d")

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

bmi = st.number_input("BMI", min_value=0.0, format="%.2f")

children = st.number_input(
    "Number of Children",
    min_value=0,
    step=1
)

smoker = st.selectbox(
    "Smoker",
    ["Yes", "No"]
)

region = st.selectbox(
    "Region",
    ["Southwest","Southeast","Northwest","Northeast"]
)

sex = 1 if sex == "Male" else 0
smoker = 1 if smoker == "Yes" else 0

region_map = {
    "Southwest":0,
    "Southeast":1,
    "Northwest":2,
    "Northeast":3
}

region = region_map[region]

# Prediction button
if st.button("Predict"):

    input_data = [[age, sex, bmi, children, smoker, region]]

    prediction = model.predict(input_data)

    st.success(f"Predicted Insurance Cost: {prediction[0]}")


