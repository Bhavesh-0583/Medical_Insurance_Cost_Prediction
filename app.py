import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(page_title="ML Prediction App", layout="centered")

st.title("Machine Learning Prediction App")
st.write("Enter the input values to make a prediction.")

# Load trained model
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    return model

model = load_model()

st.subheader("Input Features")

# Example input fields (modify based on your dataset)
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)
feature4 = st.number_input("Feature 4", value=0.0)
feature5 = st.number_input("Feature 5", value=0.0)
feature6 = st.number_input("Feature 6", value=0.0)

# Prediction button
if st.button("Predict"):

    input_data = np.array([[feature1, feature2, feature3, feature4]])

    try:
        prediction = model.predict(input_data)
        st.success(f"Prediction: {prediction[0]}")
    except Exception as e:

        st.error(f"Error during prediction: {e}")
