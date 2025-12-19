import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')

st.set_page_config(page_title="Paris House Price Predictor", layout="wide")

st.title("🏠 Paris House Price Predictor")
st.write("Enter the details of the property below to estimate its market value.")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    squareMeters = st.number_input("Square Meters", min_value=0, value=75000)
    numberOfRooms = st.number_input("Number of Rooms", min_value=1, value=5)
    hasYard = st.selectbox("Has Yard", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    hasPool = st.selectbox("Has Pool", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    floors = st.number_input("Number of Floors", min_value=1, value=10)
    cityCode = st.number_input("City Code", min_value=0, value=50000)
    cityPartRange = st.number_input("City Part Range (1-10)", min_value=1, max_value=10, value=5)
    numPrevOwners = st.number_input("Number of Previous Owners", min_value=0, value=1)

with col2:
    made = st.number_input("Year Built", min_value=1800, max_value=2025, value=2010)
    isNewBuilt = st.selectbox("Is New Built", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    hasStormProtector = st.selectbox("Has Storm Protector", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    basement = st.number_input("Basement Square Meters", min_value=0, value=1000)
    attic = st.number_input("Attic Square Meters", min_value=0, value=1000)
    garage = st.number_input("Garage Square Meters", min_value=0, value=500)
    hasStorageRoom = st.selectbox("Has Storage Room", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    hasGuestRoom = st.number_input("Number of Guest Rooms", min_value=0, value=1)

# Prediction Logic
if st.button("Predict Price", use_container_width=True):
    # Prepare input data in the correct order
    features = np.array([[squareMeters, numberOfRooms, hasYard, hasPool, floors, cityCode, cityPartRange, 
                           numPrevOwners, made, isNewBuilt, hasStormProtector, basement, attic, 
                           garage, hasStorageRoom, hasGuestRoom]])
    
    # Scale the input features using the saved scaler
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    
    st.divider()
    st.subheader(f"Estimated House Price: ${prediction[0]:,.2f}")
