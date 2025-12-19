import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the newly generated files
model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')

st.title("Paris House Price Predictor")

# Input collection (ensure all 16 are here)
col1, col2 = st.columns(2)
with col1:
    squareMeters = st.number_input("Square Meters", value=75000)
    numberOfRooms = st.number_input("Rooms", value=5)
    hasYard = st.selectbox("Yard", [0, 1])
    hasPool = st.selectbox("Pool", [0, 1])
    floors = st.number_input("Floors", value=2)
    cityCode = st.number_input("City Code", value=50000)
    cityPartRange = st.number_input("City Part Range", value=5)
    numPrevOwners = st.number_input("Prev Owners", value=1)

with col2:
    made = st.number_input("Year Built", value=2010)
    isNewBuilt = st.selectbox("New Built", [0, 1])
    hasStormProtector = st.selectbox("Storm Protector", [0, 1])
    basement = st.number_input("Basement Size", value=50)
    attic = st.number_input("Attic Size", value=50)
    garage = st.number_input("Garage Size", value=1)
    hasStorageRoom = st.selectbox("Storage Room", [0, 1])
    hasGuestRoom = st.number_input("Guest Rooms", value=1)

if st.button("Predict"):
    # Create the array with EXACTLY 16 items
    input_features = np.array([[
        squareMeters, numberOfRooms, hasYard, hasPool, floors, cityCode, 
        cityPartRange, numPrevOwners, made, isNewBuilt, hasStormProtector, 
        basement, attic, garage, hasStorageRoom, hasGuestRoom
    ]])
    
    # This line will NO LONGER throw an error if you ran the Step 1 script!
    input_scaled = scaler.transform(input_features)
    
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")
