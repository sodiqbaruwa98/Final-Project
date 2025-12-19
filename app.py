import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(page_title="Paris Housing Predictor", layout="wide")

# 1. Load the model and scaler
@st.cache_resource
def load_assets():
    model = joblib.load('random_forest_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

try:
    rf_model, scaler = load_assets()
except:
    st.error("Model files not found! Make sure .joblib files are in the same folder.")
    st.stop()

st.title('🏡 Paris House Price Predictor')
st.info("Based on your Random Forest model from Colab.")

# 2. Setup Inputs (matching your training columns)
col1, col2, col3 = st.columns(3)

with col1:
    squareMeters = st.number_input("Square Meters", 10, 100000, 50000)
    numberOfRooms = st.number_input("Rooms", 1, 100, 50)
    floors = st.number_input("Floors", 1, 100, 50)
    made = st.number_input("Year Built", 1990, 2021, 2005)

with col2:
    basement = st.number_input("Basement Area", 0, 10000, 5000)
    attic = st.number_input("Attic Area", 0, 10000, 5000)
    garage = st.number_input("Garage Size", 100, 1000, 500)
    numPrevOwners = st.number_input("Prev Owners", 1, 10, 5)

with col3:
    cityCode = st.number_input("City Code", 0, 100000, 50000)
    cityPartRange = st.number_input("City Prestige (1-10)", 1, 10, 5)
    hasGuestRoom = st.number_input("Guest Rooms", 0, 10, 5)

# Checkboxes for your one-hot encoded features
st.subheader("Extra Amenities")
c1, c2, c3, c4, c5 = st.columns(5)
hasYard = c1.checkbox("Yard")
hasPool = c2.checkbox("Pool")
isNewBuilt = c3.checkbox("New Built")
hasStormProtector = c4.checkbox("Storm Protector")
hasStorageRoom = c5.checkbox("Storage Room")

# 3. Prediction
if st.button('Predict Price', type="primary"):
    # Create input array in exact training order
    # Note: Categorical columns have '_1' suffix in your trained X columns
    data = {
        'squareMeters': squareMeters, 'numberOfRooms': numberOfRooms, 'floors': floors, 
        'cityCode': cityCode, 'cityPartRange': cityPartRange, 'numPrevOwners': numPrevOwners, 
        'made': made, 'basement': basement, 'attic': attic, 'garage': garage, 
        'hasGuestRoom': hasGuestRoom, 'hasYard_1': int(hasYard), 'hasPool_1': int(hasPool), 
        'isNewBuilt_1': int(isNewBuilt), 'hasStormProtector_1': int(hasStormProtector), 
        'hasStorageRoom_1': int(hasStorageRoom)
    }
    
    input_df = pd.DataFrame([data])
    
    # Scale numerical parts (just like your training)
    num_cols = ['squareMeters', 'numberOfRooms', 'floors', 'cityCode', 'cityPartRange', 
                'numPrevOwners', 'made', 'basement', 'attic', 'garage', 'hasGuestRoom']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # Predict
    price = rf_model.predict(input_df)[0]
    st.success(f"### Estimated Price: €{price:,.2f}")