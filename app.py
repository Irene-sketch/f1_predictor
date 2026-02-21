import streamlit as st
import requests
#used to send http request,sends data from frontend to backedn(fast api)


st.set_page_config(page_title="F1 Predictor", page_icon="🏎️")
st.title("🏎️ F1 Podium Predictor")

st.markdown("Enter race details to see the probability of a podium finish.")

# Inputs
grid = st.slider("Starting Grid Position", 1, 20, 1)
driver_id = st.number_input("Driver ID", value=1)
constructor_id = st.number_input("Constructor ID", value=1) # Added this to match the new model
circuit_id = st.number_input("Circuit ID", value=1)
year = st.number_input("Year", value=2024)

if st.button("Calculate Probability"):
    payload = {
        "grid": grid,
        "year": year,
        "circuitId": circuit_id,
        "driverId": driver_id,
        "constructorId": constructor_id # Included this to utilize the full dataset
    } #this dictonary matches exactly with the fast api schema
    
    # Call FastAPI
    # the flow:Streamlit → FastAPI → TensorFlow → Prediction → FastAPI → Streamlit
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        prob = response.json()['probability']
        #extracting prediction
        st.metric(label="Podium Chance", value=f"{prob*100:.2f}%")
        
        if prob > 0.5:
            st.success("High chance of a podium!")
        else:
            st.warning("Low chance of a podium.")
    except:
        st.error("Make sure your Backend (FastAPI) is running!")