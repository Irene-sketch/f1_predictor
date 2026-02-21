from fastapi import FastAPI
from pydantic import BaseModel
#used for data validation ,ensures ip data matches the format(int,float etc)
import tensorflow as tf
#load your trained nn model and make predicytions
import joblib
#load saved pyhton objects
import numpy as np
#create arrays,nn expects num arrayas as ip

app = FastAPI()
#create backend server

# Load the saved model and scaler
model = tf.keras.models.load_model('models/f1_podium_model.h5')
scaler = joblib.load('models/scaler.pkl')

class PredictionInput(BaseModel):
    grid: int
    year: int
    circuitId: int
    driverId: int
    constructorId: int # Added to utilize the constructor (team) data
#This defines what the API expects in request body

@app.get("/")
def home():
    return {"message": "F1 Prediction API is running"}

@app.post("/predict")
def predict(data: PredictionInput):
    # Prepare input - Updated to include data.constructorId
    features = np.array([[data.grid, data.year, data.circuitId, data.driverId, data.constructorId]])
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0][0]
    return {"probability": float(prediction)}