import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Ensure models directory exists
if not os.path.exists('models'): os.makedirs('models')

# 1. Load ALL relevant CSVs
results = pd.read_csv('data/results.csv')
races = pd.read_csv('data/races.csv')
drivers = pd.read_csv('data/drivers.csv')
constructors = pd.read_csv('data/constructors.csv')
status = pd.read_csv('data/status.csv')

# 2. Advanced Merging
# Link results to races, drivers, teams, and car status
df = results.merge(races[['raceId', 'year', 'circuitId']], on='raceId')
df = df.merge(drivers[['driverId', 'driverRef']], on='driverId')
df = df.merge(constructors[['constructorId', 'name']], on='constructorId', suffixes=('', '_team'))
df = df.merge(status[['statusId', 'status']], on='statusId')

# 3. Feature Engineering
# Target: Podium finish
df['is_podium'] = df['positionOrder'].apply(lambda x: 1 if x <= 3 else 0)

# We now include constructorId so the model learns which CARS are fastest
features = ['grid', 'year', 'circuitId', 'driverId', 'constructorId']
X = df[features]
y = df['is_podium']

# 4. Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, 'models/scaler.pkl')

# 5. Build a deeper Neural Network for more complex data
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(len(features),)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=15, batch_size=32)

model.save('models/f1_podium_model.h5')
print("✅ Advanced Model Trained with Team Data!")