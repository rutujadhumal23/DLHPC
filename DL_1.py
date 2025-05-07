import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.layers import LeakyReLU

# Load data
data = pd.read_csv("Boston_Housing.csv")

# Drop unwanted column only if it exists
if 'Unnamed: 14' in data.columns:
    data = data.drop(columns=['Unnamed: 14'])

# Separate features and target
X = data.drop(columns=['MEDV'])  # Features
y = data['MEDV']  # Target (house price)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the model
model = Sequential([
    Dense(128, input_shape=(X_train_scaled.shape[1],)),
    LeakyReLU(alpha=0.1),
    Dropout(0.2),
    Dense(64),
    LeakyReLU(alpha=0.1),
    Dropout(0.2),
    Dense(32),
    LeakyReLU(alpha=0.1),
    Dropout(0.2),
    Dense(1)  # Output layer for regression
])

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, validation_split=0.2)

# Predict
predictions = model.predict(X_test_scaled)

# Print some predictions and actual values
print("Some Predictions and Actual Values:")
for i in range(10):
    print(f"Predicted Price: {predictions[i][0]:.2f}, Actual Price: {y_test.iloc[i]}")
