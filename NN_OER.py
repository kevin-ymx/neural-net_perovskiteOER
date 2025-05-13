import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Load data
data = pd.read_excel("data_gp.xlsx")

# Select input features and target
features = ['t', 'mu', 'RA', 'XA', 'XB', 'QA', 'Nd']
target = 'VRHE'
X = data[features].values
y = data[target].values

# Normalize input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
history = model.fit(X_scaled, y, epochs=100, verbose=1)

# Evaluate model
train_loss, train_mae = model.evaluate(X_scaled.T, y)
print(f"Train MAE from model.evaluate: {train_mae:.4f}")

y_pred = model.predict(X_scaled.T)
mae = mean_absolute_error(y_pred, y)
print(f"Train MAE from mean_absolute_error: {mae:.4f}")
