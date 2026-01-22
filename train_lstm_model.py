import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pickle

print("=" * 60)
print("LSTM Energy Prediction Model Training")
print("=" * 60)

# Load the Bangladesh weather data
print("\n[1/6] Loading dataset...")
df = pd.read_csv('dataset/sorted_temp_and_rain_dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst few rows:")
print(df.head())

# Check for required columns
if 'YEAR' in df.columns:
    year_col = 'YEAR'
elif 'Year' in df.columns:
    year_col = 'Year'
else:
    print("Available columns:", df.columns.tolist())
    year_col = input("Enter the year column name: ")

# Find temperature and rainfall columns
temp_cols = [col for col in df.columns if 'temp' in col.lower() or 'tmp' in col.lower()]
rain_cols = [col for col in df.columns if 'rain' in col.lower() or 'prcp' in col.lower() or 'precip' in col.lower()]

print(f"\nFound temperature columns: {temp_cols}")
print(f"Found rainfall columns: {rain_cols}")

# Use the first temperature and rainfall column found
temp_col = temp_cols[0] if temp_cols else input("Enter temperature column name: ")
rain_col = rain_cols[0] if rain_cols else input("Enter rainfall column name: ")

print(f"\nUsing: {year_col}, {temp_col}, {rain_col}")

# Prepare data
print("\n[2/6] Preparing data...")

# Extract relevant columns
data = df[[year_col, temp_col, rain_col]].copy()

# Remove any rows with missing values
data = data.dropna()

# Convert to numpy array
weather_data = data[[temp_col, rain_col]].values

print(f"Weather data shape: {weather_data.shape}")
print(f"Date range: {data[year_col].min()} to {data[year_col].max()}")

# Simulate energy output based on weather
rain_norm = (weather_data[:, 1] - weather_data[:, 1].min()) / (weather_data[:, 1].max() - weather_data[:, 1].min())
temp_norm = (weather_data[:, 0] - weather_data[:, 0].min()) / (weather_data[:, 0].max() - weather_data[:, 0].min())

# Simulate energy output (MWh per month)
base_waterfall = 2500
base_geothermal = 1500

waterfall_energy = base_waterfall * (0.5 + rain_norm)
geothermal_energy = base_geothermal * (0.9 + 0.2 * temp_norm)
total_energy = waterfall_energy + geothermal_energy

# Add some realistic noise
np.random.seed(42)
noise = np.random.normal(0, 100, len(total_energy))
total_energy = total_energy + noise

# Create feature matrix: [temperature, rainfall]
X = weather_data
y = total_energy

print(f"\nEnergy range: {y.min():.0f} to {y.max():.0f} MWh/month")
print(f"Average energy: {y.mean():.0f} MWh/month")

# Normalize features
print("\n[3/6] Normalizing features...")
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Save scalers
with open('scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

print("Scalers saved!")

# Create sequences for LSTM
def create_sequences(X, y, time_steps=12):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

print("\n[4/6] Creating sequences...")
time_steps = 12
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

print(f"Sequence shape: {X_seq.shape}")
print(f"Target shape: {y_seq.shape}")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Build LSTM model
print("\n[5/6] Building LSTM model...")

model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(time_steps, 2)),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

# FIX: Use string names for loss and metrics instead of function references
model.compile(
    optimizer='adam',
    loss='mean_squared_error',  # Changed from 'mse'
    metrics=['mean_absolute_error']  # Changed from ['mae']
)

print("\nModel Summary:")
model.summary()

# Train the model
print("\n[6/6] Training model...")
print("This may take a few minutes...\n")

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Save the trained model with the new format
model.save('energy_predictor.h5')
print("\n✅ Model saved as 'energy_predictor.h5'")

# Evaluate the model
print("\n" + "=" * 60)
print("Model Evaluation")
print("=" * 60)

train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTraining Loss (MSE): {train_loss:.6f}")
print(f"Training MAE: {train_mae:.6f}")
print(f"Test Loss (MSE): {test_loss:.6f}")
print(f"Test MAE: {test_mae:.6f}")

# Make test predictions
y_pred_scaled = model.predict(X_test[:10])
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_actual = scaler_y.inverse_transform(y_test[:10])

print("\n" + "=" * 60)
print("Sample Predictions (First 10 Test Samples)")
print("=" * 60)
print(f"{'Actual (MWh)':<15} {'Predicted (MWh)':<15} {'Error (MWh)':<15}")
print("-" * 45)
for i in range(10):
    actual = y_actual[i][0]
    predicted = y_pred[i][0]
    error = abs(actual - predicted)
    print(f"{actual:<15.0f} {predicted:<15.0f} {error:<15.0f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['mean_absolute_error'], label='Training MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("\n📊 Training history plot saved as 'training_history.png'")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print("\nGenerated files:")
print("  1. energy_predictor.h5 - Trained LSTM model")
print("  2. scaler_X.pkl - Feature scaler")
print("  3. scaler_y.pkl - Target scaler")
print("  4. training_history.png - Training visualization")
print("\nYou can now use these files in your Streamlit app!")
print("=" * 60)