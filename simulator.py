import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Model Definition
def build_model(input_shape, hidden_units):
    model = models.Sequential([
        layers.LSTM(hidden_units, input_shape=input_shape, return_sequences=False),
        layers.Dense(2)  # Output layer for B_final_x and B_final_y
    ])
    return model

def create_sequences(X, y, seq_length):
    """
    Create sequences of length `seq_length` from the time-ordered data.
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])  # Take `seq_length` consecutive steps
        y_seq.append(y[i+seq_length])   # Corresponding target is after the sequence
    return np.array(X_seq), np.array(y_seq)

def create_model(X, y):
    # Hyperparameters
    seq_length = 10  # Number of previous steps to include
    input_shape = (seq_length, X.shape[1])  # (sequence_length, features)
    hidden_units = 64

    # Prepare the data
    X_seq, y_seq = create_sequences(X, y, seq_length)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # Build and compile the model
    model = build_model(input_shape, hidden_units)
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on Test Data: {mse:.4f}")

    return model, mse