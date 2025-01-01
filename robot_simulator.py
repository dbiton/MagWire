import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Model Definition
def build_model(input_shape, output_length, hidden_units):
    model = models.Sequential([
        layers.LSTM(hidden_units, input_shape=input_shape, return_sequences=False),
        layers.Dense(output_length)
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

def process_data(data):
    X = []
    y = []
    seconds_past = 3
    seconds_future = 3
    mean_dt = np.mean(np.diff(data['time']))
    frames_count_past = int(seconds_past / mean_dt)
    frames_count_future = int(seconds_future / mean_dt)
    for i_frame in range(frames_count_past, len(data) - frames_count_future):
        frames = data[i_frame-frames_count_past:i_frame+frames_count_future]
        frames_past = data[i_frame-frames_count_past:i_frame]
        frames_future = data[i_frame:i_frame+frames_count_future]
        magwire_pos_future = np.stack(frames_future["wire"].to_numpy()).flatten()
        y.append(magwire_pos_future)
        robot_pos = np.stack(frames["robot"].to_numpy())
        magwire_pos_past = np.stack(frames_past["wire"].to_numpy())
        magwire_pos_past_padding = np.zeros((len(frames)-len(magwire_pos_past), 2))
        magwire_pos_past = np.concatenate((magwire_pos_past, magwire_pos_past_padding), axis=0)
        features = np.concatenate((magwire_pos_past,robot_pos), axis=1)
        X.append(features)        
    return np.array(X), np.array(y)

def create_model(data):
    X, y = process_data(data)
    
    input_shape = (X.shape[1], X.shape[2])  # (sequence_length, features)
    output_length = y.shape[1]
    hidden_units = 64

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and compile the model
    model = build_model(input_shape, output_length, hidden_units)
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=32, batch_size=32, validation_split=0.1)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on Test Data: {mse:.4f}")

    return model