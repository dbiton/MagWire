from typing import Tuple
import numpy as np
from keras import layers, models
from keras.models import load_model
from keras.losses import MeanSquaredError
import pandas as pd
from sklearn.metrics import mean_squared_error
from robot_interface import RobotInterface


def transform_to_euler_angles(matrix):
    assert matrix.shape == (4, 4), "Input must be a 4x4 matrix"
    R = matrix[:3,:3]
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arcsin(-R[2, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    return np.array([yaw, pitch, roll])


def transform_to_position(matrix):
    assert matrix.shape == (4, 4), "Input must be a 4x4 matrix"
    return matrix[:, -1][:3]


class RobotSimulator(RobotInterface):

    def __init__(self, save_model_path="robot_simulator_model.h5", load_model_path=None):
        super().__init__()
        self.model = None
        self.save_model_path = save_model_path
        if load_model_path:
            print("loading model from", load_model_path)
            self.model = load_model(load_model_path)
        self.magwire_pos = [0.5, 0.5]
        self.robot_config = [
            -1.6397998968707483,
            -1.5479772009751578,
            -1.6689372062683105,
            -1.50113898635421,
            1.5396380424499512,
            2.2953386306762695
        ]
        self.target = [0.5, 0.5] 
    
    def build_model(self, input_shape: Tuple[int, int], output_length: int, hidden_units: int) -> None:
        self.model = models.Sequential([
            layers.Input(input_shape),
            layers.LSTM(hidden_units, return_sequences=False),
            layers.Dense(output_length)
        ])
        self.model.compile(optimizer='adam', loss=MeanSquaredError())
        
    def get_config(self):
        return self.robot_config, self.magwire_pos
    
    def move_waypoint(self, waypoint):
        curr_robot_config = self.robot_config
        next_robot_transform = self.estimate_actuator_transform(waypoint[:6])
        next_robot_configs = self.estimate_robot_config(next_robot_transform)
        if next_robot_configs.shape[1] == 0:
            return self.magwire_pos
        next_robot_config = next_robot_configs[:,0]
        input_shape = self.model.input_shape
        data = {
            "time": [],
            "robot": [],
            "wire": []
        }
        seconds_past = 1
        seconds_future = 1
        for t_curr in np.arange(-seconds_past, 0, 1/100):
            data["time"].append(t_curr)
            data["robot"].append(curr_robot_config)
            data["wire"].append(self.magwire_pos)
        for t_curr in np.arange(0, seconds_future, 1/100):
            data["time"].append(t_curr)
            data["robot"].append(next_robot_config)
            data["wire"].append(None)
        data = pd.DataFrame(data)
        X_move = self.process_features(data)
        y_move = self.model.predict(X_move, verbose=0)
        self.magwire_pos = list(y_move[0][-2:])
        self.robot_config = next_robot_config.flatten().tolist()[0]
        self.last_waypoint = waypoint
        return self.magwire_pos
    
    def process_features(self, data:dict):
        data['actuator_transform'] = data['robot'].apply(self.estimate_actuator_transform)
        data['actuator_eular'] = data['actuator_transform'].apply(transform_to_euler_angles)
        data['actuator_pos'] = data['actuator_transform'].apply(transform_to_position)
        data['actuator_diff'] = data['actuator_pos'].apply(np.array).diff().bfill()
        data['actuator_delta'] = data['actuator_pos'].apply(np.array).diff().apply(lambda x: np.linalg.norm(x) if x is not None else 0).fillna(0)
        data['time_diff'] = data['time'].diff().bfill()
        data['actuator_velocity'] = (data['actuator_delta'] / data['time_diff']).fillna(0)
        data['actuator_velocity_diff'] = data['actuator_velocity'].diff().fillna(0)
        data['actuator_acceleration'] = (data['actuator_velocity_diff'] / data['time_diff']).fillna(0)
        seconds_past = 1
        seconds_future = 1
        mean_dt = np.mean(np.diff(data['time']))
        frames_count_past = int(seconds_past / mean_dt)
        frames_count_future = int(seconds_future / mean_dt)
        X = []
        for i_frame in range(frames_count_past, len(data) - frames_count_future + 1, frames_count_past):
            frames_past = data[i_frame - frames_count_past:i_frame]
            frames_future = data[i_frame:i_frame + frames_count_future].copy()
            frames_future['wire'] = [np.array([0, 0]) for _ in range(len(frames_future))]
            features = np.concatenate((frames_past.to_numpy(), frames_future.to_numpy()))
            features = [np.concatenate([np.array(v).flatten() for v in f]) for f in features]
            X.append(features)
        return np.array(X)
    
    def process_labels(self, data:dict):
        seconds_past = 1
        seconds_future = 1
        mean_dt = np.mean(np.diff(data['time']))
        frames_count_past = int(seconds_past / mean_dt)
        frames_count_future = int(seconds_future / mean_dt)
        y = []
        for i_frame in range(frames_count_past, len(data) - frames_count_future + 1, frames_count_past):
            frames_future = data[i_frame:i_frame + frames_count_future].copy()
            magwire_pos_future = np.stack(frames_future["wire"].to_numpy()).flatten()
            y.append(magwire_pos_future)
        return np.array(y)
    
    def process_data(self, data):
        return self.process_features(data), self.process_labels(data)

    def train_model(self, data):
        X, y = self.process_data(data)

        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        if self.model == None:
            input_shape = (X.shape[1], X.shape[2])
            output_length = y.shape[1]
            hidden_units = 64
            self.build_model(input_shape, output_length, hidden_units)

        self.model.fit(X_train, y_train, epochs=64, batch_size=64, validation_split=0.1)

        if self.save_model_path:
            self.model.save(self.save_model_path)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test, verbose=0)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error on Test Data: {mse:.4f}")
