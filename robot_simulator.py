from typing import Tuple
import numpy as np
from keras import layers, models
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from robot_interface import RobotInterface

def transform_to_euler_angles(matrix):
    assert matrix.shape == (4, 4), "Input must be a 4x4 matrix"
    R = matrix[:3, :3]
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arcsin(-R[2, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    return np.array([yaw, pitch, roll])

def transform_to_position(matrix):
    assert matrix.shape == (4, 4), "Input must be a 4x4 matrix"
    return matrix[:, -1][:3]

class RobotSimulator(RobotInterface):
    def __init__(self, save_model_path = "robot_simulator_model.h5", load_model_path = None):
        super().__init__()
        self.model = None
        # self.robot_integration = BaseRobot()
        self.save_model_path = save_model_path
        if load_model_path:
            print("loading model from", load_model_path)
            self.model = load_model(load_model_path)
        self.magwire_pos = np.array([0.5, 0.5])
    
    def build_model(self, input_shape: Tuple[int, int], output_length: int, hidden_units: int) -> None:
        self.model = models.Sequential([
            layers.Input(input_shape),
            layers.LSTM(hidden_units, return_sequences=False),
            layers.Dense(output_length)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        
    def get_config(self):
        return self.magwire_pos, self.robot_config
    
    def move(self, robot_config):
        curr_robot_config = self.robot_config
        next_robot_config = robot_config
        input_shape = self.model.input_shape
        frames_count = input_shape[1]
        robot_pos = np.linspace(curr_robot_config, next_robot_config, frames_count)
        magwire_pos = np.concatenate((np.full((frames_count//2, 2), self.magwire_pos), np.zeros((frames_count//2, 2))), axis=0)
        X_move = np.concatenate((magwire_pos, robot_pos), axis=1)
        X_move = X_move.reshape(1, 578, 8)
        y_move = self.model.predict(X_move)
        self.magwire_pos = y_move[0][-2:]
        self.robot_config = next_robot_config
        return self.magwire_pos

    def process_data(self, data: dict):
        data['actuator_transform'] = data['robot'].apply(self.estimate_actuator_transform)
        data['actuator_eular'] = data['actuator_transform'].apply(transform_to_euler_angles)
        data['actuator_pos'] = data['actuator_transform'].apply(transform_to_position)
        # Calculate distance moved
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

        y = []
        X = []

        for i_frame in range(frames_count_past, len(data) - frames_count_future):
            frames = data[i_frame-frames_count_past:i_frame+frames_count_future]
            frames_past = data[i_frame-frames_count_past:i_frame]
            frames_future = data[i_frame:i_frame+frames_count_future].copy()
            magwire_pos_future = np.stack(frames_future["wire"].to_numpy()).flatten()
            y.append(magwire_pos_future)
            frames_future['wire'] = [np.array([0, 0]) for _ in range(len(frames_future))]
            features = np.concatenate((frames_past.to_numpy(), frames_future.to_numpy()))
            features = [np.concatenate([np.array(v).flatten() for v in f]) for f in features]
            X.append(features)        
        return np.array(X), np.array(y)

    def train_model(self, data):
        X, y = self.process_data(data)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.model == None:
            input_shape = (X.shape[1], X.shape[2])
            output_length = y.shape[1]
            hidden_units = 64
            self.build_model(input_shape, output_length, hidden_units)

        self.model.fit(X_train, y_train, epochs=32, batch_size=32, validation_split=0.1)

        if self.save_model_path:
            self.model.save(self.save_model_path)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error on Test Data: {mse:.4f}")