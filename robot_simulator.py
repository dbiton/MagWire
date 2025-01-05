from typing import Tuple
import numpy as np
from keras import layers, models
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from robot_interface import RobotInterface
from robot_controller import BaseRobot

class RobotSimulator(RobotInterface):
    def __init__(self, save_model_path = "robot_simulator_model.h5", load_model_path = None):
        super().__init__()
        self.model = None
        # self.robot_integration = BaseRobot()
        self.save_model_path = save_model_path
        if load_model_path:
            print("loading model from", load_model_path)
            self.model = load_model(load_model_path)
        self.robot_config = np.zeros((6,))
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

    def train_model(self, data):
        X, y = self.process_data(data)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.model == None:
            input_shape = (X.shape[1], X.shape[2])
            output_length = y.shape[1]
            hidden_units = 64
            self.build_model(input_shape, output_length, hidden_units)

        self.model.fit(X_train, y_train, epochs=50, batch_size=128, validation_split=0.1)

        if self.save_model_path:
            self.model.save(self.save_model_path)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error on Test Data: {mse:.4f}")