# New RPPO with two LSTM layers 


import gymnasium
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
import numpy as np
from typing import Optional
import json
import argparse

from data_processing import prepare_tensor
from utils.midi_utils import write_midi_from_timings
from utils.files import save_model
from utils.lstm_augmented import create_augmented_sequence_with_flags
from rl.custom_network import CustomRPPO



class MusicAccompanistEnv(gymnasium.Env):
    """
    Observations: A sliding window of historical data plus a future reference timing.
    """
    def __init__(self, data, windows, config_file, option):
        super(MusicAccompanistEnv, self).__init__()
        self.data = data 
        self.n_notes = self.data.shape[1]
        self.option = option

        with open(config_file, 'r') as f:
            self.config = json.load(f)

        self.window_size = self.config['window_size']
        self.windows = windows
        self.current_index = self.window_size
        
        # Create a Dict observation space with separate components for historical data and future reference
        # This better aligns with how TempoPredictor processes the data
        if option in ["difference", "2row_with_ratio", "ratio"]:
            # For options that use a 2D array for historical data plus a future reference
            # The historical window size is window_size-1 since we use the last position for future ref
            hist_shape = (2, self.window_size - 1)
            # The observation space is flattened when passed to the model
            flat_dim = 2 * (self.window_size - 1) + 1  # +1 for future ref
            self.observation_space = spaces.Box(low=0, high=10.0, shape=(flat_dim,), dtype=np.float32)
        elif option == "raw":
            flat_dim = 2 * self.window_size + 1
            self.observation_space = spaces.Box(low=0, high=30.0, shape=(flat_dim,), dtype=np.float32)
        else:
            # Default for other options
            hist_shape = (2, self.window_size)
            flat_dim = 2 * self.window_size 
            self.observation_space = spaces.Box(low=0, high=10.0, shape=(flat_dim,), dtype=np.float32)
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.forecast_window = 3
    
    def obs_prep(self, reset):
        if reset:
            self.current_index = self.window_size
        
        match self.option:
            case "raw":
                next_window = self.data[:, self.current_index - self.window_size:self.current_index].astype(np.float32)
                next_window = next_window - next_window[:, 0:1]

                future_ref = np.array([self.data[1, self.current_index]], dtype=np.float32)
                next_window = np.concatenate([next_window.flatten(), future_ref], axis=0)
                # Flatten for model compatibility
                return next_window
                

            case "difference":
                # Historical data: differences between consecutive time steps
                first_note = self.data[:, self.current_index - self.window_size:self.current_index - 1].astype(np.float32)
                second_note = self.data[:, self.current_index - self.window_size + 1:self.current_index].astype(np.float32)
                historical_data = second_note - first_note
                
                # Future reference: next reference timing difference
                future_ref = np.array([self.data[1, self.current_index] - self.data[1, self.current_index - 1]], dtype=np.float32)
                
                # Flatten historical data and append future ref
                return np.concatenate([historical_data.flatten(), future_ref])

            case "ratio":
                first_note = self.data[:, self.current_index - self.window_size:self.current_index - 1].astype(np.float32)
                second_note = self.data[:, self.current_index - self.window_size + 1:self.current_index].astype(np.float32)
                historical_data = second_note - first_note
                historical_data[0] = historical_data[0] / (historical_data[1] + 1e-8)

                future_ref = np.array([self.data[1, self.current_index] - self.data[1, self.current_index - 1]], dtype=np.float32)
                return np.concatenate([historical_data.flatten(), future_ref])
                
            case "2row_with_ratio":
                # Historical data: differences between consecutive time steps
                first_note = self.data[:, self.current_index - self.window_size:self.current_index - 1].astype(np.float32)
                second_note = self.data[:, self.current_index - self.window_size + 1:self.current_index].astype(np.float32)
                historical_data = second_note - first_note
                # Calculate ratios for row 0
                historical_data[0] = historical_data[0] / (historical_data[1] + 1e-8)
                
                # Future reference: next reference timing difference
                future_ref = np.array([self.data[1, self.current_index] - self.data[1, self.current_index - 1]], dtype=np.float32)
                
                # Flatten historical data and append future ref
                return np.concatenate([historical_data.flatten(), future_ref])
                
            case "1row+next":
                next_window = self.data[:, self.current_index - self.window_size:self.current_index].astype(np.float32)
                next_window = next_window - next_window[:, 0:1] 
                # Next prediction note is the future reference
                future_ref = np.array([self.data[1, self.current_index]], dtype=np.float32)
                return np.concatenate([next_window.flatten(), future_ref])
                
            case "forecast":
                first_note_real = self.data[0:1, self.current_index - self.window_size:self.current_index - 1].astype(np.float32)
                second_note_real = self.data[:, self.current_index - self.window_size + 1:self.current_index].astype(np.float32)
                next_window_real = second_note_real - first_note_real

                first_note_ref = self.data[1:2, self.current_index - self.window_size:self.current_index - 1 + self.forecast_window].astype(np.float32)
                second_note_ref = self.data[1:2, self.current_index - self.window_size + 1:self.current_index + self.forecast_window].astype(np.float32)
                next_window_ref = second_note_ref - first_note_ref

                obs = create_augmented_sequence_with_flags(next_window_real[0], next_window_ref[0], self.forecast_window).flatten()
                return obs
    
    def reset(self, seed=None, options=None):
        # Update to match current gymnasium API
        if self.windows == 'all':
            obs = self.obs_prep(True)
        else:
            self.current_index = self.windows[0]
            obs = self.obs_prep(False)
        return obs, {}  # Return empty info dict for gymnasium compatibility

    def step(self, action):
        """
        Apply the speed adjustment factor to the reference timing,
        then compute the reward based on how close the predicted timing (ref_timing * action)
        is to the soloist's actual timing.
        """
        # Extract the current reference and soloist timing
        ref_timing = self.data[1, self.current_index] - self.data[1, self.current_index - 1]
        solo_timing = self.data[0, self.current_index] - self.data[0, self.current_index - 1]
        predicted_log_speed = action[0]
        speed_factor = np.exp(predicted_log_speed)


        predicted_timing = ref_timing * speed_factor


        # reward = self.reward_function(predicted_timing, solo_timing, action[0])
        reward = self.new_reward_function(solo_timing, ref_timing, action[0])
        
        self.current_index += 1
        if self.windows == 'all':
            done = (self.current_index >= self.n_notes - self.forecast_window)
        else:
            done = (self.current_index > self.windows[1] - self.forecast_window)
        
        if not done:
            obs = self.obs_prep(False)
            if self.current_index % 200 == 0:
                print(f"Current index: {self.current_index}, Reward: {reward}, Action: {action}")
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            
        info = {"predicted_timing": predicted_timing}
        # Update to match current gymnasium API
        return obs, reward, done, False, info

    def new_reward_function(self, solo_timing, ref_timing, action):
        epsilon = 1e-8
        ideal_log_action = np.log((solo_timing + epsilon) / (ref_timing + epsilon))
        
        scale = 1.0  # Tune this parameter as needed

        error = scale * np.abs(action - ideal_log_action)
        reward = -error
        return reward


class RecurrentPPOAgent:
    def __init__(self, env, file_path: Optional[str] = None):
        self.file_path = file_path
        self.lstm_states = None
        # episode_starts must be an array-like value; we use shape (1,) here.
        self.episode_starts = np.ones((1,), dtype=bool)
        self.env = env
        self.model = None
        self._initialize()
        self.model.set_env(env)


    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = CustomRPPO("Custom", env, verbose=1, batch_size = 64, n_steps = 128, policy_kwargs={"lstm_features": 2 * 6})
        else:
            self.model = CustomRPPO.load(self.file_path)

    def reset(self) -> None:
        """Reset the agent's LSTM states and episode_start flag."""
        self.episode_starts = np.ones((1,), dtype=bool)
        self.lstm_states = None

    def predict(self, obs):
        """
        Predict an action given an observation, while maintaining LSTM states.
        The `episode_start` flag ensures that the recurrent network resets at the start of an episode.
        """
        action, self.lstm_states = self.model.predict(
            obs,
            state=self.lstm_states,
            episode_start=self.episode_starts,
            deterministic=True
        )
        self.episode_starts = np.zeros((1,), dtype=bool)
        
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, total_timesteps, log_interval: int = 1, verbose=0):
        """
        Set the environment, adjust verbosity, and begin training.
        """
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
    def get_policy(self):
        return self.model.policy


def test_trained_agent(agent, env, n_episodes=1):
    """
    Run one or more episodes with the trained agent and record the predicted timings.
    Returns a list of predicted timing sequences (one per episode).
    """
    episodes_timings = []
    for episode in range(n_episodes):
        obs = env.reset()  # Reset environment
        agent.reset()      # Reset agent's LSTM states
        done = False
        total_reward = 0.0
        predicted_timings = []
        
        while not done:
            action = agent.predict(obs)
            obs, reward, done, info = env.step(action)
            predicted_timings.append(info[0]["predicted_timing"])
            # print(f"Prediction: {info[0]['predicted_timing']}, Reward: {reward}, Action: {action}")  
            total_reward += reward

        episodes_timings.append(predicted_timings)
    print(f"Total reward: {total_reward}")
    return episodes_timings


# ----------------------------
# Main Testing Script
# ----------------------------
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train a model to accompany a soloist.')
    parser.add_argument('--traintest', '-t', type=str, help='1 to train, 2 to test', required=True)
    parser.add_argument('--output_midi_file', '-o', type=str, help='output midi file', default = "../assets/adjusted_output.mid")
    args = parser.parse_args()

    date = "0313"
    model_number = "01"
    window_size = 7

    data = prepare_tensor("../assets/real_chopin.mid", "../assets/reference_chopin.mid")

    
    # Uncomment these lines to train/save the model if needed.
    if args.traintest == '1':
        env = DummyVecEnv([lambda: MusicAccompanistEnv(data[1:, :], 'all', "rppoconfig.json", "difference")])
        agent = RecurrentPPOAgent(env)
        agent.learn(total_timesteps=200000, log_interval=10, verbose=1)
        agent.save(save_model(date, model_number))

    elif args.traintest == '2':
        env = DummyVecEnv([lambda: MusicAccompanistEnv(data[1:, :], 'all', "rppoconfig.json", "ratio")])
        agent = RecurrentPPOAgent(env)
        agent.model = agent.model.load(f"../models/{date}/{date}_{model_number}")
        episodes_timings = test_trained_agent(agent, env, n_episodes=1)
        predicted_timings = episodes_timings[0]
        write_midi_from_timings(predicted_timings, data[0, :], window_size, output_midi_file="../assets/adjusted_output.mid", default_duration=0.3)
    
    elif args.traintest == '3':
        env = DummyVecEnv([lambda: MusicAccompanistEnv(data[1:, :], [277, 330], "rppoconfig.json", "2row_with_ratio")])
        agent = RecurrentPPOAgent(env)
        agent.model = agent.model.load(f"../models/{date}/{date}_{model_number}")
        episodes_timings = test_trained_agent(agent, env, n_episodes=1)
        predicted_timings = episodes_timings[0]
    
    elif args.traintest == '4':
        env = DummyVecEnv([lambda: MusicAccompanistEnv(data[1:, :], [277, 330], "rppoconfig.json", "2row_with_ratio")])
        agent = RecurrentPPOAgent(env)
        agent.model = agent.model.load(f"../models/{date}/{date}_{model_number}")

        print(agent.get_policy())



