import gym
from gym import spaces
import numpy as np
from typing import Optional
import json
from math import log
import argparse

from data_processing import prepare_tensor
from utils.midi_utils import write_midi_from_timings
from utils.files import save_model



class MusicAccompanistEnv(gym.Env):
    """
    
    Observations: A sliding window (3 x window_size) from the input data.
       - Row 0: Reference pitch
       - Row 1: Soloist pitch timing
       - Row 2: Reference pitch's metronomic timing
    """
    def __init__(self, data, windows, config_file, option):
        super(MusicAccompanistEnv, self).__init__()
        self.data = data 
        self.n_notes = self.data.shape[1]
        self.option = option

        with open(config_file, 'r') as f:
            self.config = json.load(f)

        self.obs_space = self.config['option'][option]['observation_space']
        self.window_size = self.config['window_size']
        self.windows = windows
        self.current_index = self.window_size
        self.observation_space = spaces.Box(low = 0.0, high = 10.0, shape = self.obs_space, dtype = np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def reset(self):
        if self.windows == 'all':
            obs = self.obs_prep(True)
        else:
            self.current_index = self.windows[0]
            obs = self.obs_prep(False)
        return obs
    
    def obs_prep(self, reset):
        if reset:
            self.current_index = self.window_size
        match self.option:
            case "2row":
                next_window = self.data[:, self.current_index - self.window_size:self.current_index].astype(np.float32)
                next_window = next_window - next_window[:, 0:1]
                return next_window
            case "1row+next":
                next_window = self.data[:, self.current_index - self.window_size:self.current_index].astype(np.float32)
                next_window = next_window - next_window[:, 0:1] # normalize by setting the first time to 0
                next_pred_note = self.data[1, self.current_index]
                return np.concatenate([next_window.flatten(), [next_pred_note]])
            case "2row_normalized":
                first_note = self.data[:, self.current_index - self.window_size:self.current_index - 1].astype(np.float32)
                second_note = self.data[:, self.current_index - self.window_size + 1:self.current_index].astype(np.float32)
                next_window = second_note - first_note
                return next_window
            case "2row_with_next":
                first_note = self.data[:, self.current_index - self.window_size:self.current_index - 1].astype(np.float32)
                second_note = self.data[:, self.current_index - self.window_size + 1:self.current_index].astype(np.float32)
                next_window = second_note - first_note
                extra_next_timing = np.array([self.data[1, self.current_index] - self.data[1, self.current_index - 1]], dtype=np.float32)
                extra_column = np.vstack([np.zeros_like(extra_next_timing), extra_next_timing])  # shape: (2, 1)
                observation = np.concatenate([next_window, extra_column], axis=1)  # shape: (2, window_size)
                return observation
            case "2row_with_ratio":
                first_note = self.data[:, self.current_index - self.window_size:self.current_index - 1].astype(np.float32)
                second_note = self.data[:, self.current_index - self.window_size + 1:self.current_index].astype(np.float32)
                next_window = second_note - first_note
                next_window[0] = next_window[0] / next_window[1]
                extra_next_timing = np.array([self.data[1, self.current_index] - self.data[1, self.current_index - 1]], dtype=np.float32)
                extra_column = np.vstack([np.zeros_like(extra_next_timing), extra_next_timing])  # shape: (2, 1)
                observation = np.concatenate([next_window, extra_column], axis=1)  # shape: (2, window_size)
                return observation


    def step(self, action):
        """
        Apply the speed adjustment factor to the reference timing,
        then compute the reward based on how close the predicted timing (ref_timing * action)
        is to the soloist's actual timing.
        """
        # Extract the current reference and soloist timing
        ref_timing = self.data[1, self.current_index] - self.data[1, self.current_index - 1]
        solo_timing = self.data[0, self.current_index] - self.data[0, self.current_index - 1]
        speed_factor = np.exp(action[0])
        predicted_timing = ref_timing * speed_factor

        reward = self.reward_function(predicted_timing, solo_timing)
        
        self.current_index += 1
        if self.windows == 'all':
            done = (self.current_index >= self.n_notes)
        else:
            done = (self.current_index > self.windows[1])
        
        if not done:
            obs = self.obs_prep(False)
            if self.current_index % 200 == 0:
                print(f"Current index: {self.current_index}, Reward: {reward}, Action: {action}")
        else:
            obs = np.zeros(self.obs_space, dtype=np.float32)
        info = {"predicted_timing": predicted_timing}
        return obs, reward, done, info

    def reward_function(self, predicted_timing, solo_timing):
        epsilon = 1e-8
        ratio_diff = 5 * log((predicted_timing + epsilon) / (solo_timing + epsilon)) ** 2
        reward = -ratio_diff
        return reward

    def render(self, mode='human'):
        pass


from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

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
            self.model = RecurrentPPO("MlpLstmPolicy", self.env, verbose=0)
        else:
            self.model = RecurrentPPO.load(self.file_path)

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
            predicted_timings.append(info[0].get("predicted_timing"))
            print(f"Prediction: {info[0].get('predicted_timing')}, Reward: {reward}, \nObs: {obs}")  
            total_reward += reward[0]
            # print(f"Episode: {episode+1}, Action: {action}, Reward: {reward[0]:.4f}, Predicted Timing: {info[0].get('predicted_timing'):.4f}, Note: {info[0].get('note')}")


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

    date = "0308"
    model_number = "11"
    window_size = 7

    data = prepare_tensor("../assets/real_chopin.mid", "../assets/reference_chopin.mid")



    
    # Uncomment these lines to train/save the model if needed.
    if args.traintest == '1':
        env = DummyVecEnv([lambda: MusicAccompanistEnv(data[1:, :], 'all', "rppoconfig.json", "2row_with_ratio")])
        agent = RecurrentPPOAgent(env)
        agent.learn(total_timesteps=200000, log_interval=10, verbose=1)
        agent.save(save_model(date, model_number))
    elif args.traintest == '2':
        env = DummyVecEnv([lambda: MusicAccompanistEnv(data[1:, :], 'all', "rppoconfig.json", "2row_with_ratio")])
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



        



