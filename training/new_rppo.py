# New RPPO with two LSTM layers 


import gymnasium
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from typing import Optional
import argparse
from torchsummary import summary

from data_processing import prepare_tensor
from utils.midi_utils import write_midi_from_timings
from utils import OBS_PREP_DIC
from utils.files import save_model, save_memory
from rl.custom_network import CustomRPPO
from rl.memory import initialize_memory_file, store_memory, memory_noise



class MusicAccompanistEnv(gymnasium.Env):
    """
    Observations: A sliding window of historical data plus a future reference timing.
    """
    def __init__(self, data, windows, window_size, option, memory_file = 'rl/memory.h5', write_to_memory = 0.3):
        super(MusicAccompanistEnv, self).__init__()
        self.data = data 
        self.n_notes = self.data.shape[1]
        self.option = option

        self.window_size = window_size
        self.windows = windows
        self.current_index = self.window_size
        self.write_to_memory = write_to_memory
        self.memory_file = memory_file

        if option in ["difference", "2row_with_ratio", "ratio", "normalized_reference"]:
            self.observation_space = spaces.Box(low=0, high=10.0, shape=(self.window_size * 2 - 1,), dtype=np.float32)
        elif option == "raw":
            flat_dim = 2 * self.window_size + 1
            self.observation_space = spaces.Box(low=0, high=30.0, shape=(flat_dim,), dtype=np.float32)
        else:
            flat_dim = 2 * self.window_size 
            self.observation_space = spaces.Box(low=0, high=5.0, shape=(flat_dim,), dtype=np.float32)
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.forecast_window = 3
    
    def obs_prep(self, reset):
        if reset:
            self.current_index = self.window_size
        return OBS_PREP_DIC[self.option](self.data, self.current_index, self.window_size)
        
    
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

        predicted_timing = ref_timing * speed_factor # * base_scale

        reward = self.new_reward_function(solo_timing, ref_timing, action[0])

        if np.random.rand() < self.write_to_memory:
            obs = self.obs_prep(False)
            first_note = self.data[:, self.current_index - self.window_size:self.current_index].astype(np.float32)
            second_note = self.data[:, self.current_index - self.window_size + 1:self.current_index + 1].astype(np.float32)
            memory = second_note - first_note
            memory = memory[0] / (memory[1] + 1e-8)
            new_obs = memory_noise(memory, 0.05)
            store_memory(piece_index = self.current_index - self.window_size, memory_vector= new_obs, filename = self.memory_file)
        
        self.current_index += 1
        if self.windows == 'all':
            done = (self.current_index >= self.n_notes - self.forecast_window)
        else:
            done = (self.current_index > self.windows[1] - self.forecast_window)
        
        if not done:
            obs = self.obs_prep(False)
            # if self.current_index % 200 == 0:
            #     print(f"Current index: {self.current_index}, Reward: {reward}, Action: {action}")
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            
        info = {"predicted_timing": predicted_timing}
        # Update to match current gymnasium API
        return obs, reward, done, False, info

    def new_reward_function(self, solo_timing, ref_timing, action):
        epsilon = 1e-8
        ideal_log_action = np.log((solo_timing) / (ref_timing + epsilon))
        
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
            self.model = CustomRPPO.load(path = self.file_path)

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

def parse_windows(value):
    if value.lower() == "all":
        return value.lower()
    parts = value.split(',')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Argument must be 'all' or two integers separated by a comma, e.g., '277,330'")
    try:
        ints = [int(part) for part in parts]
    except ValueError:
        raise argparse.ArgumentTypeError("Both values must be integers.")
    return ints

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
    parser.add_argument('--traintest', '-t', type=str, help='1 to train, 2 to test, 3 to test for a small window, 4 to get policy summary', required=True)
    parser.add_argument('--output_midi_file', '-o', type=str, help='output midi file', default = "../assets/adjusted_output.mid")
    parser.add_argument(
        '--windows', '-w',
        type=parse_windows,
        default='all',
        help="Specify 'all' or two integers separated by a comma (e.g. '277,330')"
    )    
    parser.add_argument("--memory_reset", action="store_true", help="Resets the memory file")
    args = parser.parse_args()


    date = "0319"
    model_number = "02"
    window_size = 7
    windows = args.windows
    memory_write_prob = 0.5

    model_name = save_model(date, model_number)
    memory_name = save_memory(date, model_number)

    if args.memory_reset:
        confirm = input("Are you sure you want to delete the memory? (y/N): ")
        if confirm.lower() not in ['y', 'yes']:
            print("Operation cancelled.")
            exit(1)
        initialize_memory_file(memory_name)
        
        print("Memory reset successful.")


    data = prepare_tensor("../assets/real_chopin.mid", "../assets/reference_chopin.mid")
    fed_data = data[1:, :] # Remove the pitches for now
    env = DummyVecEnv([lambda: MusicAccompanistEnv(fed_data, windows, window_size, "memory_enhanced", memory_file = memory_name, write_to_memory=memory_write_prob)])
    agent = RecurrentPPOAgent(env)

    # Uncomment these lines to train/save the model if needed.
    if args.traintest == '1': # Train
        agent.model.policy.memory_file = memory_name
        agent.learn(total_timesteps=50000, log_interval=10, verbose=1)
        agent.save(save_model(date, model_number))

    elif args.traintest == '2': # Testing
        agent.model = agent.model.load(model_name)
        agent.model.policy.memory_file = memory_name
        episodes_timings = test_trained_agent(agent, env, n_episodes=1)
        predicted_timings = episodes_timings[0]
        write_midi_from_timings(predicted_timings, data[0, :], window_size, output_midi_file="../assets/adjusted_output.mid", default_duration=0.3)
    
    elif args.traintest == '3': # Testing with specified windows size
        agent.model = agent.model.load(model_name)
        agent.model.policy.memory_file = memory_name
        episodes_timings = test_trained_agent(agent, env, n_episodes=1)
        predicted_timings = episodes_timings[0]
    
    elif args.traintest == '4': # getting summary of model architecture
        agent.model = agent.model.load(model_name)
        state_dict = agent.model.policy.state_dict()
        for name, param in state_dict.items():
            print(name, param.size())
            if name == 'post_concat_layer.0.weight':
                print(param)






