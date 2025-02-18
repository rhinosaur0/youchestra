import gym
from gym import spaces
import numpy as np
from typing import Optional
from data_processing import prepare_tensor
from math import log

class MusicAccompanistEnv(gym.Env):
    """
    A custom Gym environment for a music accompanist.
    
    Observations: A sliding window (4 x window_size) from the input data.
       - Row 0: Reference pitch
       - Row 1: Soloist pitch timing
       - Row 2: Reference pitch's metronomic timing
    
    Action: A continuous speed adjustment factor (e.g., between 0.3 and 3.0).
    """
    def __init__(self, data, window_size=10):
        super(MusicAccompanistEnv, self).__init__()
        self.data = data 
        self.n_notes = self.data.shape[1]
        self.window_size = window_size
        self.current_index = window_size  

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2, window_size - 1), dtype=np.float32
        )
        # Action: A single continuous speed factor.
        self.action_space = spaces.Box(
            low=0.3, high=3.0, shape=(1,), dtype=np.float32
        )

    def reset(self):
        self.current_index = self.window_size
        return self.data[:, self.current_index - self.window_size:self.current_index - 1].astype(np.float32)

    def step(self, action):
        """
        Apply the speed adjustment factor to the reference timing,
        then compute the reward based on how close the predicted timing (ref_timing * action)
        is to the soloist's actual timing.
        """
        # Extract the current reference and soloist timing
        ref_timing = self.data[1, self.current_index] - self.data[1, self.current_index - 1]
        solo_timing = self.data[0, self.current_index] - self.data[0, self.current_index - 1]
        speed_factor = action[0]
        predicted_timing = ref_timing * speed_factor

        reward = self.reward_function(predicted_timing, solo_timing)
        
        self.current_index += 1
        done = (self.current_index >= self.n_notes)
        
        if not done:
            obs = self.data[:, self.current_index - self.window_size:self.current_index - 1].astype(np.float32)
            obs = obs - obs[:, 0:1]
            if self.current_index % 100 == 0:
                print(f"Current observation: {obs}, Reward: {reward}, Action: {action}")
        else:
            obs = np.zeros((2, self.window_size - 1), dtype=np.float32)
        info = {}
        return obs, reward, done, info

    def reward_function(self, predicted_timing, solo_timing):
        ratio_diff = log(predicted_timing / solo_timing) ** 2
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
            # Create the RecurrentPPO model using an LSTM policy.
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
        # After the first call in an episode, set episode_starts to False.
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


def test_trained_agent(agent, env, n_episodes=5):
    for episode in range(n_episodes):
        obs = env.reset()  # Reset the environment for a new episode
        agent.reset()      # Reset agent's LSTM states and episode_start flag
        done = False
        total_reward = 0.0
        step_count = 0
        
        while not done:
            # Use the agent to predict an action given the current observation.
            action = agent.predict(obs)
            
            # Take a step in the environment with the predicted action.
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]  # reward is wrapped in an array because of DummyVecEnv
            step_count += 1
            
            print(f"Episode: {episode+1}, Step: {step_count}, Action: {action}, Reward: {reward[0]}")
        
        print(f"Episode {episode+1} finished with total reward: {total_reward}\n")


if __name__ == "__main__":
    # Generate dummy 4 x n_notes data for demonstration.
    data = prepare_tensor("assets/real_chopin.mid", "assets/reference_chopin.mid").T

    env = DummyVecEnv([lambda: MusicAccompanistEnv(data[1:, :], window_size=10)])
    agent = RecurrentPPOAgent(env)
    
    # agent.learn(total_timesteps=10000, log_interval=10, verbose=1)
    # agent.save("recurrent_ppo_music_accompanist")

    agent.model = agent.model.load("recurrent_ppo_music_accompanist")
    test_trained_agent(agent, env, n_episodes=5) 
    
