import gym
from gym import spaces
import numpy as np

class AccompanistLSTMPredictionEnv(gym.Env):
    """
    The agent outputs a continuous speed factor (action), which is applied to the omitted
    reference metronomic timings to predict the soloist timings via:
        predicted_timing = reference_time / speed_factor

    The reward is computed as the negative mean relative error between the predicted and actual
    soloist timings from the omitted portion.
    """
    def __init__(self, episode_data, provided_length, speed_bounds=(0.7, 1.3)):
        """
        Parameters:
        - episode_data: np.ndarray, shape (num_notes, 4) for one episode.
        - provided_length: int, the number of rows (notes) provided to the agent.
        - speed_bounds: tuple, the lower and upper bounds for the speed factor action.
        """
        super(AccompanistLSTMPredictionEnv, self).__init__()
        
        self.episode_data = episode_data  # shape: (num_notes, 4)
        self.provided_length = provided_length
        self.speed_bounds = speed_bounds
        self.num_notes = self.episode_data.shape[0]
        
        if self.provided_length >= self.num_notes:
            raise ValueError("provided_length must be less than the total number of notes in the episode.")
        
        self.omitted_length = self.num_notes - self.provided_length

        # Action space: a single continuous value (the speed factor)
        self.action_space = spaces.Box(low=np.array([speed_bounds[0]], dtype=np.float32),
                                       high=np.array([speed_bounds[1]], dtype=np.float32),
                                       dtype=np.float32)

        # Observation space: the "given" portion of the data with shape (provided_length, 4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(self.provided_length, 4), 
                                            dtype=np.float32)
    
    def reset(self):
        """
        Resets the environment for a new episode.
        Returns:
            The observation: the given portion of the episode data.
        """
        # Split the episode data into given and omitted parts.
        self.given_data = self.episode_data[:self.provided_length]
        self.omitted_data = self.episode_data[self.provided_length:]
        return self.given_data.astype(np.float32)
    
    def step(self, action):
        """
        Takes a step using the agent's action (speed factor) and computes the reward.
        
        Parameters:
            action (array-like): A single-element array representing the speed factor.
        
        Returns:
            observation (np.ndarray): (For a one-step episode, this remains the given data.)
            reward (float): Negative mean relative error between predicted and actual timings.
            done (bool): Whether the episode is finished (True after one step).
            info (dict): Additional diagnostic information.
        """
        # Ensure the action is within bounds.
        speed_factor = float(np.clip(action[0], self.speed_bounds[0], self.speed_bounds[1]))
        
        # Extract the reference metronomic timings from the omitted portion.
        reference_timings = self.omitted_data[:, 1]  # Column 1
        
        # Compute predicted soloist timings by applying the speed factor.
        predicted_timings = reference_timings / speed_factor
        
        # Actual soloist timings are in column 3.
        actual_timings = self.omitted_data[:, 3]
        
        # Compute relative errors. (A small epsilon is added to avoid division by zero.)
        eps = 1e-8
        relative_errors = np.abs(predicted_timings - actual_timings) / (np.abs(actual_timings) + eps)
        mean_relative_error = np.mean(relative_errors)
        
        # Reward is negative mean relative error (we want to minimize the error).
        reward = -mean_relative_error
        
        # Since this environment is a one-decision episode, mark it as done.
        done = True
        
        info = {
            "speed_factor": speed_factor,
            "predicted_timings": predicted_timings,
            "actual_timings": actual_timings,
            "mean_relative_error": mean_relative_error
        }
        
        # In a one-step episode, you might return the same observation (or a dummy value).
        observation = self.given_data.astype(np.float32)
        
        return observation, reward, done, info

    def render(self, mode="human"):
        # Optional: Implement visualization if needed.
        pass

    def close(self):
        pass

episode_data = np.array([
    # [reference_note, reference_timing, soloist_pitch, soloist_timing],
    [60, 1.0, 62, 0.95],
    [62, 2.0, 64, 1.90],
    # ... more rows ...
])

provided_length = 35
env = AccompanistLSTMPredictionEnv(episode_data, provided_length)
obs = env.reset() 