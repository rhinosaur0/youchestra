import gym
from gym import spaces
import numpy as np

# Define the custom environment.
class AccompanistTimingEnv(gym.Env):
    """
    A simple simulation environment for training an RL agent to adjust an accompanist’s speed.
    At each note event, the agent chooses a speed factor.
    The expected timing of the next soloist note is computed as:
        expected_time = reference_time / speed_factor
    The soloist note time is simulated as:
        actual_time = reference_time + noise
    The reward is negative absolute error between the expected and actual note time.
    """
    def __init__(self, num_notes=50, nominal_interval=1.0, noise_std=0.1):
        super(AccompanistTimingEnv, self).__init__()
        self.num_notes = num_notes
        self.nominal_interval = nominal_interval
        self.noise_std = noise_std

        # Action: the agent picks a speed factor in a continuous range.
        # For example, 0.5 means slower, 1.0 is neutral, 1.5 is faster.
        self.action_space = spaces.Box(low=np.array([0.5]), high=np.array([1.5]), dtype=np.float32)

        # Observation: a 3-dimensional vector.
        #   [normalized note index, last error, current speed factor]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        self.reset()

    def reset(self):
        # Reset note index and generate a new episode.
        self.current_note_index = 0
        
        # Create reference times: note i's reference time is (i+1) * nominal_interval.
        self.reference_times = np.array([(i+1) * self.nominal_interval for i in range(self.num_notes)])
        # Simulate the soloist's note times by adding Gaussian noise.
        self.soloist_times = self.reference_times + np.random.normal(0, self.noise_std, size=self.num_notes)
        
        # Initialize the last error to 0 and starting speed factor to 1.0 (neutral tempo).
        self.last_error = 0.0
        self.current_speed = 1.0

        # Observation: normalized note index, last error, and current speed factor.
        obs = np.array([self.current_note_index / self.num_notes, self.last_error, self.current_speed], dtype=np.float32)
        return obs

    def step(self, action):
        # Clip the action to the allowed range.
        speed_factor = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        
        # Get the reference time for the current note.
        ref_time = self.reference_times[self.current_note_index]
        # Compute the expected time for the soloist note if the accompanist applies the speed factor.
        # (A higher speed factor speeds up the performance, so expected time decreases.)
        expected_time = ref_time / speed_factor
        
        # The actual soloist note time is simulated as the reference time plus some noise.
        actual_time = self.soloist_times[self.current_note_index]
        # Compute the error.
        error = actual_time - expected_time
        
        # Reward: we want to minimize the absolute error.
        reward = -abs(error)
        
        # Update the environment state for the next step.
        self.last_error = error
        self.current_speed = speed_factor
        self.current_note_index += 1
        
        done = self.current_note_index >= self.num_notes
        
        if not done:
            obs = np.array([self.current_note_index / self.num_notes, self.last_error, self.current_speed], dtype=np.float32)
        else:
            # End-of-episode observation.
            obs = np.array([1.0, self.last_error, self.current_speed], dtype=np.float32)
        
        # Additional info (optional).
        info = {
            'expected_time': expected_time,
            'actual_time': actual_time,
            'error': error,
            'note_index': self.current_note_index
        }
        return obs, reward, done, info

# ------------------------------------------------------------------------------
# Training the RL Agent using PPO from stable-baselines3
# ------------------------------------------------------------------------------

# Make sure to install stable-baselines3 if you haven't already:
# pip install stable-baselines3[extra]

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# For faster training, we use a vectorized environment (multiple copies running in parallel).
env_id = lambda: AccompanistTimingEnv(num_notes=50, nominal_interval=1.0, noise_std=0.1)
vec_env = make_vec_env(env_id, n_envs=4)

# Create the PPO model using an MLP policy.
model = PPO("MlpPolicy", vec_env, verbose=1)

# Train the model. Increase total_timesteps as needed.
model.learn(total_timesteps=100_000)

# Save the trained model.
model.save("accompanist_timing_model")

# ------------------------------------------------------------------------------
# Example usage: Testing the trained agent in a single environment instance.
# ------------------------------------------------------------------------------

# Create a single instance of the environment.
env = AccompanistTimingEnv(num_notes=50, nominal_interval=1.0, noise_std=0.1)
obs = env.reset()
done = False

print("\n--- Testing the trained agent ---\n")
while not done:
    # Predict the next action (speed factor) using the trained policy.
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(f"Note {info['note_index']}:")
    print(f"  Action (speed factor): {action[0]:.3f}")
    print(f"  Expected time: {info['expected_time']:.3f} | Actual time: {info['actual_time']:.3f}")
    print(f"  Error: {info['error']:.3f} | Reward: {reward:.3f}\n")
