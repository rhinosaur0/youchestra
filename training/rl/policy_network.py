import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
from typing import Callable


class TempoPredictor(nn.Module):
    def __init__(self, hist_input_dim, hidden_dim, decoder_input_dim=1):
        """
        hist_input_dim: number of features in historical data (e.g., 2: [real_diff, ref_diff])
        hidden_dim: hidden size for the encoder LSTM
        decoder_input_dim: number of features for the future reference timing (typically 1)
        """
        super().__init__()
        self.encoder = nn.LSTM(hist_input_dim, hidden_dim, batch_first=True)
        # A simple feed-forward layer that maps the concatenated context and future ref to a prediction

        self.latent_dim_pi = hidden_dim + decoder_input_dim
        self.latent_dim_vf = hidden_dim + decoder_input_dim
        self.fc_actor = nn.Sequential(
            nn.Linear(hidden_dim + decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.fc_critic = nn.Sequential(
            nn.Linear(hidden_dim + decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # For compatibility with SB3's policy interface, we need to handle a single tensor input
        # x shape: (batch, features)
        batch_size = x.shape[0]
        # Extract the future reference (last element)
        future_ref = x[:, -1:]  # Shape: (batch, 1)
        
        # Extract and reshape historical data
        # All but the last element are historical data
        historical_features = x.shape[1] - 1
        window_length = historical_features // 2  # 2 rows of data
        
        historical_data = x[:, :historical_features].reshape(batch_size, window_length, 2)
        
        return self.forward_actor(historical_data, future_ref), self.forward_critic(historical_data, future_ref)
    
    def forward_actor(self, historical_data, future_ref):
        _, (h, _) = self.encoder(historical_data)  # h shape: (num_layers, batch, hidden_dim)
        # Use the final layer's hidden state as context (shape: (batch, hidden_dim))
        context = h[-1]
        combined = torch.cat([context, future_ref], dim=1)
        
        prediction = self.fc_actor(combined)
        return prediction
    
    def forward_critic(self, historical_data, future_ref):
        _, (h, _) = self.encoder(historical_data)  # h shape: (num_layers, batch, hidden_dim)
        # Use the final layer's hidden state as context (shape: (batch, hidden_dim))
        context = h[-1]
        combined = torch.cat([context, future_ref], dim=1)
        prediction = self.fc_critic(combined)
        return prediction
    
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 lr_schedule: Callable[[float], float],
                 *args,
                 **kwargs):
        # Make sure we pass all arguments to the parent class
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        # Get the observation dimension from the observation space
        if isinstance(self.observation_space, spaces.Box):
            obs_dim = int(np.prod(self.observation_space.shape))
        else:
            raise ValueError(f"Unsupported observation space: {self.observation_space}")
            
        # For our case with a window of shape (window_size, 2) plus one extra feature
        hist_input_dim = 2  # Two rows of data
        # Assuming the last element is the future reference
        self.mlp_extractor = TempoPredictor(hist_input_dim=hist_input_dim, hidden_dim=64, decoder_input_dim=1)



class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, hidden_dim, n):
        # observation_space is expected to be a Box with shape (2*n + 1,)
        super().__init__(observation_space, features_dim=hidden_dim + 1)
        self.n = n
        self.hidden_dim = hidden_dim
        
        # LSTM for processing historical data (first 2*n features)
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_dim, batch_first=True)
        # FC layer after concatenating LSTM output with the extra feature
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1)  # output dimension as needed
        )
        
    def forward(self, observations):
        # observations shape: (batch_size, 2*n + 1)
        batch_size = observations.shape[0]
        historical = observations[:, :2*self.n].reshape(batch_size, self.n, 2)
        extra_feature = observations[:, -1:]  # shape: (batch_size, 1)
        
        # Process historical data with LSTM. We only need the final hidden state.
        _, (h_n, _) = self.lstm(historical)  # h_n shape: (1, batch_size, hidden_dim)
        lstm_out = h_n[-1]  # shape: (batch_size, hidden_dim)
        
        # Concatenate LSTM output with extra feature
        combined = torch.cat([lstm_out, extra_feature], dim=1)  # shape: (batch_size, hidden_dim + 1)
        
        features = self.fc(combined)
        return features