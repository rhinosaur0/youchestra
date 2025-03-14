from typing import Optional, Union, Any

from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy, MlpLstmPolicy, MultiInputLstmPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.utils import zip_strict


from torch import nn
import torch as th
from gymnasium import spaces

class CustomRecurrentACP(RecurrentActorCriticPolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        lstm_hidden_size: int = 64,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[dict[str, Any]] = None,
        lstm_features: int = 7,
    ):
        self.lstm_features = lstm_features
        self.lstm_output_dim = lstm_hidden_size

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            shared_lstm,
            enable_critic_lstm,
            lstm_kwargs,
        )

        self.lstm_kwargs = lstm_kwargs or {}
        self.shared_lstm = shared_lstm
        self.enable_critic_lstm = enable_critic_lstm
        # self.features_dim = 14
        self.lstm_actor = nn.LSTM(
            2, # remove the future reference
            lstm_hidden_size,
            num_layers=n_lstm_layers,
            **self.lstm_kwargs,
        )
        # For the predict() method, to initialize hidden states
        # (n_lstm_layers, batch_size, lstm_hidden_size)
        self.lstm_hidden_state_shape = (n_lstm_layers, 1, lstm_hidden_size)
        self.critic = None
        self.lstm_critic = None
        assert not (
            self.shared_lstm and self.enable_critic_lstm
        ), "You must choose between shared LSTM, seperate or no LSTM for the critic."

        assert not (
            self.shared_lstm and not self.share_features_extractor
        ), "If the features extractor is not shared, the LSTM cannot be shared."

        # No LSTM for the critic, we still need to convert
        # output of features extractor to the correct size
        # (size of the output of the actor lstm)
        if not (self.shared_lstm or self.enable_critic_lstm):
            self.critic = nn.Linear(self.features_dim, lstm_hidden_size)

        # Use a separate LSTM for the critic
        if self.enable_critic_lstm:
            self.lstm_critic = nn.LSTM(
                2,
                lstm_hidden_size,
                num_layers=n_lstm_layers,
                **self.lstm_kwargs,
            )

        self.post_concat_layer = nn.Sequential(
            nn.Linear(lstm_hidden_size + 1, lstm_hidden_size),
            nn.ReLU()
        )
        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


    def obs_split(self, obs):
        return obs[:, :-1], obs[:, -1]

    # @staticmethod
    def _process_sequence(self, 
        features: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        lstm: nn.LSTM,
    ) -> tuple[th.Tensor, th.Tensor]:
        """
        Do a forward pass in the LSTM network.

        :param features: Input tensor
        :param lstm_states: previous hidden and cell states of the LSTM, respectively
        :param episode_starts: Indicates when a new episode starts,
            in that case, we need to reset LSTM states.
        :param lstm: LSTM object.
        :return: LSTM output and updated LSTM states.
        """
        # LSTM logic
        # (sequence length, batch size, features dim)
        # (batch size = n_envs for data collection or n_seq when doing gradient update)
        n_seq = lstm_states[0].shape[1]
        # Batch to sequence
        # (padded batch size, features_dim) -> (n_seq, max length, features_dim) -> (max length, n_seq, features_dim)
        # note: max length (max sequence length) is always 1 during data collection

        #original features [[nth_time_step], [n+1th_time_step], ..... [kth_time_step], [k+1th_time_step]]
        features_sequence = features.reshape((n_seq, -1, self.lstm_features + 1)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        # print(f'features_sequence: {features_sequence.shape}, episode_starts: {episode_starts.shape}')

        # If we don't have to reset the state in the middle of a sequence
        # we can avoid the for loop, which speeds up things
        # if th.all(episode_starts == 0.0):

        #     # lstm states should be [1, 2, 64] or [1, 1, 64]

        #     features, future_feature = self.obs_split(features)
        #     batch_size = features.shape[0] // n_seq
        #     features = features.view(-1, 2, 6)
        #     print(features[0])
        #     features = features.permute(2, 0, 1)
        #     print(features[:, :7, :])

        #     # print(features[[0, batch_size, 2 * batch_size, 3 * batch_size, 4 * batch_size, 5 * batch_size], 0, :])
        #     lstm_output, lstm_states = lstm(features, lstm_states)
        #     # print(f'lstm_output: {lstm_output.shape}')
            
        #     lstm_output = lstm_output[-1]
        #     # lstm_output = th.flatten(lstm_output, start_dim=0, end_dim=1)
        #     # print(f'lstm_output after flatten: {lstm_output.shape}')
        #     # print(f'lstm_states after flatten: {lstm_states[0].shape}')
        #     return lstm_output, lstm_states

        lstm_output = []

        for features, episode_start in zip_strict(features_sequence, episode_starts):
            batch_size = features.shape[0]
            features, future_feature = self.obs_split(features)
            features = features.view(batch_size, 2, 6).permute(2, 0, 1)


            hidden, lstm_states = lstm(
                features,
                (
                    # Reset the states at the beginning of a new episode
                    (1.0 - episode_start).view(1, batch_size, 1) * lstm_states[0],
                    (1.0 - episode_start).view(1, batch_size, 1) * lstm_states[1],
                ),
            )
            
            hidden_last = hidden[-1]  # shape: [batch_size, 64]
            # combined = th.cat([hidden_last, future_feature.unsqueeze(1)], dim=1)  
            # final_hidden = self.post_concat_layer(combined)  
            
            lstm_output += [hidden_last.unsqueeze(0)]
        
        lstm_output = th.flatten(th.cat(lstm_output).transpose(0, 1), start_dim=0, end_dim=1)

        return lstm_output, lstm_states
    

class CustomRPPO(RecurrentPPO):
    policy_aliases = {
        "MlpLstmPolicy": MlpLstmPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "MultiInputLstmPolicy": MultiInputLstmPolicy,
        "Custom": CustomRecurrentACP
    }


