from typing import Optional, Union, Any

from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy, MlpLstmPolicy, MultiInputLstmPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
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
        window_size: int = 7,
    ):
        self.window_size = window_size
        self.lstm_output_dim = lstm_hidden_size
        print(lstm_hidden_size)
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

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        ### NEW

        # self.concat_future_lstm = nn.Sequential(
        #     nn.Linear(lstm_hidden_size + some_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1)
        # )
        self.post_concat_layer = nn.Sequential(
            nn.Linear(lstm_hidden_size + 1, lstm_hidden_size),
            nn.ReLU()
        )

    def obs_split(self, obs):
        historical = obs[:, :2*self.window_size].reshape(obs.shape[0], self.window_size, 2)
        future_feature = obs[:, -1:]
        return historical, future_feature

    # @staticmethod
    def _process_sequence(self, 
        features: th.Tensor,
        future_feature: th.Tensor, # The extra note at the end of the sequence for reference
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
        features_sequence = features.reshape((n_seq, -1, 14)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        # print(f'features_sequence: {features_sequence.shape}, episode_starts: {episode_starts.shape}')

        # If we don't have to reset the state in the middle of a sequence
        # we can avoid the for loop, which speeds up things
        # if th.all(episode_starts == 0.0):
        #     print('th.all moment')
        #     lstm_output, lstm_states = lstm(features_sequence, lstm_states)
        #     lstm_output = th.flatten(lstm_output.transpose(0, 1), start_dim=0, end_dim=1)
        #     return lstm_output, lstm_states

        lstm_output = []

        for features, episode_start in zip_strict(features_sequence, episode_starts):
            # print(f'lstm_states: {lstm_states[0].shape}, {lstm_states[1].shape}')
            batch_size = features.shape[0]
            features = features.reshape(2, batch_size, 7).permute(2, 1, 0)


            # print(f'features: {features.shape}, episode_start: {episode_start.shape}')


            hidden, lstm_states = lstm(
                features,
                (
                    # Reset the states at the beginning of a new episode
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[0],
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[1],
                ),
            )

            # print(f'hidden: {hidden.shape}, lstm_states: {lstm_states[0].shape}, {lstm_states[1].shape}')
            
            hidden_last = hidden[-1]  # shape: [batch_size, 256]
            combined = th.cat([hidden_last, future_feature], dim=1)  
            final_hidden = self.post_concat_layer(combined)  
            
            lstm_output += [final_hidden.unsqueeze(0)]
        
        # Sequence to batch
        # (sequence length, n_seq, lstm_out_dim) -> (batch_size, lstm_out_dim)
        lstm_output = th.flatten(th.cat(lstm_output).transpose(0, 1), start_dim=0, end_dim=1)
        # print(f'lstm_output: {lstm_output.shape}')

        return lstm_output, lstm_states

    def forward(
            self, 
            obs: th.Tensor,
            lstm_states: RNNStates,
            episode_starts: th.Tensor,
            deterministic: bool = False,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
        # New forward method

        # split from (2 * n + 1) to (n, 2) and (1,)
        historical, future_feature = self.obs_split(obs)
        
        
        if self.share_features_extractor:
            pi_features = vf_features = historical  # alis
        else:
            raise NotImplementedError
        

        latent_pi, lstm_states_pi = self._process_sequence(pi_features, future_feature, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(vf_features, future_feature, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            # Re-use LSTM features but do not backpropagate
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            # Critic only has a feedforward network
            latent_vf = self.critic(vf_features)
            lstm_states_vf = lstm_states_pi


        print(latent_pi.shape, future_feature.shape)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)
    
    # TODO

    def get_distribution(self, observation, lstm_states, episode_starts):
        historical, future_feature = self.obs_split(observation)

        features = super(RecurrentActorCriticPolicy, self).extract_features(historical, self.pi_features_extractor)
        latent_pi, lstm_states = self._process_sequence(features, future_feature, lstm_states, episode_starts, self.lstm_actor)
        # print(latent_pi.shape, future_feature.shape)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        return self._get_action_dist_from_latent(latent_pi), lstm_states

    def predict_values(self, observation, lstm_states, episode_starts):
        features, future_feature = self.obs_split(observation)
        
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(features, future_feature, lstm_states, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            # Use LSTM from the actor
            latent_pi, _ = self._process_sequence(features, future_feature, lstm_states, episode_starts, self.lstm_actor)
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)

    def evaluate_actions(self, observation, actions, lstm_states, episode_starts):
        features, future_feature = self.obs_split(observation)

        if self.share_features_extractor:
            pi_features = vf_features = features  # alias
        else:
            pi_features, vf_features = features
        latent_pi, _ = self._process_sequence(pi_features, future_feature, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(vf_features, future_feature, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def predict(self, observation, state, episode_starts, deterministic):
        raise NotImplementedError



class CustomRPPO(RecurrentPPO):
    policy_aliases = {
        "MlpLstmPolicy": MlpLstmPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "MultiInputLstmPolicy": MultiInputLstmPolicy,
        "Custom": CustomRecurrentACP
    }


