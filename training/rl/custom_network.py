from typing import Optional, Union, Any

from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy, MlpLstmPolicy, MultiInputLstmPolicy
from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer

from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.utils import zip_strict, get_schedule_fn
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.common.recurrent.type_aliases import RecurrentRolloutBufferSamples


import numpy as np
from torch import nn
import torch as th
from gymnasium import spaces


class CustomRecurrentRolloutBuffer(RecurrentRolloutBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def get(self, batch_size: Optional[int] = None):
        '''
        This is modified from the original SB3 code to handle the custom LSTM
        The original treats individual data as sequential data, outputting actions as if each data is a time-step
        However, the design of the accompanist has a fixed window_length of 7, which means that the data is not sequential
        but rather a fixed window of data. This function takes a random distribution of size batch_size from n_steps
        '''
        assert self.full, "Rollout buffer must be full before sampling from it"

        # Prepare the data
        if not self.generator_ready:
            # hidden_state_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
            # swap first to (self.n_steps, self.n_envs, lstm.num_layers, lstm.hidden_size)
            for tensor in ["hidden_states_pi", "cell_states_pi", "hidden_states_vf", "cell_states_vf"]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

            # flatten but keep the sequence order
            # 1. (n_steps, n_envs, *tensor_shape) -> (n_envs, n_steps, *tensor_shape)
            # 2. (n_envs, n_steps, *tensor_shape) -> (n_envs * n_steps, *tensor_shape)
            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
                "episode_starts",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # Sampling strategy that allows any mini batch size but requires
        # more complexity and use of padding
        # Trick to shuffle a bit: keep the sequence order
        # but split the indices in two
        # split_index = np.random.randint(self.buffer_size * self.n_envs)
        # indices = np.arange(self.buffer_size * self.n_envs)
        # indices = np.concatenate((indices[split_index:], indices[:split_index]))

        indices = np.random.permutation(self.buffer_size * self.n_envs)

        env_change = np.zeros(self.buffer_size * self.n_envs).reshape(self.buffer_size, self.n_envs)
        # Flag first timestep as change of environment
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield RecurrentRolloutBufferSamples(
                observations = th.from_numpy(self.observations[batch_inds]).to(self.device),
                actions = th.from_numpy(self.actions[batch_inds]).to(self.device),
                old_values = th.from_numpy(self.values[batch_inds].squeeze(1)).to(self.device),
                old_log_prob = th.from_numpy(self.log_probs[batch_inds].squeeze(1)).to(self.device),
                advantages = th.from_numpy(self.advantages[batch_inds].squeeze(1)).to(self.device),
                returns = th.from_numpy(self.returns[batch_inds].squeeze(1)).to(self.device),
                lstm_states = RNNStates(
                    (th.from_numpy(self.hidden_states_pi[batch_inds]).permute(1, 0, 2).to(self.device), 
                    th.from_numpy(self.cell_states_pi[batch_inds]).permute(1, 0, 2).to(self.device)),
                    (th.from_numpy(self.hidden_states_vf[batch_inds]).permute(1, 0, 2).to(self.device),
                    th.from_numpy(self.cell_states_vf[batch_inds]).permute(1, 0, 2).to(self.device)),
                ),
                episode_starts = th.from_numpy(self.episode_starts[batch_inds].squeeze(1)).to(self.device),
                mask = th.from_numpy(np.ones_like(self.returns[batch_inds]).squeeze(1)).to(self.device)
            )
            # yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size
            



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
        if th.all(episode_starts == 0.0):

            # lstm states should be [1, 2, 64] or [1, 1, 64]

            features, future_feature = self.obs_split(features)
            batch_size = features.shape[0] // n_seq
            features = features.view(-1, 2, 6)
            features = features.permute(2, 0, 1)
            lstm_output, lstm_states = lstm(features, lstm_states)
            # print(f'lstm_output: {lstm_output.shape}')
            
            lstm_output = lstm_output[-1]
            # lstm_output = th.flatten(lstm_output, start_dim=0, end_dim=1)
            # print(f'lstm_output after flatten: {lstm_output.shape}')
            # print(f'lstm_states after flatten: {lstm_states[0].shape}')
            return lstm_output, lstm_states

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

    @property
    def get_env(self):
        return self.env
    
    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = RecurrentDictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else CustomRecurrentRolloutBuffer

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        # We assume that LSTM for the actor and the critic
        # have the same architecture
        lstm = self.policy.lstm_actor

        if not isinstance(self.policy, RecurrentActorCriticPolicy):
            raise ValueError("Policy must subclass RecurrentActorCriticPolicy")

        single_hidden_state_shape = (lstm.num_layers, self.n_envs, lstm.hidden_size)
        # hidden and cell states for actor and critic
        self._last_lstm_states = RNNStates(
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
        )

        hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
