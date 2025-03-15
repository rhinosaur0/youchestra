from typing import Optional, Union, Any
from copy import deepcopy
from functools import partial

from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy, MlpLstmPolicy, MultiInputLstmPolicy
from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer

from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.utils import zip_strict, explained_variance, get_schedule_fn, obs_as_tensor
from stable_baselines3.common.buffers import RolloutBuffer
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.common.recurrent.buffers import create_sequencers
from sb3_contrib.common.recurrent.type_aliases import RecurrentRolloutBufferSamples




import numpy as np
from torch import nn
import torch as th
from gymnasium import spaces


class CustomRecurrentRolloutBuffer(RecurrentRolloutBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slice_ticker = 0


    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env = None,
    ):
        

        self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(
            self.episode_starts[batch_inds], env_change[batch_inds], self.device
        )
        
        # print(self.episode_starts.shape)
        # print(self.observations.shape)
        batch_size = len(batch_inds)
        # Number of sequences
        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length
        # We retrieve the lstm hidden states that will allow
        # to properly initialize the LSTM at the beginning of each sequence

        lstm_states_pi = (
            # 1. (n_envs * n_steps, n_layers, dim) -> (batch_size, n_layers, dim)
            # 2. (batch_size, n_layers, dim)  -> (n_seq, n_layers, dim)
            # 3. (n_seq, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_vf = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_pi = (self.to_torch(lstm_states_pi[0]).contiguous(), self.to_torch(lstm_states_pi[1]).contiguous())
        lstm_states_vf = (self.to_torch(lstm_states_vf[0]).contiguous(), self.to_torch(lstm_states_vf[1]).contiguous())
        # print(self.pad(self.observations[batch_inds]).reshape((padded_batch_size, *self.obs_shape)).shape)
        # print(self.pad(self.actions[batch_inds]).reshape((padded_batch_size,) + self.actions.shape[1:]).shape)
        # print(self.pad_and_flatten(self.values[batch_inds]).shape)
        # print(self.pad_and_flatten(self.log_probs[batch_inds]).shape)
        # print(self.pad_and_flatten(self.advantages[batch_inds]).shape)
        # print(self.pad_and_flatten(self.returns[batch_inds]).shape)
        # print(lstm_states_pi[0].shape)
        # print(self.pad_and_flatten(self.episode_starts[batch_inds]).shape)
        # print(self.pad_and_flatten(np.ones_like(self.returns[batch_inds])).shape)

        return RecurrentRolloutBufferSamples(
            # (batch_size, obs_dim) -> (n_seq, max_length, obs_dim) -> (n_seq * max_length, obs_dim)
            observations=self.pad(self.observations[batch_inds]).reshape((padded_batch_size, *self.obs_shape)),
            actions=self.pad(self.actions[batch_inds]).reshape((padded_batch_size,) + self.actions.shape[1:]),
            old_values=self.pad_and_flatten(self.values[batch_inds]),
            old_log_prob=self.pad_and_flatten(self.log_probs[batch_inds]),
            advantages=self.pad_and_flatten(self.advantages[batch_inds]),
            returns=self.pad_and_flatten(self.returns[batch_inds]),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
            mask=self.pad_and_flatten(np.ones_like(self.returns[batch_inds])),
        )
    
    def get(self, batch_size: Optional[int] = None):
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
                observations = th.from_numpy(self.observations[batch_inds]),
                actions = th.from_numpy(self.actions[batch_inds]),
                old_values = th.from_numpy(self.values[batch_inds].squeeze(1)),
                old_log_prob = th.from_numpy(self.log_probs[batch_inds].squeeze(1)),
                advantages = th.from_numpy(self.advantages[batch_inds].squeeze(1)),
                returns = th.from_numpy(self.returns[batch_inds].squeeze(1)),
                lstm_states = RNNStates(
                    (th.from_numpy(self.hidden_states_pi[batch_inds]).permute(1, 0, 2), 
                    th.from_numpy(self.cell_states_pi[batch_inds]).permute(1, 0, 2)),
                    (th.from_numpy(self.hidden_states_vf[batch_inds]).permute(1, 0, 2),
                    th.from_numpy(self.cell_states_vf[batch_inds]).permute(1, 0, 2)),
                ),
                episode_starts = th.from_numpy(self.episode_starts[batch_inds].squeeze(1)),
                mask = th.from_numpy(np.ones_like(self.returns[batch_inds]).squeeze(1))
            )
            # yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size
            # yield RecurrentRolloutBufferSamples(
            #     observations=self.observations[batch_inds]
            # )
            



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
        print(hidden_state_buffer_shape)

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

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):

        assert isinstance(
            rollout_buffer, (RecurrentRolloutBuffer, RecurrentDictRolloutBuffer)
        ), f"{rollout_buffer} doesn't support recurrent policy"

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        lstm_states = deepcopy(self._last_lstm_states)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = th.tensor(self._last_episode_starts, dtype=th.float32, device=self.device)
                # print(lstm_states.vf)
                actions, values, log_probs, lstm_states = self.policy.forward(obs_tensor, lstm_states, episode_starts)

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done_ in enumerate(dones):
                if (
                    done_
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, idx : idx + 1, :].contiguous(),
                            lstm_states.vf[1][:, idx : idx + 1, :].contiguous(),
                        )
                        
                        # terminal_lstm_state = None
                        episode_starts = th.tensor([False], dtype=th.float32, device=self.device)
                        terminal_value = self.policy.predict_values(terminal_obs, terminal_lstm_state, episode_starts)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

        with th.no_grad():
            # Compute value for the last timestep
            episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), lstm_states.vf, episode_starts)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
    
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Convert mask from float to bool
                mask = rollout_data.mask > 1e-8


                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()[mask]).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                # Mask padded sequences
                value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])

                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob[mask])
                else:
                    entropy_loss = -th.mean(entropy[mask])

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


