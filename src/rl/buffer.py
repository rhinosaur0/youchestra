from typing import Optional
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.common.recurrent.type_aliases import RecurrentRolloutBufferSamples
import numpy as np
import torch as th


class CustomRecurrentRolloutBuffer(RecurrentRolloutBuffer):

    '''
    This is modified from the original SB3 code to handle the custom LSTM
    The original treats individual data as sequential data, outputting actions as if each data is a time-step
    However, the design of the accompanist has a fixed window_length of 7, which means that the data is not sequential
    but rather a fixed window of data. This function takes a random distribution of size batch_size from n_steps
    '''
    def add(self, *args, lstm_states: RNNStates, **kwargs):
        super().add(*args, lstm_states = lstm_states, **kwargs)

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
            