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
from stable_baselines3.common.utils import get_schedule_fn
from sb3_contrib.common.recurrent.type_aliases import RNNStates


from torch import nn
from torch.nn import functional as F
import torch as th
from gymnasium import spaces

from .buffer import CustomRecurrentRolloutBuffer
from .memory import retrieve_memory, store_memory


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
        memory_dim: int = 64,
        memory_discount_factor: float = 0.9,
        ref_context_dim: int = 32,
        training_mode: str = 'init', # 'init' means background training, 'live' means app feature training, 'test' means evaluation
        memory_dropout_prob: float = 0.5,
    ):
        self.lstm_features = lstm_features
        self.lstm_output_dim = lstm_hidden_size
        self.training = training_mode
        self.memory_dim = memory_dim
        self.memory_discount_factor = memory_discount_factor
        self.ref_context_dim = ref_context_dim
        self.memory_file = None
        self.memory_dropout_prob = memory_dropout_prob

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
        self.lstm_actor = nn.LSTM(
            1, # remove the future reference
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
                1,
                lstm_hidden_size,
                num_layers=n_lstm_layers,
                **self.lstm_kwargs,
            )
        self.ref_encoder = nn.Sequential(
            nn.Linear(7, self.ref_context_dim),
            nn.ReLU(),
        )
        self.memory_projector = nn.Linear(7, self.memory_dim)
        self.memory_gate = nn.Parameter(th.tensor([0.9, 0.09, 0.009], requires_grad=True))
        self.mem_ref_fusion_layer = nn.Sequential(
            nn.Linear(self.memory_dim + self.ref_context_dim, self.memory_dim),
            nn.ReLU()
        )
        self.cur_ref_fusion_layer = nn.Sequential(
            nn.Linear(lstm_hidden_size + self.ref_context_dim, lstm_hidden_size),
            nn.ReLU()
        )

        self.future_feature_proj = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        self.post_concat_layer = nn.Sequential(
            nn.Linear(lstm_hidden_size + self.memory_dim, lstm_hidden_size),
            nn.ReLU()
        )

        # self.post_concat_layer = nn.Sequential(
        #     nn.Linear(lstm_hidden_size + 16, lstm_hidden_size),
        #     nn.ReLU()
        # )
        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


    def split_end(self, obs):
        return obs[:, :-1], obs[:, -1]
    
    def split_begin(self, obs):
        return obs[:, 0], obs[:, 1:]
    
    def mem_cur_ref_split(self, obs):
        return obs[:, 0], obs[:, 1:7], obs[:, 7:]
    
    def set_training_mode(self, mode):
        if mode == True:
            self.training = 'init'
            super().set_training_mode(mode)
        elif mode == False:
            self.training = 'test'
            super().set_training_mode(mode)
        else:
            self.training = 'live'

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
        n_seq = lstm_states[0].shape[1]
        
        batch_size = features.shape[0]
        memory_indices, cur_features, ref_features = self.mem_cur_ref_split(features)

        ref_features = self.ref_encoder(ref_features) # ref features for both memory concatenation and 


        memoryraw = th.from_numpy(retrieve_memory(memory_indices, filename = self.memory_file)).to(self.device)

        m1 = self.memory_projector(memoryraw[:, 0, :].squeeze(dim = 1).float())
        m2 = self.memory_projector(memoryraw[:, 1, :].squeeze(dim = 1).float())
        m3 = self.memory_projector(memoryraw[:, 2, :].squeeze(dim = 1).float()) 

        weights = th.softmax(self.memory_gate, dim=0)
        mem_features = weights[0] * m1 + weights[1] * m2 + weights[2] * m3

        lstm_output, lstm_states = lstm(
            cur_features.permute(1, 0).unsqueeze(2), 
            (
                (1.0 - episode_starts).view(1, n_seq, 1) * lstm_states[0],
                (1.0 - episode_starts).view(1, n_seq, 1) * lstm_states[1],
            )
        )
        cur_features = lstm_output[-1]

        mem_ref_features = self.mem_ref_fusion_layer(th.cat([mem_features, ref_features], dim=1))
        cur_ref_features = self.cur_ref_fusion_layer(th.cat([cur_features, ref_features], dim=1))
        if self.training == 'init':
            mem_ref_features = F.dropout(mem_ref_features, p = self.memory_dropout_prob)
        final_hidden_features = self.post_concat_layer(th.cat([cur_ref_features, mem_ref_features], dim=1))

        return final_hidden_features, lstm_states
        # features, future_feature = self.split_end(features)


        # features = features.view(-1, 2, 6)
        # features = features.permute(2, 0, 1)
        # lstm_output, lstm_states = lstm(
        #     features, 
        #     (
        #         (1.0 - episode_starts).view(1, n_seq, 1) * lstm_states[0],
        #         (1.0 - episode_starts).view(1, n_seq, 1) * lstm_states[1],
        #     )
        # )
        # hidden_last = lstm_output[-1]
        # future_feature = self.future_feature_proj(future_feature.unsqueeze(1))

        # combined = th.cat([hidden_last, future_feature], dim=1)  
        # final_hidden = self.post_concat_layer(combined)

        # if self.training == 'live':
        #     store_memory(memory_indices, features)
            
        # return final_hidden, lstm_states
        



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