# pointnet在特征提取器里面

from typing import Optional

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from stable_baselines3.td3.policies import Actor, TD3Policy
from stable_baselines3.common.policies import ContinuousCritic
# from sb3_contrib.tqc.policies import Actor, CnnPolicy, Critic, MlpPolicy, MultiInputPolicy, TQCPolicy

# from solutions.actor_and_critics import CustomCritic, PointNetActor, LongOpenLockPointNetActor
from solutions.feature_extractors import (CriticFeatureExtractor,
                                          FeatureExtractorForPointFlowEnv, CriticFeatureExtractorForLongOpenLock)


import gymnasium as gym
from gymnasium import spaces
from solutions.networks_RecurrentPPO import PointNetFeatureExtractor
import torch
from torch import nn

from stable_baselines3.common.preprocessing import get_action_dim

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
 
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[-1, :, :])
        return out

class customActor(Actor):
    """
    Actor network (policy) for TD3.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        # net_arch: List[int],
        features_extractor: nn.Module,
        pointnet_out_dim: int,
        layernorm=True,
        zero_init_output=False,
        # activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            **kwargs,
        )

        # self.net_arch = net_arch
        self.pointnet_out_dim = pointnet_out_dim
        # self.activation_fn = activation_fn

        action_dim = get_action_dim(self.action_space)
        # actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=True)
        # # Deterministic action
        # self.mu = nn.Sequential(*actor_net)
        self.mlp_policy = nn.Sequential(
            nn.Linear(pointnet_out_dim * 2, 256),
            nn.LayerNorm(256) if layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256) if layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        if zero_init_output:
            last_linear = None
            for m in self.mlp_policy.children():
                if isinstance(m, nn.Linear):
                    last_linear = m
            if last_linear is not None:
                nn.init.zeros_(last_linear.bias)
                last_linear.weight.data.copy_(0.01 * last_linear.weight.data)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs, self.features_extractor)
        return self.mlp_policy(features)

class FeatureExtractorWithPointNetEncoder(BaseFeaturesExtractor):
    """
    this feature extractor can extract the latent feature from the marker flow, which is different from those in feature_extractors.py.
    """
    def __init__(
            self, 
            observation_space: gym.spaces.Dict, 
            features_dim=64,
            use_relative_motion=False,
            dim_add=0,
            batchnorm=False,
            ):  
        super(FeatureExtractorWithPointNetEncoder, self).__init__(
            observation_space, 
            features_dim=features_dim
            )
        self._features_dim = features_dim
        self.use_relative_motion = use_relative_motion
        pointnet_in_dim = observation_space["marker_flow"].shape[-1] * 2 # 4 observation_space["marker_flow"] (2, 2, 128, 2)
        
        if self.use_relative_motion==True:
            self.feature_extractor_net = PointNetFeatureExtractor(
                dim=pointnet_in_dim, 
                out_dim=int((features_dim-dim_add) / 2),
                batchnorm=batchnorm
                )
        else:
            self.feature_extractor_net = PointNetFeatureExtractor(
                dim=pointnet_in_dim, 
                out_dim=int(features_dim / 2),
                batchnorm=batchnorm
                )
        
        self.lstm=LSTM(
            input_size = features_dim, 
            hidden_size = 256, 
            num_layers = 2, 
            output_size = features_dim
            )

    def forward(self, observations) -> torch.Tensor:       
        with torch.set_grad_enabled(False):
            # ---------替换：FeatureExtractorForPointFlowEnv-------------
            original_obs = observations["marker_flow_seq"]
            if original_obs.ndim == 5: # 如果只有w五维，加入1维表示batch_num torch.Size([2, 4, 2, 2, 128, 2]) 
                original_obs = torch.unsqueeze(original_obs, 0)
            # (batch_num, seq_n, 2 (left_and_right), 2 (no-contact and contact), 128 (marker_num), 2 (u, v))
            feature_extractor_input = torch.cat([original_obs[:, :, :, 0, ...], original_obs[:, :, :, 1, ...]], dim=-1) # 将no-contact and contact的坐标合并在一起  torch.Size([4, 2, 128, 4])
            # (batch_num, seq_n, 2 (left_and_right), 128 (marker_num), 4 (u, v, u', v'))
        # ---------替换：FeatureExtractorForPointFlowEnv-------------
        # ---------替换：PointNetActor中forward 中mlp_policy之前部分-------------
        batch_num = original_obs.shape[0]
        feature_extractor_input = torch.cat([feature_extractor_input[:, :, 0, ...], feature_extractor_input[:, :, 1, ...]], dim=0) # 将left_and_right torch.Size([4, 4, 128, 4])
        # (batch_num *2 , seq_n, 128 (marker_num), 4 (u, v, u', v'))
        # (batch_num * 2, 128, 4)
        # l_marker_pos = feature_extractor_input[:, 0, ...]
        # r_marker_pos = feature_extractor_input[:, 1, ...]
        # shape: (batch, num_points, 4)

        # with torch.inference_mode():
        # self.point_net_feature_extractor.eval()
        marker_flow_fea = self.feature_extractor_net(feature_extractor_input) # 每一张照片输出32维，两张照片输出64维  torch.Size([4, 4, 32])
        # l_marker_flow_fea = self.feature_extractor_net(l_marker_pos)
        # r_marker_flow_fea = self.feature_extractor_net(r_marker_pos)  # (batch_num, pointnet_feature_dim)
        marker_flow_fea = torch.cat([marker_flow_fea[:batch_num, : , :], marker_flow_fea[batch_num:,: , :]], dim=-1) # 2 * pointnet_feature_dim torch.Size([4, 64])

        # if self.use_relative_motion:
        #     marker_flow_fea = [marker_flow_fea, ]
        #     relative_motion = observations["relative_motion"]
        #     if relative_motion.ndim == 1:
        #         relative_motion = torch.unsqueeze(relative_motion, dim=0)
        #     # repeat_num = l_point_flow_fea.shape[-1] // 4
        #     # xz = xz.repeat(1, repeat_num)
        #     marker_flow_fea.append(relative_motion)
        #     marker_flow_fea = torch.cat(marker_flow_fea, dim=-1)
        marker_flow_fea = torch.transpose(marker_flow_fea, 0, 1)

        # ---------LSTM---------
        # if marker_flow_fea.ndim == 2:
        #     marker_flow_fea = torch.unsqueeze(marker_flow_fea, 0)

        marker_flow_fea = self.lstm(marker_flow_fea)
        # ---------LSTM---------

        return marker_flow_fea # torch.Size([2, 4, 64]) batch_num seq_n features_dim

class TD3PolicyForPointFlowEnv(TD3Policy):
    def __init__(
            self,
            *args,
            pointnet_in_dim,
            pointnet_out_dim,
            pointnet_batchnorm,
            pointnet_layernorm,
            zero_init_output,
            **kwargs,
    ):
        self.pointnet_in_dim = pointnet_in_dim
        self.pointnet_out_dim = pointnet_out_dim
        self.pointnet_layernorm = pointnet_layernorm
        self.pointnet_batchnorm = pointnet_batchnorm
        self.zero_init_output = zero_init_output
        super(TD3PolicyForPointFlowEnv, self).__init__(*args, **kwargs)

    # def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
    #     actor_kwargs = self._update_features_extractor(
    #         self.actor_kwargs, FeatureExtractorWithPointNetEncoder(self.observation_space,features_dim=self.pointnet_out_dim * 2)
    #         )
    #     return Actor(**actor_kwargs).to(self.device)

    # def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Critic:
    #     critic_kwargs = self._update_features_extractor(
    #         self.critic_kwargs, CriticFeatureExtractor(self.observation_space)
    #         )
    #     return Critic(**critic_kwargs).to(self.device)
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, FeatureExtractorWithPointNetEncoder(self.observation_space,features_dim=self.pointnet_out_dim * 2)
            )
        return customActor(
            pointnet_out_dim=self.pointnet_out_dim,
            layernorm=self.pointnet_layernorm,
            zero_init_output=self.zero_init_output,
            **actor_kwargs,
        ).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, CriticFeatureExtractor(self.observation_space)
            )
        return ContinuousCritic(**critic_kwargs).to(self.device)

# class TQCPolicyForLongOpenLockPointFlowEnv(TD3Policy):
#     def __init__(
#             self,
#             *args,
#             pointnet_in_dim,
#             pointnet_out_dim,
#             pointnet_batchnorm,
#             pointnet_layernorm,
#             zero_init_output,
#             use_relative_motion: bool,
#             **kwargs,
#     ):
#         self.pointnet_in_dim = pointnet_in_dim
#         self.pointnet_out_dim = pointnet_out_dim
#         self.pointnet_layernorm = pointnet_layernorm
#         self.pointnet_batchnorm = pointnet_batchnorm
#         self.use_relative_motion = use_relative_motion
#         self.zero_init_output = zero_init_output
#         super(TD3PolicyForLongOpenLockPointFlowEnv, self).__init__(*args, **kwargs)

#     def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
#         actor_kwargs = self._update_features_extractor(
#             self.actor_kwargs,
#         )
#         return LongOpenLockPointNetActor(
#             pointnet_in_dim=self.pointnet_in_dim,
#             pointnet_out_dim=self.pointnet_out_dim,
#             batchnorm=self.pointnet_batchnorm,
#             layernorm=self.pointnet_layernorm,
#             zero_init_output=self.zero_init_output,
#             use_relative_motion=self.use_relative_motion,
#             **actor_kwargs,
#         ).to(self.device)

#     def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomCritic:
#         critic_kwargs = self._update_features_extractor(
#             self.critic_kwargs, CriticFeatureExtractorForLongOpenLock(self.observation_space)
#         )
#         return CustomCritic(**critic_kwargs).to(self.device)
