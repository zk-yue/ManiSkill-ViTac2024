from typing import Optional

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from stable_baselines3.sac.policies import Actor, SACPolicy
from solutions.sb3_sac_policies import Actor, SACPolicy

# from solutions.actor_and_critics_sac import CustomActor
from solutions.feature_extractors import (CriticFeatureExtractor,
                                          FeatureExtractorForPointFlowEnv, CriticFeatureExtractorForLongOpenLock)
from stable_baselines3.common.policies import ContinuousCritic

import gymnasium as gym
from solutions.networks import PointNetFeatureExtractor
import torch
from torch import nn
from stable_baselines3.common.preprocessing import get_action_dim

class FeatureExtractorWithPointNetEncoder(BaseFeaturesExtractor):
    """
    this feature extractor can extract the latent feature from the marker flow, which is different from those in feature_extractors.py.
    """
    def __init__(
            self, 
            observation_space: gym.spaces.Dict, 
            features_dim=64,
            use_relative_motion=False,
            dim_add=0
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
                out_dim=int((features_dim-dim_add) / 2)
                )
        else:
            self.feature_extractor_net = PointNetFeatureExtractor(
                dim=pointnet_in_dim, 
                out_dim=int(features_dim / 2)
                )

    def forward(self, observations) -> torch.Tensor:       
        # ---------替换：FeatureExtractorForPointFlowEnv-------------
        original_obs = observations["marker_flow"]
        if original_obs.ndim == 4: # 如果只有四维，加入1维表示batch_num torch.Size([4, 2, 2, 128, 2])
            original_obs = torch.unsqueeze(original_obs, 0)
        # (batch_num, 2 (left_and_right), 2 (no-contact and contact), 128 (marker_num), 2 (u, v))
        feature_extractor_input = torch.cat([original_obs[:, :, 0, ...], original_obs[:, :, 1, ...]], dim=-1) # 将no-contact and contact的坐标合并在一起  torch.Size([4, 2, 128, 4])
        # (batch_num, 2 (left_and_right), 128 (marker_num), 4 (u, v, u', v'))
        # ---------替换：FeatureExtractorForPointFlowEnv-------------
        # ---------替换：PointNetActor中forward 中mlp_policy之前部分-------------
        batch_num = original_obs.shape[0]
        feature_extractor_input = torch.cat([feature_extractor_input[:, 0, ...], feature_extractor_input[:, 1, ...]], dim=0) # 将left_and_right torch.Size([8, 128, 4])
        # (batch_num *2 , 128 (marker_num), 4 (u, v, u', v'))
        # (batch_num * 2, 128, 4)
        # l_marker_pos = feature_extractor_input[:, 0, ...]
        # r_marker_pos = feature_extractor_input[:, 1, ...]
        # shape: (batch, num_points, 4)

        # with torch.inference_mode():
        # self.point_net_feature_extractor.eval()
        marker_flow_fea = self.feature_extractor_net(feature_extractor_input) # 每一张照片输出32维，两张照片输出64维 torch.Size([8, 32])
        # l_marker_flow_fea = self.feature_extractor_net(l_marker_pos)
        # r_marker_flow_fea = self.feature_extractor_net(r_marker_pos)  # (batch_num, pointnet_feature_dim)
        marker_flow_fea = torch.cat([marker_flow_fea[:batch_num], marker_flow_fea[batch_num:]], dim=-1) # 2 * pointnet_feature_dim torch.Size([4, 64])

        if self.use_relative_motion:
            marker_flow_fea = [marker_flow_fea, ]
            relative_motion = observations["relative_motion"]
            if relative_motion.ndim == 1:
                relative_motion = torch.unsqueeze(relative_motion, dim=0)
            # repeat_num = l_point_flow_fea.shape[-1] // 4
            # xz = xz.repeat(1, repeat_num)
            marker_flow_fea.append(relative_motion)
            marker_flow_fea = torch.cat(marker_flow_fea, dim=-1)

        return marker_flow_fea


# class CustomActor(Actor):
#     def __init__(
#         self,
#         observation_space: gym.spaces.Space, # Dict('gt_offset': Box(-3.4028235e+38, 3.4028235e+38, (3,), float32), 'marker_flow': Box(-3.4028235e+38, 3.4028235e+38, (2, 2, 128, 2), float32))
#         action_space: gym.spaces.Box, # Box(-1.0, 1.0, (3,), float32) Box(-1.0, 1.0, (3,), float32)
#         features_extractor: nn.Module,
#         # features_dim: int,
#         layernorm=True,
#         zero_init_output=False,
#         **kwargs,
#     ):
#         super().__init__(
#             observation_space,
#             action_space,
#             features_extractor=features_extractor,
#             **kwargs,
#         )

#         action_dim = get_action_dim(self.action_space) # 3

#         self.mlp_policy = nn.Sequential(
#             nn.Linear(64, 256),
#             nn.LayerNorm(256) if layernorm else nn.Identity(),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.LayerNorm(256) if layernorm else nn.Identity(),
#             nn.ReLU(),
#             nn.Linear(256, action_dim),
#             nn.Tanh(),
#         )

#         if zero_init_output:
#             last_linear = None
#             for m in self.mlp_policy.children():
#                 if isinstance(m, nn.Linear):
#                     last_linear = m
#             if last_linear is not None:
#                 nn.init.zeros_(last_linear.bias)
#                 last_linear.weight.data.copy_(0.01 * last_linear.weight.data)

#         # self.mu = nn.Sequential(*actor_net)

#     def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
#         point_flow_fea = self.extract_features(obs,self.features_extractor) # 4*64
#         pred = self.mlp_policy(point_flow_fea) # 4*3 
#         return pred
'''原始的actor的forward
def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)
'''
    

class SACPolicyForPointFlowEnv(SACPolicy):
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
        super(SACPolicyForPointFlowEnv, self).__init__(*args, **kwargs)

    # 不用自定义的actor网络，修改actor网络的输入特征提取器
    # def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
    #     actor_kwargs = self._update_features_extractor(
    #         self.actor_kwargs, FeatureExtractorWithPointNetEncoder(self.observation_space,features_dim=self.pointnet_out_dim * 2)
    #     )
    #     return CustomActor(
    #         # features_dim=self.pointnet_out_dim * 2,
    #         layernorm=self.pointnet_layernorm,
    #         zero_init_output=self.zero_init_output,
    #         **actor_kwargs,
    #     ).to(self.device)
    
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, FeatureExtractorWithPointNetEncoder(self.observation_space,features_dim=self.pointnet_out_dim * 2)
            )
        return Actor(**actor_kwargs).to(self.device)

    # 使用默认critic网络，修改critic网络的输入特征提取器
    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, CriticFeatureExtractor(self.observation_space)
            ) # 提取observations["gt_offset"]
        return ContinuousCritic(**critic_kwargs).to(self.device)



class SACPolicyForLongOpenLockPointFlowEnv(SACPolicy):
    def __init__(
            self,
            *args,
            pointnet_in_dim,
            pointnet_out_dim,
            pointnet_batchnorm,
            pointnet_layernorm,
            zero_init_output,
            use_relative_motion: bool,
            **kwargs,
    ):
        self.pointnet_in_dim = pointnet_in_dim
        self.pointnet_out_dim = pointnet_out_dim
        self.pointnet_layernorm = pointnet_layernorm
        self.pointnet_batchnorm = pointnet_batchnorm
        self.use_relative_motion = use_relative_motion
        self.zero_init_output = zero_init_output
        super(SACPolicyForLongOpenLockPointFlowEnv, self).__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        if self.use_relative_motion==True:
            dim_add=3
        else:
            dim_add=0
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, 
            FeatureExtractorWithPointNetEncoder(
                self.observation_space,
                features_dim=self.pointnet_out_dim * 2+dim_add,
                use_relative_motion=self.use_relative_motion,
                dim_add=dim_add)
            )
        return Actor(**actor_kwargs).to(self.device)


    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, CriticFeatureExtractorForLongOpenLock(self.observation_space)
        )
        return ContinuousCritic(**critic_kwargs).to(self.device)

# class TD3PolicyForLongOpenLockPointFlowEnv(TD3Policy):
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
