from typing import Optional

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from stable_baselines3.td3.policies import Actor, TD3Policy
from stable_baselines3.sac.policies import Actor, SACPolicy
from stable_baselines3.common.policies import ContinuousCritic

# from solutions.actor_and_critics import CustomCritic, PointNetActor, LongOpenLockPointNetActor
from solutions.feature_extractors import (CriticFeatureExtractor,
                                          FeatureExtractorForPointFlowEnv, CriticFeatureExtractorForLongOpenLock)

import gymnasium as gym
from solutions.networks import PointNetFeatureExtractor
import torch
from torch import nn

# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
#         self.fc = nn.Linear(hidden_size, output_size)
 
#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[-1, :, :])
#         return out
 

# class CNN(nn.Module):
#     """
#     CNN from DQN Nature paper:
#         Mnih, Volodymyr, et al.
#         "Human-level control through deep reinforcement learning."
#         Nature 518.7540 (2015): 529-533.

#     :param observation_space:
#     :param features_dim_cnn: Number of features extracted.
#         This corresponds to the number of unit for the last layer.
#     :param normalized_image: Whether to assume that the image is already normalized
#         or not (this disables dtype and bounds checks): when True, it only checks that
#         the space is a Box and has 3 dimensions.
#         Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
#     """
#     def __init__(self, n_input_channels = 3, features_dim_cnn: int = 512):
#         super(CNN, self).__init__()
#         # We assume CxHxW images (channels first)
#         self.cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         # Compute shape by doing one forward pass
#         with torch.no_grad():
#             n_flatten = 59904

#         self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim_cnn), nn.ReLU())

#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
#         return self.linear(self.cnn(observations))

class FeatureExtractorWithPointNetEncoder(BaseFeaturesExtractor):
    """
    this feature extractor can extract the latent feature from the marker flow, which is different from those in feature_extractors.py.
    """
    def __init__(
            self, 
            observation_space: gym.spaces.Dict, 
            # features_dim=64,
            features_dim=128,
            # features_dim=512,
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
                # out_dim=32
                )
        else:
            self.feature_extractor_net = PointNetFeatureExtractor(
                dim=pointnet_in_dim, 
                out_dim=int(features_dim / 2)
                # out_dim=32
                )
        
        # self.feature_extractor_net_diff = PointNetFeatureExtractor(
        #         dim=2, 
        #         out_dim=32
        #         )
        
        # self.cnn = CNN(n_input_channels=3, features_dim_cnn=256)
        # self.lstm=LSTM(
        #     input_size = features_dim, 
        #     hidden_size = 256, 
        #     num_layers = 2, 
        #     output_size = features_dim
        #     )

    def forward(self, observations) -> torch.Tensor:       
        with torch.set_grad_enabled(False):
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
        
        # diff
        # diff_flow = original_obs[:, :, 0, ...] - original_obs[:, :, 1, ...]
        # feature_extractor_diff_input = torch.cat([diff_flow[:, 0, ...], diff_flow[:, 1, ...]], dim=0) # 将left_and_right torch.Size([8, 128, 4])
        # marker_flow_fea_diff = self.feature_extractor_net_diff(feature_extractor_diff_input)
        # marker_flow_fea_diff = torch.cat([marker_flow_fea_diff[:batch_num], marker_flow_fea_diff[batch_num:]], dim=-1)
        # return torch.cat([marker_flow_fea, marker_flow_fea_diff], dim=1)


        # obs
        # obs_rgb = observations["rgb_images"] # torch.Size([4, 2, 240, 320, 3])
        # obs_rgb = obs_rgb.permute(0, 1, 4, 2, 3) #torch.Size([4, 2, 3, 240, 320])
        # if obs_rgb.ndim == 4:
        #     obs_rgb = torch.unsqueeze(obs_rgb, 0)
        # batch_num = obs_rgb.shape[0]
        # cnn_input = torch.cat([obs_rgb[:, 0, ...], obs_rgb[:, 1, ...]], dim=0) # torch.Size([8 , 3, 240, 320])
        # obs_rgb_fea = self.cnn(cnn_input)
        # obs_rgb_fea = torch.cat([obs_rgb_fea[:batch_num], obs_rgb_fea[batch_num:]], dim=-1) 
        # return obs_rgb_fea
    
        # return torch.cat([marker_flow_fea, obs_rgb_fea], dim=1)

        # # ---------LSTM---------
        # if marker_flow_fea.ndim == 2:
        #     marker_flow_fea = torch.unsqueeze(marker_flow_fea, 0)

        # marker_flow_fea = self.lstm(marker_flow_fea)
        # # ---------LSTM---------


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

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, FeatureExtractorWithPointNetEncoder(self.observation_space,features_dim=self.pointnet_out_dim * 2)
            # self.actor_kwargs, FeatureExtractorWithPointNetEncoder(self.observation_space,features_dim=128)
        )

        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, CriticFeatureExtractor(self.observation_space)
        )
        return ContinuousCritic(**critic_kwargs).to(self.device)

# class SACPolicyForLongOpenLockPointFlowEnv(TD3Policy):
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
#         super(SACPolicyForLongOpenLockPointFlowEnv, self).__init__(*args, **kwargs)

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
