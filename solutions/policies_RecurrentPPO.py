from typing import Optional

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# from stable_baselines3.td3.policies import Actor, TD3Policy
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy

from solutions.actor_and_critics import CustomCritic, PointNetActor, LongOpenLockPointNetActor
from solutions.feature_extractors import (CriticFeatureExtractor,
                                          FeatureExtractorForPointFlowEnv, CriticFeatureExtractorForLongOpenLock)


import gymnasium as gym
# from solutions.networks_RecurrentPPO import PointNetFeatureExtractor
from solutions.networks import PointNetFeatureExtractor
import torch
from torch import nn

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

    # def forward(self, observations) -> torch.Tensor:       
    #     with torch.set_grad_enabled(False):
    #         # ---------替换：FeatureExtractorForPointFlowEnv-------------
    #         original_obs = observations["marker_flow_seq"]
    #         if original_obs.ndim == 5: # 如果只有w五维，加入1维表示batch_num torch.Size([2, 4, 2, 2, 128, 2]) 
    #             original_obs = torch.unsqueeze(original_obs, 0)
    #         # (batch_num, seq_n, 2 (left_and_right), 2 (no-contact and contact), 128 (marker_num), 2 (u, v))
    #         feature_extractor_input = torch.cat([original_obs[:, :, :, 0, ...], original_obs[:, :, :, 1, ...]], dim=-1) # 将no-contact and contact的坐标合并在一起  torch.Size([4, 2, 128, 4])
    #         # (batch_num, seq_n, 2 (left_and_right), 128 (marker_num), 4 (u, v, u', v'))
    #     # ---------替换：FeatureExtractorForPointFlowEnv-------------
    #     # ---------替换：PointNetActor中forward 中mlp_policy之前部分-------------
    #     batch_num = original_obs.shape[0]
    #     feature_extractor_input = torch.cat([feature_extractor_input[:, :, 0, ...], feature_extractor_input[:, :, 1, ...]], dim=0) # 将left_and_right torch.Size([4, 4, 128, 4])
    #     # (batch_num *2 , seq_n, 128 (marker_num), 4 (u, v, u', v'))
    #     # (batch_num * 2, 128, 4)
    #     # l_marker_pos = feature_extractor_input[:, 0, ...]
    #     # r_marker_pos = feature_extractor_input[:, 1, ...]
    #     # shape: (batch, num_points, 4)

    #     # with torch.inference_mode():
    #     # self.point_net_feature_extractor.eval()
    #     marker_flow_fea = self.feature_extractor_net(feature_extractor_input) # 每一张照片输出32维，两张照片输出64维  torch.Size([4, 4, 32])
    #     # l_marker_flow_fea = self.feature_extractor_net(l_marker_pos)
    #     # r_marker_flow_fea = self.feature_extractor_net(r_marker_pos)  # (batch_num, pointnet_feature_dim)
    #     marker_flow_fea = torch.cat([marker_flow_fea[:batch_num, : , :], marker_flow_fea[batch_num:,: , :]], dim=-1) # 2 * pointnet_feature_dim torch.Size([4, 64])

    #     # if self.use_relative_motion:
    #     #     marker_flow_fea = [marker_flow_fea, ]
    #     #     relative_motion = observations["relative_motion"]
    #     #     if relative_motion.ndim == 1:
    #     #         relative_motion = torch.unsqueeze(relative_motion, dim=0)
    #     #     # repeat_num = l_point_flow_fea.shape[-1] // 4
    #     #     # xz = xz.repeat(1, repeat_num)
    #     #     marker_flow_fea.append(relative_motion)
    #     #     marker_flow_fea = torch.cat(marker_flow_fea, dim=-1)
    #     marker_flow_fea = torch.transpose(marker_flow_fea, 1, 2)
    #     return marker_flow_fea # torch.Size([2, 4, 64]) batch_num seq_n features_dim

class RecurrentPPOPolicyForPointFlowEnv(MlpLstmPolicy):
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
        policy_kwargs = dict(
            features_extractor_class=FeatureExtractorWithPointNetEncoder,
            features_extractor_kwargs=dict(
                features_dim=pointnet_out_dim * 2,
                use_relative_motion=False,
                dim_add=0,
            ),
        )

        super(RecurrentPPOPolicyForPointFlowEnv, self).__init__(*args, **policy_kwargs, **kwargs)

        # self.pi_features_extractor = FeatureExtractorWithPointNetEncoder(self.observation_space,features_dim=self.pointnet_out_dim * 2)
        # self.vf_features_extractor = CriticFeatureExtractor(self.observation_space)
        # self.features_dim = self.pointnet_out_dim * 2

