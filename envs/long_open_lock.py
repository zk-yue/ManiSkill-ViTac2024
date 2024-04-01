import copy
import json
import os
import math
import sys
from collections import OrderedDict

from matplotlib import pyplot as plt
from path import Path
from sapienipc.ipc_utils.user_utils import ipc_update_render_all

script_path = os.path.dirname(os.path.realpath(__file__))
repo_path = os.path.join(script_path, "..")
sys.path.append(script_path)
sys.path.append(repo_path)

from envs.common_params import CommonParams
import time
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import sapien
import warp as wp
from gymnasium import spaces
from sapien.utils.viewer import Viewer
from sapienipc.ipc_system import IPCSystem, IPCSystemConfig

from envs.tactile_sensor_sapienipc import (TactileSensorSapienIPC,
                                           VisionTactileSensorSapienIPC)
from utils.common import Params, randomize_params, suppress_stdout_stderr
from utils.gym_env_utils import convert_observation_to_space
from utils.sapienipc_utils import build_sapien_entity_ABD
import cv2
wp.init()
wp_device = wp.get_preferred_device()

GUI = False


class LongOpenLockParams(CommonParams):
    def __init__(self,
                 key_lock_path_file: str = "",
                 key_friction: float = 1.0,
                 lock_friction: float = 1.0,
                 indentation_depth: float = 0.5,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.key_lock_path_file = key_lock_path_file
        self.indentation_depth = indentation_depth # 刻痕深度
        self.key_friction = key_friction # 摩擦系数
        self.lock_friction = lock_friction


class LongOpenLockSimEnv(gym.Env):
    def __init__(
            self,
            max_action: np.ndarray, #   max_action: [4.0, 2.0, 2.0]
            step_penalty: float, # 1
            final_reward: float, # 10
            key_x_max_offset: float = 10.0,
            key_y_max_offset: float = 0.0,
            key_z_max_offset: float = 0.0,
            max_steps: int = 10,
            sensor_offset_x_range_len: float = 0.0,
            senosr_offset_z_range_len: float = 0.0,
            params=None,
            params_upper_bound=None,
            device: str = "cuda:0",
            no_render: bool = False,
            **kwargs
    ):
        super(LongOpenLockSimEnv, self).__init__()

        self.no_render = no_render
        self.index = None
        self.step_penalty = step_penalty # 惩罚
        self.final_reward = final_reward # 奖励
        self.max_steps = max_steps
        self.max_action = np.array(max_action)
        assert self.max_action.shape == (3,)

        self.key_x_max_offset = key_x_max_offset
        self.key_y_max_offset = key_y_max_offset
        self.key_z_max_offset = key_z_max_offset
        self.sensor_offset_x_range_len = sensor_offset_x_range_len
        self.sensor_offset_z_range_len = senosr_offset_z_range_len

        self.current_episode_elapsed_steps = 0 # 当前episode已经走过的step
        self.current_episode_over = False
        self.sensor_grasp_center_init = np.array([0, 0, 0])
        self.sensor_grasp_center_current = self.sensor_grasp_center_init


        if not params: # 如果为空
            self.params_lb = LongOpenLockParams()
        else:
            self.params_lb = copy.deepcopy(params)
        if not params_upper_bound: #none
            self.params_ub = copy.deepcopy(self.params_lb)
        else:
            self.params_ub = copy.deepcopy(params_upper_bound)
        self.params: LongOpenLockParams = randomize_params(self.params_lb, self.params_ub)

        # key_lock_path_file="configs/key_and_lock/key_lock.txt"
        key_lock_path_file = Path(repo_path) / self.params.key_lock_path_file  #  / 运算符来连接路径。将这两个路径连接起来，得到键锁文件的绝对路径。
        self.key_lock_path_list = []
        with open(key_lock_path_file, "r") as f:
            for l in f.readlines(): # 使用 readlines 方法读取文件的所有行，返回一个包含每一行内容的列表。
                self.key_lock_path_list.append([ss.strip() for ss in l.strip().split(",")])
        # [['key_lock/key1.STL', 'key_lock/lock_core_sim1.STL'], ['key_lock/key2.STL', 'key_lock/lock_core_sim2.STL'], ['key_lock/key3.STL', 'key_lock/lock_core_sim3.STL'], ['key_lock/key4.STL', 'key_lock/lock_core_sim4.STL']]
        # 锁和锁芯
        # 使用 strip 方法移除行首和行尾的空白字符。
        # 使用 split 方法以逗号为分隔符，将行内容分割成多个子字符串，得到一个列表。
        # 对列表中的每个子字符串 ss，使用 strip 方法移除其首尾的空白字符。
        # 将处理后的列表添加到 self.key_lock_path_list 中。

        self.init_left_surface_pts = None
        self.init_right_surface_pts = None

        self.viewer = None
        if not no_render:
            self.scene = sapien.Scene()
            self.scene.set_ambient_light([0.5, 0.5, 0.5]) #  环境光
            self.scene.add_directional_light([0, -1, -1], [0.5, 0.5, 0.5], True)
        else:
            self.scene = sapien.Scene(systems=[])

        # add a camera to indicate shader
        if not no_render:
            cam_entity = sapien.Entity()
            cam = sapien.render.RenderCameraComponent(512, 512) # 像素512 512
            cam_entity.add_component(cam)
            cam_entity.name = "camera"
            self.scene.add_entity(cam_entity)

        ######## Create system ######## #
        # 处理刚体或软体物理行为的IPC（Incremental Potential Contact 或 类似技术）系统
        ipc_system_config = IPCSystemConfig()
        # memory config 内存配置
        ipc_system_config.max_scenes = 1
        ipc_system_config.max_surface_primitives_per_scene = 1 << 11
        ipc_system_config.max_blocks = 1000000
        # scene config 场景配置
        ipc_system_config.time_step = self.params.sim_time_step # 仿真的时间步长 0.05
        ipc_system_config.gravity = wp.vec3(0, 0, 0) # 重力向量
        ipc_system_config.d_hat = self.params.sim_d_hat  # 2e-4
        ipc_system_config.eps_d = self.params.sim_eps_d  # 1e-3
        ipc_system_config.eps_v = self.params.sim_eps_v  # 1e-3
        ipc_system_config.v_max = 1e-1
        ipc_system_config.kappa = self.params.sim_kappa  # 1e3 1000 控制仿真稳定性的 kappa 系列参数
        ipc_system_config.kappa_affine = self.params.sim_kappa_affine # 100000
        ipc_system_config.kappa_con = self.params.sim_kappa_con
        ipc_system_config.ccd_slackness = self.params.ccd_slackness # 0.7# 设置连续碰撞检测（CCD）的相关参数 
        ipc_system_config.ccd_thickness = self.params.ccd_thickness # 0.0
        ipc_system_config.ccd_tet_inversion_thres = self.params.ccd_tet_inversion_thres # 0.0
        ipc_system_config.ee_classify_thres = self.params.ee_classify_thres # 0.001
        ipc_system_config.ee_mollifier_thres = self.params.ee_mollifier_thres # 0.001
        ipc_system_config.allow_self_collision = bool(self.params.allow_self_collision) # 0.0
        # ipc_system_config.allow_self_collision = False

        # solver config 求解器配置
        ipc_system_config.newton_max_iters = int(self.params.sim_solver_newton_max_iters)  # key param
        ipc_system_config.cg_max_iters = int(self.params.sim_solver_cg_max_iters)
        ipc_system_config.line_search_max_iters = int(self.params.line_search_max_iters)
        ipc_system_config.ccd_max_iters = int(self.params.ccd_max_iters)
        ipc_system_config.precondition = "jacobi"
        ipc_system_config.cg_error_tolerance = self.params.sim_solver_cg_error_tolerance
        ipc_system_config.cg_error_frequency = int(self.params.sim_solver_cg_error_frequency)

        # set device 设备设置
        device = wp.get_device(device)
        ipc_system_config.device = wp.get_device(device)

        # 使用前面配置的 ipc_system_config 初始化 IPCSystem，然后将这个系统添加到场景中。
        self.ipc_system = IPCSystem(ipc_system_config)
        self.scene.add_system(self.ipc_system)

        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32) # low=-1 和 high=1 表示这个空间的每个维度的值都在 -1 到 1 之间。  shape=(3,) 表示这个空间是一个三维空间，即每个动作都是一个三维的向量。
        self.default_observation, _ = self.reset()
        self.observation_space = convert_observation_to_space(self.default_observation)

        # build scene, system

    def reset(self, offset=None, seed=None):

        if self.viewer:
            self.viewer.close()
            self.viewer = None
        self.params = randomize_params(self.params_lb, self.params_ub) # 该函数接受两个 Params 类型的参数：lower_bound 和 upper_bound，并返回一个新的 Params 对象，其属性值在 lower_bound 和 upper_bound 之间随机选择。
        self.current_episode_elapsed_steps = 0
        self.current_episode_over = False

        self.initialize(key_offset=offset)
        self.init_left_surface_pts = self.no_contact_surface_mesh[0] # [52,3]
        self.init_right_surface_pts = self.no_contact_surface_mesh[1]
        self.error_evaluation_list = []
        info = self.get_info()
        self.error_evaluation_list.append(self.evaluate_error(info))

        return self.get_obs(info), {}

    def evaluate_error(self, info, error_scale=4):
        error_sum = 0
        key1_pts_center = info["key1_pts"].mean(0) * 1000
        key2_pts_center = info["key2_pts"].mean(0) * 1000
        key1_pts_max = info["key1_pts"].max(0) * 1000
        key2_pts_max = info["key2_pts"].max(0) * 1000
        lock1_pts_center = info["lock1_pts"].mean(0) * 1000
        lock2_pts_center = info["lock2_pts"].mean(0) * 1000

        error_sum += np.abs(key1_pts_center[0] - lock1_pts_center[0])  # x direction
        error_sum += np.abs(key2_pts_center[0] - lock2_pts_center[0])
        # print(f"reward start: {reward}")
        # z_offset
        if self.index == 0 or self.index == 2:
            if key1_pts_max[0] < 46 and key2_pts_max[0] < 46:
                # if key is inside the lock, then encourage it to fit in to the holes
                error_sum += np.abs(37 - key1_pts_center[2])  # must be constrained in both directions
                error_sum += np.abs(37 - key2_pts_center[2])  # otherwise the policy would keep lifting the key
                # and smooth the error to avoid sudden change
            else:
                # else, align it with the hole
                error_sum += np.abs(key1_pts_center[2] - 30) + 7
                error_sum += np.abs(key2_pts_center[2] - 30) + 7
                pass
        if self.index == 1:
            if key1_pts_max[0] < 52 and key2_pts_max[0] < 52:
                # if key is inside the lock, then encourage it to fit in to the holes
                error_sum += np.abs(37 - key1_pts_center[2])  # must be constrained in both directions
                error_sum += np.abs(37 - key2_pts_center[2])  # otherwise the policy would keep lifting the key
                # and smooth the error to avoid sudden change
            else:
                # else, align it with the hole
                error_sum += np.abs(key1_pts_center[2] - 30) + 7
                error_sum += np.abs(key2_pts_center[2] - 30) + 7
                pass
        if self.index == 3:
            if key1_pts_max[0] < 62 and key2_pts_max[0] < 62:
                # if key is inside the lock, then encourage it to fit in to the holes
                error_sum += np.abs(37 - key1_pts_center[2])  # must be constrained in both directions
                error_sum += np.abs(37 - key2_pts_center[2])  # otherwise the policy would keep lifting the key
                # and smooth the error to avoid sudden change
            else:
                # else, align it with the hole
                error_sum += np.abs(key1_pts_center[2] - 30) + 7
                error_sum += np.abs(key2_pts_center[2] - 30) + 7
                pass

        # y_offset
        error_sum += np.abs(key1_pts_center[1])
        error_sum += np.abs(key2_pts_center[1])
        # error_sum = np.sqrt(error_sum)
        error_sum *= error_scale
        return error_sum

    def seed(self, seed=None):
        if seed is None:
            seed = (int(time.time() * 1000) % 10000 * os.getpid()) % 2 ** 30
        np.random.seed(seed)

    def initialize(self, key_offset=None):
        
        # 清除场景中所有非相机实体，为新的仿真周期做准备。
        for e in self.scene.entities:
            if "camera" not in e.name:
                e.remove_from_scene()
        self.ipc_system.rebuild() # 重建IPC（增量潜在接触）系统，以应对场景中实体的变化。

        self.index = np.random.randint(len(self.key_lock_path_list)) # 随机选择一个锁和钥匙
        key_path, lock_path = self.key_lock_path_list[self.index]
        asset_dir = Path(repo_path) / "assets"
        key_path = asset_dir / key_path
        lock_path = asset_dir / lock_path

        if key_offset is None: # 计算钥匙的位置偏移。
            if self.index == 0:
                x_offset = np.random.rand() * self.key_x_max_offset + 46 # 0-1随机取
            elif self.index == 1:
                x_offset = np.random.rand() * self.key_x_max_offset + 52
            elif self.index == 2:
                x_offset = np.random.rand() * self.key_x_max_offset + 46
            elif self.index == 3:
                x_offset = np.random.rand() * self.key_x_max_offset + 62

            y_offset = (np.random.rand() * 2 - 1) * self.key_y_max_offset
            z_offset = (np.random.rand() * 2 - 1) * self.key_z_max_offset
            key_offset = (x_offset, y_offset, z_offset)
            print("index=", self.index, "keyoffset=", key_offset, )
        else:
            x_offset, y_offset, z_offset = tuple(key_offset)
            x_offset = np.clip(x_offset, 0, self.key_x_max_offset) # 限制在0-key_x_max_offset 间，用于截取数组中小于或者大于某值的部分，并使得被截取部分等于固定值。
            y_offset = np.clip(y_offset, -self.key_y_max_offset, self.key_y_max_offset)
            z_offset = np.clip(z_offset, -self.key_z_max_offset, self.key_z_max_offset)
            if self.index == 0:
                x_offset += 46
            elif self.index == 1:
                x_offset += 52
            elif self.index == 2:
                x_offset += 46
            elif self.index == 3:
                x_offset += 62
            key_offset = (x_offset, y_offset, z_offset)
            print("index=", self.index, "keyoffset=", key_offset, )

        key_offset = [value / 1000 for value in key_offset]

#  ------------------------------导入模型------------------------------------
        with suppress_stdout_stderr(): # 上下文管理器 抑制所有打印
            self.key_entity, key_abd = build_sapien_entity_ABD(key_path, "cuda:0", density=500.0,
                                                                                color=[1.0, 0.0, 0.0, 0.9],
                                                                                friction=self.params.key_friction,
                                                               no_render=self.no_render)
        self.key_abd = key_abd
        self.key_entity.set_pose(sapien.Pose(p=key_offset, q=[0.7071068, 0, 0, 0])) # p 是一个表示位置的三维向量，q 是一个表示方向的四元数。
        self.scene.add_entity(self.key_entity)

        with suppress_stdout_stderr():
            self.lock_entity, lock_abd = build_sapien_entity_ABD(lock_path, "cuda:0", density=500.0,
                                                                              color=[0.0, 0.0, 1.0, 0.6],
                                                                              friction=self.params.lock_friction,
                                                                 no_render=self.no_render)
        self.hold_abd = lock_abd
        self.scene.add_entity(self.lock_entity)
# ---------------------------------------------------------------------------

        sensor_x = np.random.rand() * self.sensor_offset_x_range_len # 2
        sensor_x = sensor_x * np.random.choice([-1, 1])
        sensor_x /= 1e3  # mm -> m
        sensor_z = np.random.rand() * self.sensor_offset_z_range_len # 2
        sensor_z = sensor_z * np.random.choice([-1, 1])
        sensor_z /= 1e3  # mm -> m
        # 计算触觉传感器的位置和方向，基于钥匙的位置和随机生成的偏移量。
        if self.index == 0 or self.index == 2:
            init_pos_l = np.array([
                key_offset[0] + 0.07 + sensor_x,
                key_offset[1] - (6 * 1e-3 / 2 + 0.002 + 0.0005),
                key_offset[2] + 0.016 + sensor_z
            ])
            init_rot_l = np.array([0.7071068, -0.7071068, 0, 0])

            init_pos_r = np.array([
                key_offset[0] + 0.07 + sensor_x,
                key_offset[1] + 6 * 1e-3 / 2 + 0.002 + 0.0005,
                key_offset[2] + 0.016 + sensor_z
            ])
            init_rot_r = np.array([0.7071068, 0.7071068, 0, 0])

        if self.index == 1:
            init_pos_l = np.array([
                key_offset[0] + 0.075 + sensor_x,
                key_offset[1] - (6 * 1e-3 / 2 + 0.002 + 0.0005),
                key_offset[2] + 0.016 + sensor_z
            ])
            init_rot_l = np.array([0.7071068, -0.7071068, 0, 0])

            init_pos_r = np.array([
                key_offset[0] + 0.075 + sensor_x,
                key_offset[1] + 6 * 1e-3 / 2 + 0.002 + 0.0005,
                key_offset[2] + 0.016 + sensor_z
            ])
            init_rot_r = np.array([0.7071068, 0.7071068, 0, 0])

        if self.index == 3:
            init_pos_l = np.array([
                key_offset[0] + 0.08 + sensor_x,
                key_offset[1] - (6 * 1e-3 / 2 + 0.002 + 0.0005),
                key_offset[2] + 0.016 + sensor_z
            ])
            init_rot_l = np.array([0.7071068, -0.7071068, 0, 0])

            init_pos_r = np.array([
                key_offset[0] + 0.08 + sensor_x,
                key_offset[1] + 6 * 1e-3 / 2 + 0.002 + 0.0005,
                key_offset[2] + 0.016 + sensor_z
            ])
            init_rot_r = np.array([0.7071068, 0.7071068, 0, 0])
        # 触觉传感器抓握中心的初始位置。
        self.sensor_grasp_center_init = np.array([
            key_offset[0] + 0.0175 + 0.032 + sensor_x,
            key_offset[1],
            key_offset[2] + 0.016 + sensor_z
        ])
        self.sensor_grasp_center_current = self.sensor_grasp_center_init.copy()

        with suppress_stdout_stderr():
            self.add_tactile_sensors(init_pos_l, init_rot_l, init_pos_r, init_rot_r)

        if GUI:
            self.viewer = Viewer()
            self.viewer.set_scene(self.scene)
            self.viewer.set_camera_pose(
                sapien.Pose([-0.0877654, 0.0921954, 0.186787], [0.846142, 0.151231, 0.32333, -0.395766]))
            pause = False
            while pause:
                if self.viewer.window.key_down("c"):
                    pause = False
                self.scene.update_render()
                ipc_update_render_all(self.scene)
                self.viewer.render()

        grasp_step = max(round((0.5 + self.params.indentation_depth) / 1000 / 2e-3 / self.params.sim_time_step), 1) # 15
        grasp_speed = (0.5 + self.params.indentation_depth) / 1000 / grasp_step / self.params.sim_time_step

        for grasp_step_counter in range(grasp_step):
            self.tactile_sensor_1.set_active_v([0, grasp_speed, 0])
            self.tactile_sensor_2.set_active_v([0, -grasp_speed, 0])
            with suppress_stdout_stderr():
                self.hold_abd.set_kinematic_target(
                    np.concatenate([np.eye(3), np.zeros((1, 3))], axis=0))  # hole stays static
                self.ipc_system.step()

            self.tactile_sensor_1.step()
            self.tactile_sensor_2.step()
            if GUI:
                self.scene.update_render()
                ipc_update_render_all(self.scene)
                self.viewer.render()

        if isinstance(self.tactile_sensor_1, VisionTactileSensorSapienIPC):
            if isinstance(self.tactile_sensor_2, VisionTactileSensorSapienIPC):
                self.tactile_sensor_1.set_reference_surface_vertices_camera()
                self.tactile_sensor_2.set_reference_surface_vertices_camera()
        self.no_contact_surface_mesh = copy.deepcopy(self._get_sensor_surface_vertices())

    def add_tactile_sensors(self, init_pos_l, init_rot_l, init_pos_r, init_rot_r):

        self.tactile_sensor_1 = TactileSensorSapienIPC(
            scene=self.scene,
            ipc_system=self.ipc_system,
            meta_file=self.params.tac_sensor_meta_file,
            init_pos=init_pos_l,
            init_rot=init_rot_l,
            elastic_modulus=self.params.tac_elastic_modulus_l,
            poisson_ratio=self.params.tac_poisson_ratio_l,
            density=self.params.tac_density_l,
            friction=self.params.tac_friction,
            name="tactile_sensor_1",
            no_render=self.no_render,
        )

        self.tactile_sensor_2 = TactileSensorSapienIPC(
            scene=self.scene,
            ipc_system=self.ipc_system,
            meta_file=self.params.tac_sensor_meta_file,
            init_pos=init_pos_r,
            init_rot=init_rot_r,
            elastic_modulus=self.params.tac_elastic_modulus_r,
            poisson_ratio=self.params.tac_poisson_ratio_r,
            density=self.params.tac_density_r,
            friction=self.params.tac_friction,
            name="tactile_sensor_2",
            no_render=self.no_render,
        )

    def step(self, action):

        self.current_episode_elapsed_steps += 1
        action = np.array(action).flatten() * self.max_action
        action = action / 1000
        self._sim_step(action)

        info = self.get_info()
        obs = self.get_obs(info=info)
        reward = self.get_reward(info=info)
        terminated = self.get_terminated(info=info)
        truncated = self.get_truncated(info=info)

        return obs, reward, terminated, truncated, info

    def get_obs(self, info):
        observation_left_surface_pts, observation_right_surface_pts = self._get_sensor_surface_vertices()
        obs_dict = {
            "surface_pts": np.stack(
                [
                    np.stack([self.init_left_surface_pts, observation_left_surface_pts]),
                    np.stack([self.init_right_surface_pts, observation_right_surface_pts]),
                ]
            ).astype(np.float32),
        }

        # extra observation for critics
        extra_dict = {
            "key1_pts": info["key1_pts"],
            "key2_pts": info["key2_pts"],
            "key_side_pts": info["key_side_pts"],
            "lock1_pts": info["lock1_pts"],
            "lock2_pts": info["lock2_pts"],
            "lock_side_pts": info["lock_side_pts"],
            "relative_motion": info["relative_motion"].astype(np.float32),
        }
        obs_dict.update(extra_dict) # 把extra_dict的内容加到obs_dict中

        return obs_dict

    def get_info(self):
        info = {"steps": self.current_episode_elapsed_steps}

        key_pts = self.key_abd.get_positions().cpu().numpy().copy()
        lock_pts = self.hold_abd.get_positions().cpu().numpy().copy()
        if self.index == 0:
            key1_idx = np.array([16, 17, 18, 19]) # large # 四个点
            key2_idx = np.array([24, 25, 26, 27])
            key_side_index = np.array([1, 3, 30, 31])
            lock1_idx = np.array([2, 3, 6, 7])
            lock2_idx = np.array([4, 5, 30, 31])
            lock_side_index = np.array([10, 11, 9, 13])
            self.key1_pts = key_pts[key1_idx]
            self.key2_pts = key_pts[key2_idx]
            self.key_side_pts = key_pts[key_side_index]
            self.lock1_pts = lock_pts[lock1_idx]
            self.lock2_pts = lock_pts[lock2_idx]
            self.lock_side_pts = lock_pts[lock_side_index]
        elif self.index == 1:
            key1_idx = np.array([20, 21, 22, 23])
            key2_idx = np.array([28, 29, 30, 31])
            key_side_index = np.array([0, 2, 4, 5])
            lock1_idx = np.array([0, 1, 6, 7])
            lock2_idx = np.array([30, 31, 2, 3])
            lock_side_index = np.array([8, 9, 11, 13])
            self.key1_pts = key_pts[key1_idx]
            self.key2_pts = key_pts[key2_idx]
            self.key_side_pts = key_pts[key_side_index]
            self.lock1_pts = lock_pts[lock1_idx]
            self.lock2_pts = lock_pts[lock2_idx]
            self.lock_side_pts = lock_pts[lock_side_index]
        elif self.index == 2:
            key1_idx = np.array([4, 5, 6, 7])
            key2_idx = np.array([12, 13, 14, 15])
            key_side_index = np.array([18, 19, 20, 21])
            lock1_idx = np.array([6, 7, 2, 3])
            lock2_idx = np.array([4, 5, 30, 31])
            lock_side_index = np.array([10, 9, 11, 13])
            self.key1_pts = key_pts[key1_idx]
            self.key2_pts = key_pts[key2_idx]
            self.key_side_pts = key_pts[key_side_index]
            self.lock1_pts = lock_pts[lock1_idx]
            self.lock2_pts = lock_pts[lock2_idx]
            self.lock_side_pts = lock_pts[lock_side_index]
        elif self.index == 3:
            key1_idx = np.array([8, 9, 10, 11])
            key2_idx = np.array([16, 17, 18, 19])
            key_side_index = np.array([30, 31, 32, 33])
            lock1_idx = np.array([2, 3, 10, 11])
            lock2_idx = np.array([6, 7, 8, 9])
            lock_side_index = np.array([12, 13, 15, 17])
            self.key1_pts = key_pts[key1_idx]
            self.key2_pts = key_pts[key2_idx]
            self.key_side_pts = key_pts[key_side_index]
            self.lock1_pts = lock_pts[lock1_idx]
            self.lock2_pts = lock_pts[lock2_idx]
            self.lock_side_pts = lock_pts[lock_side_index]

        key1_pts = self.key1_pts
        key2_pts = self.key2_pts
        key_side_pts = self.key_side_pts
        lock1_pts = self.lock1_pts
        lock2_pts = self.lock2_pts
        lock_side_pts = self.lock_side_pts
        info["key1_pts"] = key1_pts
        info["key2_pts"] = key2_pts
        info["key_side_pts"] = key_side_pts
        info["lock1_pts"] = lock1_pts
        info["lock2_pts"] = lock2_pts
        info["lock_side_pts"] = lock_side_pts

        observation_left_surface_pts, observation_right_surface_pts = self._get_sensor_surface_vertices() # 获取触觉传感器的表面顶点。 [52,3] [52,3]
        l_diff = np.mean(
            np.sqrt(
                np.sum((self.init_left_surface_pts - observation_left_surface_pts) ** 2, axis=-1)
            )
        )
        r_diff = np.mean(
            np.sqrt(
                np.sum((self.init_right_surface_pts - observation_right_surface_pts) ** 2, axis=-1)
            )
        )
        info["surface_diff"] = np.array([l_diff, r_diff])
        info["tactile_movement_too_large"] = False
        if l_diff > 1.5e-3 or r_diff > 1.5e-3:
            info["tactile_movement_too_large"] = True
        info["relative_motion"] = 1e3 * (self.sensor_grasp_center_current - self.sensor_grasp_center_init) # 乘以1000，将单位从米转换为毫米。 传感器抓握中心的相对运动。
        info["error_too_large"] = False
    #     array([[ 0.0709715 , -0.00302903,  0.02911181],
    #    [ 0.07097052,  0.00296741,  0.02912232],
    #    [ 0.06497135, -0.00303   ,  0.02910838],
    #    [ 0.06497037,  0.00296644,  0.02911888]], dtype=float32)
        if np.abs(info["key1_pts"].mean(0)[1]) > 0.01 or np.abs(info["key2_pts"].mean(0)[1]) > 0.01 or \
                info["key1_pts"].mean(0)[2] > 0.045 or info["key2_pts"].mean(0)[2] > 0.045 or \
                info["key1_pts"].mean(0)[2] < 0.015 or info["key2_pts"].mean(0)[2] < 0.015 or \
                info["key1_pts"].mean(0)[0] > 0.110 or info["key2_pts"].mean(0)[0] > 0.110:
            info["error_too_large"] = True

        info["is_success"] = False
        # key1_pts [4,3] key1_pts[:, 0] [4,] key1_pts[:, 0].max() 0.110 key1_pts[:, 0].min() 0.06497037
        if key1_pts[:, 0].max() < info["lock_side_pts"].mean(0)[0] and key1_pts[:, 0].min() > 0 and \
                key2_pts[:, 0].max() < info["lock_side_pts"].mean(0)[0] and key2_pts[:, 0].min() > 0 and \
                np.abs(key1_pts[:, 1].mean()) < 0.002 and np.abs(key2_pts[:, 1].mean()) < 0.002 and \
                key1_pts[:, 2].min() > 0.037 and key1_pts[:, 2].max() < 0.04 and \
                key2_pts[:, 2].min() > 0.037 and key2_pts[:, 2].max() < 0.04:
            info["is_success"] = True
        return info

    def get_reward(self, info):
        self.error_evaluation_list.append(self.evaluate_error(info)) #369.50026904046535
        reward = -self.step_penalty # 1
        # x_offset
        reward += self.error_evaluation_list[-2] - self.error_evaluation_list[-1] #  6.694728814065456 最后一个reward减去倒数第二个reward

        # print(f"reward part3: {reward}")
        # punish large force
        surface_diff = info["surface_diff"].clip(0.2e-3, 1.5e-3) * 1000 # array([0.2, 0.2])
        reward -= np.sum(10 / (1.55 - surface_diff)) - 15  # max is 200 + 200 = 400 -0.1851851851851869
        # print(f"reward part4: {reward}")

        if info["is_success"]:
            reward += self.final_reward
        elif info["tactile_movement_too_large"] or info["error_too_large"]:
            # prevent the agent from suicide
            reward += -10 * self.step_penalty * (self.max_steps - self.current_episode_elapsed_steps)
        return reward

    def get_truncated(self, info):
        return info["steps"] >= self.max_steps or info["tactile_movement_too_large"] or info["error_too_large"]

    def get_terminated(self, info):
        return info["is_success"]

    def _get_sensor_surface_vertices(self):
        return [
            self.tactile_sensor_1.get_surface_vertices_world(),
            self.tactile_sensor_2.get_surface_vertices_world(),
        ]

    def _sim_step(self, action):

        substeps = max(1, round(np.max(np.abs(action)) / 2e-3 / self.params.sim_time_step))
        v = action / substeps / self.params.sim_time_step
        for _ in range(substeps):
            self.tactile_sensor_1.set_active_v([-v[0], -v[1], -v[2]])
            self.tactile_sensor_2.set_active_v([-v[0], -v[1], -v[2]])
            with suppress_stdout_stderr():
                self.hold_abd.set_kinematic_target(
                    np.concatenate([np.eye(3), np.zeros((1, 3))], axis=0))  # hole stays static
                self.ipc_system.step()
            self.tactile_sensor_1.step()
            self.tactile_sensor_2.step()
            self.sensor_grasp_center_current = (self.tactile_sensor_1.get_pose()[0] +
                                                self.tactile_sensor_2.get_pose()[0]) / 2

            if GUI:
                self.scene.update_render()
                ipc_update_render_all(self.scene)
                self.viewer.render()

    def close(self):
        self.ipc_system = None
        pass

class LongOpenLockRandPointFlowEnv(LongOpenLockSimEnv):
    def __init__(
            self,
            render_rgb: bool = True,
            marker_interval_range: Tuple[float, float] = (2., 2.),
            marker_rotation_range: float = 0.,
            marker_translation_range: Tuple[float, float] = (0., 0.),
            marker_pos_shift_range: Tuple[float, float] = (0., 0.),
            marker_random_noise: float = 0.,
            marker_lose_tracking_probability: float = 0.,
            normalize: bool = False,
            **kwargs,
    ):
        """
        param: render_rgb: whether to render RGB images.
        param: marker_interval_range, in mm.
        param: marker_rotation_range: overall marker rotation, in radian.
        param: marker_translation_range: overall marker translation, in mm. first two elements: x-axis; last two elements: y-xis.
        param: marker_pos_shift: independent marker position shift, in mm, in x- and y-axis. caused by fabrication errors.
        param: marker_random_noise: std of Gaussian marker noise, in pixel. caused by CMOS noise and image processing.
        param: loss_tracking_probability: the probability of losing tracking, appled to each marker
        """
        self.render_rgb = render_rgb
        self.sensor_meta_file = kwargs.get("params").tac_sensor_meta_file
        self.marker_interval_range = marker_interval_range
        self.marker_rotation_range = marker_rotation_range
        self.marker_translation_range = marker_translation_range
        self.marker_pos_shift_range = marker_pos_shift_range
        self.marker_random_noise = marker_random_noise
        self.marker_lose_tracking_probability = marker_lose_tracking_probability
        self.normalize = normalize
        self.default_camera_params = np.array([0, 0, 0, 0, 0, 530, 530, 0, 2.4])
        self.marker_flow_size = 128

        super(LongOpenLockRandPointFlowEnv, self).__init__(**kwargs)

    def _get_sensor_surface_vertices(self):
        return [
            self.tactile_sensor_1.get_surface_vertices_camera(),
            self.tactile_sensor_2.get_surface_vertices_camera(),
        ]

    def add_tactile_sensors(self, init_pos_l, init_rot_l, init_pos_r, init_rot_r):
        self.tactile_sensor_1 = VisionTactileSensorSapienIPC(
            scene=self.scene,
            ipc_system=self.ipc_system,
            meta_file=self.params.tac_sensor_meta_file,
            init_pos=init_pos_l,
            init_rot=init_rot_l,
            elastic_modulus=self.params.tac_elastic_modulus_l,
            poisson_ratio=self.params.tac_poisson_ratio_l,
            density=self.params.tac_density_l,
            name="tactile_sensor_1",
            marker_interval_range=self.marker_interval_range,
            marker_rotation_range=self.marker_rotation_range,
            marker_translation_range=self.marker_translation_range,
            marker_pos_shift_range=self.marker_pos_shift_range,
            marker_random_noise=self.marker_random_noise,
            marker_lose_tracking_probability=self.marker_lose_tracking_probability,
            normalize=self.normalize,
            marker_flow_size=self.marker_flow_size,
            no_render=self.no_render,
        )

        self.tactile_sensor_2 = VisionTactileSensorSapienIPC(
            scene=self.scene,
            ipc_system=self.ipc_system,
            meta_file=self.params.tac_sensor_meta_file,
            init_pos=init_pos_r,
            init_rot=init_rot_r,
            elastic_modulus=self.params.tac_elastic_modulus_r,
            poisson_ratio=self.params.tac_poisson_ratio_r,
            density=self.params.tac_density_r,
            name="tactile_sensor_2",
            marker_interval_range=self.marker_interval_range,
            marker_rotation_range=self.marker_rotation_range,
            marker_translation_range=self.marker_translation_range,
            marker_pos_shift_range=self.marker_pos_shift_range,
            marker_random_noise=self.marker_random_noise,
            marker_lose_tracking_probability=self.marker_lose_tracking_probability,
            normalize=self.normalize,
            marker_flow_size=self.marker_flow_size,
            no_render=self.no_render,
        )

    def get_obs(self, info=None):
        obs = super().get_obs(info=info)
        obs.pop("surface_pts") # 删除surface_pts
        obs["marker_flow"] = np.stack( # 增加marker_flow
            [
                self.tactile_sensor_1.gen_marker_flow(),
                self.tactile_sensor_2.gen_marker_flow(),
            ],
            axis=0
        ).astype(np.float32)

        key1_pts = obs.pop("key1_pts") # （4，3）
        key2_pts = obs.pop("key2_pts")
        # key_end_pts = obs.pop("key_end_pts")
        # key1_pts.mean(0) 每一列（四个点）求均值[0.0709715 , -0.00302903,  0.02911181]
        obs["key1"] = np.array([key1_pts.mean(0)[0] - info["lock1_pts"].mean(0)[0], key1_pts.mean(0)[1], key1_pts.mean(0)[2] - 0.03], # 高度0.3固定
                               dtype=np.float32) * 200.0
        obs["key2"] = np.array([key2_pts.mean(0)[0] - info["lock2_pts"].mean(0)[0], key2_pts.mean(0)[1], key2_pts.mean(0)[2] - 0.03],
                               dtype=np.float32) * 200.0

        if self.render_rgb:
            obs["rgb_images"] = np.stack(
                [
                    self.tactile_sensor_1.gen_rgb_image(),
                    self.tactile_sensor_2.gen_rgb_image(),
                ],
                axis=0
            )
        return obs


if __name__ == "__main__":

    def visualize_marker_point_flow(o, i, name, save_dir="marker_flow_images_4"):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        lr_marker_flow = o["marker_flow"] # 'marker_flow_images_4'
        l_marker_flow, r_marker_flow = lr_marker_flow[0], lr_marker_flow[1] # (2, 128, 2)
        plt.figure(1, (20, 9))
        ax = plt.subplot(1, 2, 1)
        ax.scatter(l_marker_flow[0, :, 0], l_marker_flow[0, :, 1], c="blue")
        ax.scatter(l_marker_flow[1, :, 0], l_marker_flow[1, :, 1], c="red")
        plt.xlim(15, 315)
        plt.ylim(15, 235)
        ax.invert_yaxis()
        ax = plt.subplot(1, 2, 2)
        ax.scatter(r_marker_flow[0, :, 0], r_marker_flow[0, :, 1], c="blue")
        ax.scatter(r_marker_flow[1, :, 0], r_marker_flow[1, :, 1], c="red")
        plt.xlim(15, 315)
        plt.ylim(15, 235)
        ax.invert_yaxis()

        # Save the figure with a filename based on the loop parameter i
        filename = os.path.join(save_dir, f"sp-from-sapien-{name}-marker_flow_{i}.png")
        plt.savefig(filename)
        # plt.show(block=False)
        plt.show()
        plt.pause(1)	
        plt.close()

    def visualize_depth(o):
        lr_depth = o["rgb_images"]
        l_depth, r_depth = lr_depth[0], lr_depth[1]
        cv2.namedWindow("l_depth")     # 创建一个image的窗口
        cv2.imshow("l_depth", l_depth)    # 显示图像
        cv2.namedWindow("r_depth")     # 创建一个image的窗口
        cv2.imshow("r_depth", r_depth)    # 显示图像 
        cv2.waitKey(1000)               # 默认为0，无限等待
        


    GUI = True
    timestep = 0.05

    params = LongOpenLockParams(
        sim_time_step=timestep,
        tac_sensor_meta_file="gelsight_mini_e430/meta_file",
        key_lock_path_file="configs/key_and_lock/key_lock.txt",
        indentation_depth=1.0,
        elastic_modulus_r=3e5,
        elastic_modulus_l=3e5,
    )
    print(params)
#     -----------Parameters-----------------
#           allow_self_collision: 0                                                           
#                  ccd_max_iters: 10                                                          
#                  ccd_slackness: 0.7                                                         
#        ccd_tet_inversion_thres: 0.0                                                         
#                  ccd_thickness: 0.0                                                         
#              ee_classify_thres: 0.001                                                       
#             ee_mollifier_thres: 0.001                                                       
#              indentation_depth: 1.0                                                         
#                   key_friction: 1.0                                                         
#             key_lock_path_file: configs/key_and_lock/key_lock.txt                           
#          line_search_max_iters: 10                                                          
#                  lock_friction: 1.0                                                         
#                      sim_d_hat: 0.0002                                                      
#                      sim_eps_d: 0.0001                                                      
#                      sim_eps_v: 0.001                                                       
#                      sim_kappa: 1000.0                                                      
#               sim_kappa_affine: 100000.0                                                    
#                  sim_kappa_con: 10000000000.0                                               
#  sim_solver_cg_error_frequency: 10                                                          
#  sim_solver_cg_error_tolerance: 0.0001                                                      
#        sim_solver_cg_max_iters: 50                                                          
#    sim_solver_newton_max_iters: 4                                                           
#                  sim_time_step: 0.05                                                        
#                  tac_density_l: 1000                                                        
#                  tac_density_r: 1000                                                        
#          tac_elastic_modulus_l: 1000000.0                                                   
#          tac_elastic_modulus_r: 1000000.0                                                   
#                   tac_friction: 1.0                                                         
#            tac_poisson_ratio_l: 0.3                                                         
#            tac_poisson_ratio_r: 0.3                                                         
#           tac_sensor_meta_file: gelsight_mini_e430/meta_file                                
# -----------------------End-----------------------

    env = LongOpenLockRandPointFlowEnv(
        params=params,
        step_penalty=1,
        final_reward=10,
        max_action=np.array([2, 2, 2]),
        max_steps=10,
        key_x_max_offset=0,
        key_y_max_offset=0,
        key_z_max_offset=0,
        sensor_offset_x_range_len=2.0,
        senosr_offset_z_range_len=2.0,
        marker_interval_range=(2.0625, 2.0625),
        marker_rotation_range=0.,
        marker_translation_range=(0., 0.),
        marker_pos_shift_range=(0., 0.),
        marker_random_noise=0.1,
        normalize=False,
    )

    np.set_printoptions(precision=4)

    offset = [0, 0, 0]

    o, _ = env.reset(offset)
    for k, v in o.items():
        print(k, v.shape)
    info = env.get_info()
    print("timestep: ", timestep)

    for i in range(25):
        obs, rew, done, _, info = env.step(np.array([0.5, 0.0, 0.0]))
        visualize_marker_point_flow(obs, i, "test")
        visualize_depth(obs)
        print(
            f"step: {env.current_episode_elapsed_steps:2d} rew: {rew:.2f} done: {done} success: {info['is_success']}"
        )

    for i in range(4):
        obs, rew, done, _, info = env.step(np.array([0.0, 0.0, -0.5]))
        visualize_marker_point_flow(obs, i + 24, "test")
        visualize_depth(obs)
        print(
            f"step: {env.current_episode_elapsed_steps:2d} rew: {rew:.2f} done: {done} success: {info['is_success']}"
        )

    for i in range(10):
        obs, rew, done, _, info = env.step(np.array([0.5, 0.0, 0.0]))
        visualize_marker_point_flow(obs, i + 28, "test")
        visualize_depth(obs)
        print(
            f"step: {env.current_episode_elapsed_steps:2d} rew: {rew:.2f} done: {done} success: {info['is_success']}"
        )

    for i in range(8):
        obs, rew, done, _, info = env.step(np.array([0.0, 0.0, -0.5]))
        visualize_marker_point_flow(obs, i + 38, "test")
        visualize_depth(obs)
        print(
            f"step: {env.current_episode_elapsed_steps:2d} rew: {rew:.2f} done: {done} success: {info['is_success']}"
        )
    cv2.destroyAllWindows() 