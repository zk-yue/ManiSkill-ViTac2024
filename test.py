import numpy as np
# # array1 = np.random.rand(3, 4, 5)
# # queue = np.full(4, array1)
# # print(queue)
# # Create four 2D numpy arrays
# array1 = np.random.rand(3, 5)
# array2 = np.random.rand(3, 5)
array2 = np.random.rand(2,3, )
array3 = np.random.rand(2,3, )
distance = np.linalg.norm(array3 - array2, axis=-1)
print(distance)
print(array3)
array4 = np.zeros((2,3,), dtype=np.float32)
print(array4)
print(array4.shape)
print(len(array4.shape))
print(len(array4.shape)>1)

# # Stack these arrays along a new dimension to form a 3D numpy array
# three_d_array = np.stack((array1, array2, array3, array4), axis=1)

# print(three_d_array.shape)

# from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
# from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
# from stable_baselines3.common.envs import BitFlippingEnv

# model_class = DQN  # works also with SAC, DDPG and TD3
# N_BITS = 15

# env = BitFlippingEnv(n_bits=N_BITS, continuous=model_class in [DDPG, SAC, TD3], max_steps=N_BITS)

# # Available strategies (cf paper): future, final, episode
# goal_selection_strategy = "future" # equivalent to GoalSelectionStrategy.FUTURE

# # Initialize the model
# model = model_class(
#     "MultiInputPolicy",
#     env,
#     replay_buffer_class=HerReplayBuffer,
#     # Parameters for HER
#     replay_buffer_kwargs=dict(
#         n_sampled_goal=4,
#         goal_selection_strategy=goal_selection_strategy,
#     ),
#     verbose=1,
# )

# # Train the model
# model.learn(1000)

# model.save("./her_bit_env")
# # Because it needs access to `env.compute_reward()`
# # HER must be loaded with the env
# model = model_class.load("./her_bit_env", env=env)

# obs, info = env.reset()
# for _ in range(100):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, _ = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()