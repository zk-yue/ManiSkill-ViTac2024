import os
import sys
import time

script_path = os.path.dirname(os.path.realpath(__file__))
repo_path = os.path.join(script_path, "..")
sys.path.insert(0, repo_path)

import gymnasium as gym
import ruamel.yaml as yaml
import torch
from path import Path

# from solutions.policies import (
#     TD3PolicyForPointFlowEnv, TD3PolicyForLongOpenLockPointFlowEnv
# )
from solutions.policies_sac import SACPolicyForPointFlowEnv
# from stable_baselines3 import TD3
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (CallbackList,
                                                CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils.common import get_time
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.evaluation import evaluate_policy




import wandb
from arguments import *

import gymnasium as gym
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization

algorithm_aliases = {
    "SAC": SAC,
}

SAC.policy_aliases["SACPolicyForPointFlowEnv"] = SACPolicyForPointFlowEnv
# SAC.policy_aliases["TD3PolicyForLongOpenLockPointFlowEnv"] = TD3PolicyForLongOpenLockPointFlowEnv


def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for SAC hyperparameters."""
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    batch_size = trial.suggest_int('batch_size', 256, 1024),
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.5, log=True)

    return {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "batch_size" : batch_size,
    }

buffer_size: 200000
  # train_freq: 2
  # gradient_steps: -1
  # learning_starts: 128
  # action_noise: 0.5
  # batch_size: 128
  # learning_rate: 0.0003


class TrialEvalCallback(EvalCallback):
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        trial: optuna.Trial,
        n_eval_episodes: int,
        eval_freq: int,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self.success_rate = -np.inf

    def _on_step(self) -> bool:
        
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)
                self.success_rate = success_rate
 
            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        # 新增部分
        self.eval_idx += 1
        self.trial.report(self.success_rate, self.eval_idx)
        # Prune trial if need.
        if self.trial.should_prune():
            self.is_pruned = True
            return False
        return True


def make_env(env_name, seed=0, i=0, **env_args):
    num_devices = torch.cuda.device_count()
    assert num_devices > 0
    wp_device = f"cuda:{i % num_devices}"
    def _init():
        env = gym.make(env_name, device=wp_device, **env_args)

        return env

    set_random_seed(seed)

    return _init

def objective(trial: optuna.Trial) -> float:
    parser = get_parser()
    args = parser.parse_args()
    args.cfg='configs/parameters/peg_insertion_sac_optuna.yaml'
    with open(args.cfg, "r") as f:
        cfg = yaml.YAML(typ='safe', pure=True).load(f)

    # solve argument conflict
    cfg = solve_argument_conflict(args, cfg)
    exp_start_time = get_time()
    exp_name = f"{cfg['train']['name']}_{exp_start_time}"
    cfg["train"]["emp"] = {}
    log_dir = os.path.join(repo_path, f"training_log/{exp_name}")
    Path(log_dir).makedirs_p()

    with open(os.path.join(log_dir, "cfg.yaml"), "w") as f:
        yaml.YAML(typ='unsafe', pure=True).dump(cfg, f)

    env_name = cfg["env"].pop("env_name")
    params = cfg["env"].pop("params")
    params_lb, params_ub = parse_params(env_name, params)

    if "max_action" in cfg["env"].keys():
        cfg["env"]["max_action"] = np.array(cfg["env"]["max_action"])

    specified_env_args = copy.deepcopy(cfg["env"])
    specified_env_args.update(
        {
            "params": params_lb,
            "params_upper_bound": params_ub,
            "no_render": args.no_render,
        }
    )
    with open(Path(log_dir) / "params_lb.txt", "w") as f:
        f.write(str(params_lb))
    with open(Path(log_dir) / "params_ub.txt", "w") as f:
        f.write(str(params_ub))

    if cfg["train"]["seed"] >= 0:
        seed = cfg["train"]["seed"]
    else:
        seed = int(time.time())
        cfg["train"]["seed"] = seed
    parallel_num = cfg["train"]["parallel"]

    env = SubprocVecEnv(
        [
            make_env(
                env_name,
                seed,
                i,
                **specified_env_args,
            )
            for i in range(parallel_num)
        ]
    )
    eval_env = gym.make(env_name, **specified_env_args)

    device = "cpu"
    if torch.cuda.is_available():
        if torch.cuda.device_count() > cfg["train"]["gpu"]:
            device = f"cuda:{cfg['train']['gpu']}"
        else:
            device = "cuda"

    cfg["train"]["device"] = device
    set_random_seed(seed)
    policy_name = cfg["policy"].pop("policy_name")
    cfg = handle_policy_args(cfg, log_dir, action_dim=env.action_space.shape[0])

    algorithm_class = algorithm_aliases[cfg["train"]["algorithm_name"]]

    kwargs=sample_sac_params(trial)

    model = algorithm_class(
        policy_name,
        env,
        verbose=1,
        **cfg["policy"],
        **kwargs,
    )

    weight_dir = os.path.join(log_dir, "weights")
    Path(weight_dir).makedirs_p()

    checkpoint_callback = CheckpointCallback(
        save_freq=max(cfg["train"]["checkpoint_every"] // parallel_num, 1),
        save_path=weight_dir,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        n_eval_episodes=cfg["train"]["n_eval"],
        eval_freq=max(cfg["train"]["eval_freq"] // parallel_num, 1),
        log_path=log_dir,
        best_model_save_path=log_dir,
        deterministic=True,
    )

    # WANDB = False
    # if WANDB:
    #     wandb_run = wandb.init(
    #         project=cfg["train"]["wandb_name"],
    #         name=f"{cfg['train']['name']}_{exp_start_time}",
    #         entity="openlock",
    #         config=cfg,
    #         sync_tensorboard=True,
    #         monitor_gym=False,
    #         save_code=True,
    #     )
    #     wandb_callback = WandbCallback(
    #         verbose=2,
    #     )
    #     callback = CallbackList([checkpoint_callback, eval_callback, wandb_callback])
    # else:
        # callback = CallbackList([checkpoint_callback, eval_callback])
    callback = CallbackList([checkpoint_callback, eval_callback])

    nan_encountered = False
    try:
        model.learn(
        total_timesteps=25000, callback=callback, log_interval=cfg["train"]["log_interval"]
    )
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        env.close()
        eval_env.close()

    # if WANDB:
    #     wandb_run.finish()
    # model.save(os.path.join(log_dir, "rl_model_final.zip"))
    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.success_rate

if __name__ == "__main__":
    N_TRIALS = 3
    N_STARTUP_TRIALS = 1

    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=2500)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=None)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))