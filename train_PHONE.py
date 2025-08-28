import copy
import os
import sys
from pathlib import Path
from typing import Callable

from myosuite.agents.in_callbacks import InfoCallback, FallbackCheckpoint, EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback

from stable_baselines3 import PPO
from myosuite.utils import gym;
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from cusstom_callaback import SolvedTrackingEvalCallback


register = gym.register



########################################################################################################################################################


def linear_schedule(initial_value: float, min_value: float, threshold: float = 1.0) -> Callable[[float], float]:
    """
    Linear learning rate schedule. Adapted from the example at
    https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#learning-rate-schedule

    :param initial_value: Initial learning rate.
    :param min_value: Minimum learning rate.
    :param threshold: Threshold (of progress) when decay begins.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if progress_remaining > threshold:
            return initial_value
        else:
            return min_value + (progress_remaining / threshold) * (initial_value - min_value)

    return func



env_id = 'DexterousTouchEnv'
current_dir = Path("").resolve()
sys.path.append(str(current_dir / 'envs'))

n_procs = 18
max_time = 3

frame_skip = 3
max_time_steps = int(max_time / (0.002 * frame_skip))

env_config = {
    'model_path': (current_dir / 'envs/scene/phone_dexterous.xml').as_posix(),
    'MAX_TIME': max_time,
    'frame_skip': frame_skip,
    'train_level': 2,
    'contact_height': 0.012,
    'contact_distance': True,
    'phone_surface_distance': 0.0001,
    'max_distance_reset': 0.16,  # 20cm reset
    "max_time_steps": max_time_steps,
    'muscle_noise': True,
    'mix_button_size': True,
    'hold_threshold': 1,
}

eval_config = copy.deepcopy(env_config)
register(
    id=env_id,
    entry_point='dexterous_env:DexterousEnv',
    kwargs={"env_config": env_config
            }
)

register(
    id='eval_env',
    entry_point='dexterous_env:DexterousEnv',
    kwargs={"env_config": eval_config
            }
)

policy_kwargs = {
    "net_arch": {
        "pi": [256, 256],
        "vf": [256, 256]
    },
    "ortho_init":True
}

# Ensure that the code is wrapped in a `if __name__ == '__main__':` block for multiprocessing
if __name__ == '__main__':
    models_dir = "policies/PPO/256_256_dexterous_touch"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_name = '256_256'
    logs_dir = f"logs/{model_name}"
    model_path = f"{models_dir}/256_256"
    normal_path = f"{models_dir}/256_256_vecnormalize.pkl"
    train_env = make_vec_env(env_id, n_envs=n_procs)

    train_env = SubprocVecEnv([lambda: gym.make(env_id) for _ in range(n_procs)])
    # Wrap with reward normalization
    train_env = VecNormalize(train_env, norm_reward=True, norm_obs=False)

    eval_env = make_vec_env('eval_env', n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True)
    # Setup logger
    new_logger = configure(logs_dir, ["stdout", "csv", "tensorboard"])
    # Custom success rate callback
    solved_callback = SolvedTrackingEvalCallback(eval_env, train_env, n_eval_episodes=100, eval_freq=10000)

    model = PPO("MlpPolicy", train_env,verbose=1, tensorboard_log=logs_dir, policy_kwargs=policy_kwargs,
                n_steps=max_time_steps, seed=42,clip_range=lambda progress_remaining: 0.2 * progress_remaining,
                batch_size=n_procs * max_time_steps,n_epochs=10,
                learning_rate=linear_schedule(6e-4, 1e-5,
                                              0.6))
    callback = []
    callback = [solved_callback]

    callback += [EvalCallback(eval_env=eval_env, n_eval_episodes=100, eval_freq=10000)]
    callback += [InfoCallback()]
    callback += [FallbackCheckpoint(27200)]
    callback += [CheckpointCallback(save_vecnormalize=True,save_freq=10000, save_path=f"{models_dir}/",
                                    name_prefix=model_name)]
    model.set_logger(new_logger)

    # load a baseline with vec norm
    # train_env = VecNormalize.load(
    #     load_path=normal_path, venv=train_env)
    # pi = PPO.load(model_path, env=train_env)
    # model.set_parameters(pi.get_parameters())

    model.learn(
        total_timesteps=500000000,
        callback=callback,
    )