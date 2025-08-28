import sys
from pathlib import Path
import argparse
from stable_baselines3 import PPO
from myosuite.utils import gym

register = gym.register

current_dir = Path("").resolve()
sys.path.append(str(current_dir / 'envs'))

env_id = 'DexterousEnv'
n_procs = 1
max_time = 4
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
    'max_distance_reset': 0.16,
    "max_time_steps": max_time_steps,
    'muscle_noise': True,
    'mix_button_size': True,
    'hold_threshold': 1,
    'action_masking': True,  # Enable action masking by default
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize a trained policy.")
    parser.add_argument('--policy_path', type=str, default="policies/PPO/512_512/512_512_curriculum",
                        help="Path to the trained policy file.")
    parser.add_argument('--action_masking', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Enable action masking (default: True).")
    args = parser.parse_args()
    env_config['action_masking'] = args.action_masking
    register(
        id=env_id,
        entry_point='dexterous_env:DexterousEnv',
        kwargs={"env_config": env_config}
    )
    model_path = args.policy_path

    render_env = gym.make(env_id)
    render_env = render_env.unwrapped

    pi = PPO.load(model_path, env=render_env, device='cpu')
    episodes = 200
    for ep in range(episodes):
        obs, _ = render_env.reset()
        while True:
            action, _ = pi.predict(obs)
            render_env.mj_render()
            obs, reward, done, term, smth = render_env.step(action)
            if done:
                print('reward:', reward)
                print('time', smth['obs_dict']['time'])
                break