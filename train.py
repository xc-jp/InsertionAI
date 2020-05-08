import argparse
import os

import gym
import gym_insertion  # noqa: F401
# from stable_baselines.common.env_checker import check_env
from stable_baselines import SAC
from stable_baselines.common.callbacks import CkptCallback


def main():
    parser = argparse.ArgumentParser("Insertion, Manual mode")
    parser.add_argument('--host', default="127.0.0.1", type=str, help='IP of the server')
    parser.add_argument('--port', default=9081, type=int, help='Port that should be used to connect to the server')
    parser.add_argument('--save', action="store_true", help=('Saves checkpoints'))
    parser.add_argument('--use_coord', action="store_true", help=('If set, the environment\'s observation space will be'
                                                                  'coordinates instead of images'))
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    env = gym.make('insertion-v0', kwargs={'host': args.host, "port": args.port, "use_coord": args.use_coord})
    # check_env(env, warn=True)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    # print(env.action_space.sample())

    # Save a checkpoint every 50000 steps
    ckpt = CkptCallback(save_freq=50000, save_path='../checkpoints/', name_prefix='rl_insertion') if args.save else None

    if args.use_coord:
        model = SAC('MlpPolicy', env, verbose=1, tensorboard_log="../insertion_tensorboard/")
    else:
        model = SAC('CnnPolicy', env, verbose=1, tensorboard_log="../insertion_tensorboard/")

    model.learn(50001, callback=ckpt)


if __name__ == "__main__":
    main()
