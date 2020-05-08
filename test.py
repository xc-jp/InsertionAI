import argparse
import os

import gym
import gym_insertion  # noqa: F401
from stable_baselines import SAC


def main():
    parser = argparse.ArgumentParser("Insertion, Manual mode")
    parser.add_argument('checkpoint_path', type=str, help='Path to checkpoint')
    parser.add_argument('--host', default="192.168.2.121", type=str, help='IP of the server (default is a Windows#2)')
    parser.add_argument('--port', default=9090, type=int, help='Port that should be used to connect to the server')
    parser.add_argument('--use_coord', action="store_true", help=('If set, the environment\'s observation space will be'
                                                                  'coordinates instead of images'))
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    env = gym.make('insertion-v0', kwargs={'host': args.host, "port": args.port, "use_coord": args.use_coord})

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    if args.use_coord:
        model = SAC('MlpPolicy', env, verbose=1, tensorboard_log="../insertion_tensorboard/")
    else:
        model = SAC('CnnPolicy', env, verbose=1, tensorboard_log="../insertion_tensorboard/")
    model.load(args.checkpoint_path, env=env)

    obs = env.reset()
    for i in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


if __name__ == "__main__":
    main()
