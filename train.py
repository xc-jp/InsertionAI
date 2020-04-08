import argparse
import time
import os

import gym
import gym_insertion
from stable_baselines.common.env_checker import check_env
from stable_baselines import SAC
from PIL import Image
import numpy as np


def main():
    parser = argparse.ArgumentParser("Insertion, Manual mode")
    parser.add_argument('--host', default="192.168.2.121", type=str, help='IP of the server (default is a Windows#2)')
    parser.add_argument('--port', default=9090, type=int, help='Port that should be used to connect to the server')
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    env = gym.make('insertion-v0', kwargs={'host': args.host, "port": args.port})
    # check_env(env, warn=True)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    # print(env.action_space.sample())

    model = SAC('MlpPolicy', env, verbose=2).learn(5000)


if __name__ == "__main__":
    main()