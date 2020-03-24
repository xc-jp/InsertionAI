import argparse
import time

import gym
import gym_insertion
import numpy as np
from Pillow import Image


def main():
    parser = argparse.ArgumentParser("Insertion, Manual mode")
    parser.add_argument('--host', default="192.168.2.121", type=str, help='IP of the server (default is a Windows server)')
    parser.add_argument('--port', default=9090, type=int, help='Port that should be used to connect to the server')
    args = parser.parse_args()

    env = gym.make('insertion-v0')
    goal_img, goal_coord = env.start(args.host, args.port)
    print(f"Goal coord: {goal_coord}")

    # Action: [0, 0, 0, 0, 0, 0]  # New coord: x, y, z, alpha, beta, gamma
    go_down_action = [0, -1, 0, 0, 0, 0]
    go_up_action = [0, 1, 0, 0, 0, 0]
    go_left_action = [-1, 0, 0, 0, 0, 0]
    go_right_action = [1, 0, 0, 0, 0, 0]

    delay = 1

    print("Rotating")
    new_state, reward, done, infos = env.step([0, 0, 0, -90, 0, 0])
    print(f"New coord: {new_state[1]},  Done: {done}")

    for i in range(5):
        print("Going down")
        new_state, reward, done, infos = env.step(go_down_action)
        print(f"New coord: {new_state[1]},  Done: {done}")
        img, coord = new_state
        img = Image.fromarray(np.uint8(img))
        img.save(f"Img_{i}", "JPEG")
        time.sleep(delay)
    for _ in range(5):
        print("Going up")
        new_state, reward, done, infos = env.step(go_up_action)
        print(f"New coord: {new_state[1]},  Done: {done}")
        time.sleep(delay)
    for _ in range(5):
        print("Going left")
        new_state, reward, done, infos = env.step(go_left_action)
        print(f"New coord: {new_state[1]},  Done: {done}")
        time.sleep(delay)
    for _ in range(5):
        print("Going right")
        new_state, reward, done, infos = env.step(go_right_action)
        print(f"New coord: {new_state[1]},  Done: {done}")
        time.sleep(delay)


if __name__ == "__main__":
    main()
