import argparse

import gym
import gym_insertion


def main():
    parser = argparse.ArgumentParser("Insertion, Manual mode")
    parser.add_argument('--host', type=str, help='IP of the server')
    parser.add_argument('--port', type=int, help='Port that should be used to connect to the server')
    args = parser.parse_args()

    if args.host:
        host = args.host
    else:
        host = "192.168.2.121"   # Windows server
    if args.port:
        port = args.port
    else:
        port = 9090

    env = gym.make('insertion-v0')
    goal_img, goal_coord = env.start(host, port)
    print(f"Goal coord: {goal_coord}")

    # Action: [0, 0, 0, 0, 0, 0]  # New coord: x, y, z, alpha, beta, gamma
    go_down_action = [0, -1, 0, 0, 0, 0]
    go_up_action = [0, 1, 0, 0, 0, 0]
    go_left_action = [-1, 0, 0, 0, 0, 0]
    go_right_action = [1, 0, 0, 0, 0, 0]

    for _ in range(5):
        print("Going down")
        new_state, reward, done, infos = env.step(go_down_action)
        print(f"New coord: {new_state[1]},  Done: {done}")
    for _ in range(5):
        print("Going up")
        new_state, reward, done, infos = env.step(go_up_action)
        print(f"New coord: {new_state[1]},  Done: {done}")
    for _ in range(5):
        print("Going left")
        new_state, reward, done, infos = env.step(go_left_action)
        print(f"New coord: {new_state[1]},  Done: {done}")
    for _ in range(5):
        print("Going right")
        new_state, reward, done, infos = env.step(go_right_action)
        print(f"New coord: {new_state[1]},  Done: {done}")


if __name__ == "__main__":
    main()
