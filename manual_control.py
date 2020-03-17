import gym
import gym_insertion


def main():
    env = gym.make('insertion-v0')
    
    go_down_action = 0
    env.step(go_down_action)


if __name__ == "__main__":
    main()
