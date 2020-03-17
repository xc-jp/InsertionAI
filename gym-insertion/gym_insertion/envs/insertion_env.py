import socket
import json
import traceback
import logging

import gym
# from gym import error, spaces, utils
# from gym.utils import seeding


logging.basicConfig(filename='errors.log', level=logging.WARNING)

CLOSE_MSG = (json.dumps({
    "action": "CLOSE",
    "coord": None
}) + "<EOF>").encode("utf-8")

RESET_MSG = (json.dumps({
    "action": "RESET",
    "coord": None
}) + "<EOF>").encode("utf-8")


def recv_end(socket, end="<EOF>", buffer_size=800000):
    total_data = []
    data = ''
    while True:
        data = socket.recv(buffer_size).decode("utf-8")
        if not data:  # recv return empty message if client disconnects
            return data
        if end in data:
            total_data.append(data[:data.find(end)])
            break
        total_data.append(data)
        if len(total_data) > 1:
            # check if end_of_data was split
            last_pair = total_data[-2] + total_data[-1]
            if end in last_pair:
                total_data[-2] = last_pair[:last_pair.find(end)]
                total_data.pop()
                break
    message = ''.join(total_data)
    try:
        message = json.loads(message)
    except (BrokenPipeError, json.decoder.JSONDecodeError):
        logging.warning("Could not decode json", exc_info=True)
        pass
    except Exception:
        logging.error(traceback.format_exc())
    return message


class InsertionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, host, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Trying to connect to the environment")
        self.socket.connect((host, port))
        print("Connected to the environment")

        self.action_space = 4  # Currently using a continuous action space, not applicable
        self.observation_space_shape = (640, 640, 4,)  # Shape of the obvervation space

    def step(self, action: [float]) -> [[int], int, bool, [float]]:
        """Executes the given action

        Args:
            action: action to execute, array of [???]

        Returns:
            next_state: new state
            reward: 1 or 0, depending on wether the thing was inserted or not
            done: True or False, depending on wether the thing was inserted or not
                  (maybe end experiment if the robot's end goes too far off ?)
            infos: Additional infos
        """
        message = recv_end(self.socket)
        new_state, reward, done, infos = decode_message(message)
        return [new_state, reward, done, infos]

    def reset(self) -> [int]:
        """Tells the server to reset the environnement

        Returns:
            new_state: observation corresponding to the first state of the new episode
        """
        # Send reset msg to Unity, get back resetted env's state
        sent = self.socket.send(RESET_MSG)
        if sent == 0:
            raise RuntimeError("socket connection broken")

        message = recv_end(self.socket)
        new_state = decode_message(message)  # Get state from message
        return new_state

    def render(self, mode='human', close=False):
        """ Does nothing since the rendering is done in Unity"""
        # Should this actually show the image sent by Unity (current state) ?
        return 0

    def close(self):
        """Terminates connection with the server"""
        sent = self.socket.send(CLOSE_MSG)
        if sent == 0:
            raise RuntimeError("socket connection broken")
