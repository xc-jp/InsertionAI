import socket
import json
import traceback
import logging
import base64
import math

import numpy as np
import gym
from gym import spaces


logging.basicConfig(filename='errors.log', level=logging.WARNING)  # logging.DEBUG

# AI_MSG =  {
#     "action": "FIRST",  # String, can be "FIRST", "RESET", "STEP" or (optional) "CLOSE"
#     "coord": [0, 0, 0, 0, 0, 0]  # New coord: x, y, z, alpha, beta, gamma   (3D coord)
# }

# --> FIRST means that this is the first/init message from the AI, "coord" is empty.
#     * Response should be of type "FIRST".
#       - "coord" shoud be the current position and orientation of the goal.
#       - "image" shoud be the goal image, i.e., image of the connector succesfully inserted.
#       - "done" should be None
# --> RESET means that the AI succesfully inserted the object previously or that it took too long, "coord" is empty
#     * Response should be of type "RESET".
#       - "coord" should be the current position and orientation of the effector: x, y, z, alpha, beta, gamma
#       - "image" should be a picture taken from a fix point that shows both end effector and the goal.
#       - "done" should be None
# --> STEP means that the AI wants to move to the point in "coord"
#     * Response should be of type "STEP".
#       - "coord" should be the current position and orientation of the effector: x, y, z, alpha, beta, gamma
#       - "image" should be a picture taken from a fix point that shows both end effector and the goal.
#       - "done" should be either 1 or 0. 1 means that the last action resulted in a succesful insertion.
# --> CLOSE means that the AI wants to disconnect

# SIM_MSG = {
#     "action": "FIRST",  # String, can be "FIRST", "RESET" or "STEP"
#     "image": (INT, INT),  #  Image (int)
#     "coord": [0, 0, 0, 0, 0, 0],  # Array of 6 floats
#     "done": 0  # Integer
# }

RESET_MSG = (json.dumps({
    "action": "RESET",
    "coord": None
}) + "<EOF>").encode("utf-8")

FIRST_MSG = (json.dumps({
    "action": "FIRST",
    "coord": None
}) + "<EOF>").encode("utf-8")

CLOSE_MSG = (json.dumps({
    "action": "CLOSE",
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


def decode_message(message):
    coord = np.asarray(message["coord"], dtype=np.float32)

    # Unity version
    # img = base64.b64decode(message["image"])[:-5]
    # img = Image.open(io.BytesIO(img))

    # Godot version
    img = base64.b64decode(message["image"])  # [:-5]
    img = img[8:]
    img = img.decode("utf-8")[1:-1]
    img = np.fromstring(img, dtype=int, sep=', ')
    img = np.reshape(img, (-1, 3))
    img = np.reshape(img, (-1, int(math.sqrt(len(img))), 3)).astype(np.uint8)
    # img = Image.fromarray(img)

    done = message["done"]

    if message["action"] == "FIRST" or message["action"] == "RESET":
        return img, coord, None
    elif message["action"] == "STEP":
        return img, coord, done
    else:
        logging.error("Received an unexpected action: {}".format(message["action"]))
        return -1


class InsertionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, kwargs):
        super(InsertionEnv, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)  # x, y, z, alpha, beta, gamma
        if kwargs["use_coord"]:
            # x, y, z, alpha, beta, gamma
            self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(6,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

        self._start(kwargs["host"], kwargs["port"])
        self.use_coord = kwargs["use_coord"]

    def _start(self, host, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Trying to connect to the environment", flush=True)
        self.socket.connect((host, port))
        print("Connected to the environment", flush=True)

        sent = self.socket.send(FIRST_MSG)
        if sent == 0:
            raise RuntimeError("socket connection broken")
        message = recv_end(self.socket)
        self.goal_img, self.goal_coord, _ = decode_message(message)

    def _get_reward(self, coord: [float], done: bool) -> float:
        reward = -sum(abs(self.goal_coord[:3] - coord[:3])) + 100 * done
        return reward

    def _get_done(self, coord: [float], done: bool) -> float:
        if done:
            return done
        else:
            return any((*(abs(coord[0:3]) > 20), *(abs(coord[3:6]) > 3.14/2)))

    def step(self, action: [float]) -> [[int], int, bool, [float]]:
        """Executes the given action

        Args:
            action: action to execute, array of 6 floats

        Returns:
            next_state: new state
            reward: The reward, depends on the distance to the goal and on the success of the task
            done: True if the peg was inserted or if the peg moved too far away from the goal
            infos: Additional infos  (potentially the coordinates of the peg)
        """
        step_msg = {}
        step_msg["action"] = "STEP"
        step_msg["coord"] = action.tolist()

        logging.debug(f"Step action: {step_msg}")
        step_msg = json.dumps(step_msg) + "<EOF>"
        sent = self.socket.send(step_msg.encode("utf-8"))
        if sent == 0:
            raise RuntimeError("socket connection broken")
        message = recv_end(self.socket)
        img, coord, done = decode_message(message)

        reward = self._get_reward(coord, done)
        done = self._get_done(coord, done)
        if self.use_coord:
            new_state = coord
            infos = {}
        else:
            new_state = img
            infos = {"coord": coord}

        logging.debug(f"New position: {coord}, new reward: {reward}")
        return new_state, reward, done, infos

    def reset(self):
        """Tells the server to reset the environnement

        Returns:
            new_state: observation corresponding to the first state of the new episode
        """
        # Send reset msg to Unity, get back resetted env's state
        sent = self.socket.send(RESET_MSG)
        if sent == 0:
            raise RuntimeError("socket connection broken")
        message = recv_end(self.socket)
        img, coord, _ = decode_message(message)
        if self.use_coord:
            new_state = coord
        else:
            new_state = img
        return new_state

    def render(self, mode='human', close=False):
        """ Does nothing since the rendering is done in Unity/Godot"""
        # Should this actually show the image sent by Unity (current state) ?
        return 0

    def close(self):
        """Terminates connection with the server"""
        sent = self.socket.send(CLOSE_MSG)
        if sent == 0:
            raise RuntimeError("socket connection broken")
