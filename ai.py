import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from matplotlib import pyplot as plt
import parseconf

# input 56x32x1

settings = parseconf.parsefile()

PLAYER_NAME = settings["p2_name"]

STRING_VALUE_MAP = {
    "walls": 0,
    "floor": 1,
    "deadly": 2,
    "own_player": 3,
    "enemy_player": 4,
}


class Net(nn.Module):
    def __init(self):
        super(Net, self).__init__()
        # input is 32x32 image 1 channel. Output is (32-3+1)x(32-3+1) in 6 channels
        self.conv1 = nn.Conv2d(1, 6, 3)
        # maxpool with 2x2 kernel gives 15x15
        # input 15x15 6 channels. Output 13x13, 16 channels
        self.conv2 = nn.Conv2d(6, 16, 3)
        # maxpool with 2x2 kernel gives 6x6
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.lin1 = nn.Linear(32*4*4, 100)
        self.lin2 = nn.Linear(100, 50)
        self.lin3 = nn.Linear(50, 10)


def string_state_to_int_state(state):
    players_health = [0,0]
    np_state = np.zeros((len(state), len(state[0])), dtype=np.uint8)
    for i in range(np_state.shape[0]):
        for j in range(np_state.shape[1]):
            if state[i][j][1] != '':
                if type(state[i][j][1]).__name__ == "Bullet":
                    np_state[i, j] = 6
                else:
                    if state[i][j][1].name == settings["p1_name"]:
                        players_health[0] = state[i][j][1].health
                    else:
                        players_health[1] = state[i][j][1].health
                    np_state[i, j] = 5
            else:
                np_state[i, j] = STRING_VALUE_MAP[state[i][j][0]]

    return np_state,players_health


def predict(observations, action_space):

    current_s, current_players_health = string_state_to_int_state(observations)
    #print(observations)
    plt.imshow(np.rot90(current_s))
    plt.gray()
    plt.show()








    actions = action_space.sample()

    return actions
