import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K


class Env:
    def __init__(self, map_name):
        self._load_map(map_name)
        self._find_entrance()

    def _load_map(self, map_name):
        with open('maps/' + map_name + '.txt', 'r') as f:
            self.laby = [line.replace('\n', '').split(',') for line in f]
        for i in range(0, len(self.laby)):
            for j in range(0, len(self.laby)):
                self.laby[i][j] = int(self.laby[i][j])

    def _find_entrance(self):
        for i in range(0, len(self.laby)):
            try:
                self.x_entrance = self.laby[i].index(1)
                self.y_entrance = i
            except ValueError:
                continue


class DQNAgent:
    def __init__(self, state_size, action_size):
        print("Agent init")
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001

    #def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
    #    error = prediction - target
    #    return K.mean(K.sqrt(1 + K.square(error)) - 1, axis=-1)


if __name__ == "__main__":
    laby_env = Env("laby1")
    print laby_env.x_entrance, laby_env.y_entrance
    print laby_env.laby
    agent = DQNAgent(4,4)
