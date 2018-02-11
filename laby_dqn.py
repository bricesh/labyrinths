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
        with open('./maps/' + map_name + '.txt', 'r') as f:
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

    def get_state(self, x_pos, y_pos):
        return self.laby[y_pos - 1][x_pos],\
               self.laby[y_pos + 1][x_pos],\
               9 if self.laby[y_pos][x_pos - 1] == 1 else self.laby[y_pos][x_pos - 1],\
               self.laby[y_pos][x_pos + 1]


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
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1 + K.square(error)) - 1, axis=-1)

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def init_Q(self):
        with open('./models/init_Q.txt', 'r') as f:
            init_Q = [line.replace('\n', '').split(',') for line in f]

        init_Q = np.array(init_Q).astype(np.float)

        self.model.fit(init_Q[:,0:4].astype(np.int), init_Q[:,4:8], epochs=1000, verbose=0)
        self.update_target_model()
        self.save("./models/model_after_Q_init.h5")
        #predicted_Q = self.target_model.predict(init_Q[:,0:4].astype(np.int))
        #np.set_printoptions(precision=3, suppress=True)
        #print(predicted_Q)
        #print(init_Q[:,4:8])

#    def move(self,dir_choice):
#        if dir_choice==0:
#            y_agent -= 1
#        elif dir_choice==1:
#            y_agent += 1
#        elif dir_choice==2:
#            x_agent -= 1
#        elif dir_choice==3:
#            x_agent += 1
#        else:
#            print("I'm stuck")

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    laby_env = Env("laby1")
    print(laby_env.x_entrance, laby_env.y_entrance)
    cur_state = laby_env.get_state(laby_env.x_entrance+2, laby_env.y_entrance)
    print(cur_state)
    agent = DQNAgent(4,4)
    agent.init_Q()
