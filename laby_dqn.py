import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

EPISODES = 1000


class Env:
    def __init__(self, map_name):
        self.state_size = 4
        self.action_size = 4
        self._x_entrance = 0
        self._y_entrance = 0
        self.cur_x_pos = 0
        self.cur_y_pos = 0
        self._load_map(map_name)
        self._find_entrance()
        self.reset()

    def _load_map(self, map_name):
        # Import map from file
        with open('./maps/' + map_name + '.txt', 'r') as f:
            self.laby = [line.replace('\n', '').split(',') for line in f]
        # Set values to int
        for i in range(0, len(self.laby)):
            for j in range(0, len(self.laby)):
                self.laby[i][j] = int(self.laby[i][j])

    def _find_entrance(self):
        for i in range(0, len(self.laby)):
            try:
                self._x_entrance = self.laby[i].index(1)
                self._y_entrance = i
            except ValueError:
                continue

    def get_state(self):
        return self.laby[self.cur_y_pos - 1][self.cur_x_pos],\
               self.laby[self.cur_y_pos + 1][self.cur_x_pos],\
               9 if self.laby[self.cur_y_pos][self.cur_x_pos - 1] == 1 else self.laby[self.cur_y_pos][self.cur_x_pos - 1],\
               self.laby[self.cur_y_pos][self.cur_x_pos + 1]

    def step(self, dir_choice):
        if dir_choice == 0:
            self.cur_y_pos -= 1
        elif dir_choice == 1:
            self.cur_y_pos += 1
        elif dir_choice == 2:
            self.cur_x_pos -= 1
        elif dir_choice == 3:
            self.cur_x_pos += 1
        else:
            print("I'm stuck")

        state = self.get_state()

        exit_found = False
        reward = 0
        if 2 in state:
            exit_found = True
            reward = 10

        return state, reward, exit_found

    def reset(self):
        self.cur_x_pos = self._x_entrance + 1  # assume entrance always on left wall
        self.cur_y_pos = self._y_entrance
        return self.get_state()


class DQNAgent:
    def __init__(self, state_size, action_size):
        print("Agent init")
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.5  # exploration rate
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

    def print_Q(self):
        with open('./models/init_Q.txt', 'r') as f:
            loaded_Q = [line.replace('\n', '').split(',') for line in f]

        loaded_Q = np.array(loaded_Q).astype(np.float)

        print_Q = self.target_model.predict(loaded_Q[:, 0:4].astype(np.int))
        np.set_printoptions(precision=3, suppress=True)
        print(print_Q)
        #print(init_Q[:,4:8])

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            dir_choice = random.randrange(self.action_size)
        else:
            dir_choice = np.argmax(self.model.predict(state)[0])

        wall = True
        while wall:
            if state[0][dir_choice] == 0:
                wall = False
            else:
                dir_choice = random.randrange(self.action_size)

        return dir_choice  # returns action

    def load(self, name):
        self.model.load_weights(name)
        self.target_model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    laby_env = Env("laby1")
    print("Entrance Coordinates")
    print(laby_env._x_entrance, laby_env._y_entrance)

    agent = DQNAgent(laby_env.state_size, laby_env.action_size)
    # agent.init_Q()
    agent.load("./models/model_after_Q_init.h5")

    np.set_printoptions(precision=3, suppress=True)
    print("Check Q is initialised")
    print(agent.target_model.predict(np.reshape([9, 9, 9, 0], [1, agent.state_size])))
    print(agent.target_model.predict(np.reshape([9, 0, 9, 0], [1, agent.state_size])))
    print(agent.target_model.predict(np.reshape([0, 0, 9, 0], [1, agent.state_size])))
    print(agent.target_model.predict(np.reshape([0, 0, 0, 0], [1, agent.state_size])))

    print("try some moves...")
    print(agent.act(np.reshape([9, 9, 9, 0], [1, agent.state_size])))
    print(agent.act(np.reshape([9, 0, 9, 0], [1, agent.state_size])))
    print(agent.act(np.reshape([0, 0, 9, 0], [1, agent.state_size])))
    print(agent.act(np.reshape([0, 0, 0, 0], [1, agent.state_size])))

    batch_size = 64

    for e in range(EPISODES):
        # Define the path variable. Each cell is a tuple (x_pos, y_pos)
        path = []
        done = False

        cur_state = laby_env.reset()
        cur_state = np.reshape(cur_state, [1, agent.state_size])
        if EPISODES % 100 == 0:
            print("episode: {}/{}".format(e, EPISODES))
            agent.print_Q()

        steps = 0
        while not done:
            path.append((laby_env.cur_x_pos, laby_env.cur_y_pos))
            steps += 1
            action = agent.act(cur_state)
            next_state, reward, done = laby_env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])

            agent.remember(cur_state, action, reward, next_state, done)
            cur_state = next_state

            if steps > 300:
                done = True

            if done:
                agent.update_target_model()
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        f = open("games/saved_games_dqn.txt", "a")
        f.write(str(path))
        f.write("\n")
        f.close()
