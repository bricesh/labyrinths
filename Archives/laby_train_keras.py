from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes).astype(int)[targets]

# fix random seed for reproducibility
np.random.seed(163)

# Load file
with open('games/saved_games_small_rdm.txt', 'r') as f:
    paths = [line.replace('\n', '').split(',') for line in f]

num_list = []

for l in range(0, len(paths)):
    num_str = [item for item in paths[l] if item.isdigit()]
    for i in range(0, len(num_str)):
        num_list.append(int(num_str[i]))

num_array = np.array(num_list).reshape(len(num_list)/7,7)

training_data = pd.DataFrame(num_array, columns=["x", "y", "up", "down", "left", "right", "direction"])
training_action = training_data[["direction"]]
training_features = training_data[["x", "y", "up", "down", "left", "right"]]
training_features["pre_x"] = training_features["x"].shift(1)
training_features["pre_y"] = training_features["y"].shift(1)
training_features.loc[training_features["x"]==0,"pre_x"]=training_features[training_features["x"]==0]["x"]
training_features.loc[training_features["x"]==0,"pre_y"]=training_features[training_features["x"]==0]["y"]

training_action["direction"].value_counts() # Could improve the class balance...

train_x, test_x, train_y, test_y = train_test_split(training_features,
                                                    training_action,
                                                    train_size=.5)

# AI
# create model
model = Sequential()
model.add(Dense(32, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='sigmoid'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(np.array(train_x), indices_to_one_hot(train_y,4), epochs=5, batch_size=200)

scores = model.evaluate(np.array(test_x), indices_to_one_hot(test_y,4))
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
