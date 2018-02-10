import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy

sess = tf.Session()
K.set_session(sess)

# Load file
with open('labys/saved_games_small_rdm.txt', 'r') as f:
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
                                                    train_size=.7)

# AI
state = tf.placeholder(tf.int, shape=(None, 8))
labels = tf.placeholder(tf.int, shape=(None, 4))

# Keras layers can be called on TensorFlow tensors:
x = Dense(512, activation='relu')(state)  # fully-connected layer with 512 units and ReLU activation
x = Dense(64, activation='relu')(x)  # fully-connected layer with 64 units and ReLU activation
preds = Dense(4, activation='softmax')(x)  # output layer with 4 units and a softmax activation

loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

range_start = 0
range_width = 120
# Run training loop
with sess.as_default():
    for i in range(500):
        range_start = i * range_width
        train_step.run(feed_dict={state: train_x[range_start:range_start+range_width],
                                  labels: train_y[range_start:range_start+range_width]})

acc_value = accuracy(labels, preds)
with sess.as_default():
    print acc_value.eval(feed_dict={state: test_x,
                                    labels: test_y})
