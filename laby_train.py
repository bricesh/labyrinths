import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

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
                                                    train_size=.4)

rf_clf = RandomForestClassifier()
rf_clf.fit(train_x, train_y)

predictions = rf_clf.predict(test_x)

print "Train Accuracy :: ", accuracy_score(train_y, rf_clf.predict(train_x))
print "Test Accuracy  :: ", accuracy_score(test_y, predictions)
print " Confusion matrix "
print confusion_matrix(test_y, predictions)

pickle.dump(rf_clf, open( "rf_clf_with_prexy.p", "wb" ))
