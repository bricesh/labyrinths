from random import randint
import pickle
import numpy as np

rf_clf = pickle.load(open( "rf_clf.p", "rb" ))
random_count = 0
decision_count = 0

# Load file and remove linefeed characters
with open('labys/laby4.txt', 'r') as f:
    laby = [line.replace('\n', '').split(',') for line in f]

# Remove linefeed characters
#for i in range(0, len(laby)):
#    laby[i][19] = laby[i][19].replace('\n', '')

# Set values to int
for i in range(0, len(laby)):
    for j in range(0, len(laby)):
        laby[i][j] = int(laby[i][j])

# Find position of entrance (assumptions: cell value = 1 and only one entrance)
for i in range(0, len(laby)):
    try:
        x_entrance = laby[i].index(1)
        y_entrance = i
    except ValueError:
        continue

# Play 100000 games
for game in range(0, 400):
    x_agent = x_entrance
    y_agent = y_entrance

    # Define the path variable. Each cell is a tuple (x_pos, y_pos, up_val, down_val, left_val, right_val)
    path = []

    # Add entrance cell
    path.append((x_agent, y_agent, 9, 9, 9, 0, 3))

    exit_not_found = True
    while exit_not_found:
        decision_count += 1
        # Pick a step randomly from available directions
        random_decision = False
        dir_choice = rf_clf.predict(np.array(np.array(path[-1][0:6])).reshape(1, -1))[0]
        wall = True
        while wall:
            if path[-1][dir_choice+2]==0:
                wall = False
            else:
                dir_choice = randint(0, 3)
                random_decision = True

        if random_decision:
            random_count += 1

        if dir_choice==0:
            y_agent -= 1
        elif dir_choice==1:
            y_agent += 1
        elif dir_choice==2:
            x_agent -= 1
        elif dir_choice==3:
            x_agent += 1
        else:
            print("I'm stuck")

        #print(x_agent,y_agent,len(path))

        # Look around and save what agent sees
        path.append((x_agent,
                     y_agent,
                     laby[y_agent-1][x_agent],
                     laby[y_agent+1][x_agent],
                     laby[y_agent][x_agent-1],
                     laby[y_agent][x_agent+1],
                     dir_choice))

        if (2 in path[-1][2:6]) or (len(path) > 402):
            exit_not_found = False

    # Write to file if number of steps at most ...
    if len(path) <= 400:
        f = open("labys/saved_games_rf.txt", "a")
        f.write(str(path))
        f.write("\n")
        f.close()
        print("No. of steps = " + str(len(path)) + ". Game no. "+ str(game) +" added to file")