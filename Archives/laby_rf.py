from random import randint
import pickle
import numpy as np

rf_clf = pickle.load(open( "rf_clf_with_predir.p", "rb" ))
random_count = 0
decision_count = 0
saved_games = 0

# Load file and remove linefeed characters
with open('labys/laby1.txt', 'r') as f:
    laby = [line.replace('\n', '').split(',') for line in f]

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

# Play 10000 games
for game in range(0, 100000):
    if game % 100 == 0:
        print str(game), str(random_count*100/(.0001 + decision_count)), str(saved_games*100/(.0001+game))

    x_agent = x_entrance
    y_agent = y_entrance
    x_prev = x_agent
    y_prev = y_agent
    dir_prev = 3

    # Define the path variable. Each cell is a tuple (x_pos, y_pos, up_val, down_val, left_val, right_val)
    path = []

    # Add entrance cell
    path.append((x_agent, y_agent, 9, 9, 9, 0, 3))

    exit_not_found = True
    while exit_not_found:
        decision_count += 1

        dir_choice = path[-1][6]

        x_prev = x_agent
        y_prev = y_agent
        dir_prev = dir_choice

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

        dir_choice = -1
        # Look around and save what agent sees
        path.append((x_agent,
                     y_agent,
                     laby[y_agent-1][x_agent],
                     laby[y_agent+1][x_agent],
                     laby[y_agent][x_agent-1],
                     laby[y_agent][x_agent+1],
                     dir_choice))

        # Pick a step randomly from available directions
        random_decision = False
        if randint(0, 100) > 80:
            random_decision = True
            dir_choice = randint(0, 3)
        else:
            dir_choice = rf_clf.predict(np.array(path[-1][2:6] + (dir_prev,)).reshape(1, -1))[0]

        wall = True
        while wall:
            if path[-1][dir_choice + 2] == 0:
                wall = False
            else:
                dir_choice = randint(0, 3)
                random_decision = True
        if random_decision:
            random_count += 1

        path[-1] = path[-1][:6] + (dir_choice,)

        if (2 in path[-1][2:6]) or (len(path) > 202):
            exit_not_found = False

    # Write to file if number of steps at most ...
    if len(path) <= 200:
        f = open("labys/saved_games_rf.txt", "a")
        f.write(str(path))
        f.write("\n")
        f.close()
        saved_games += 1
        print("No. of steps = " + str(len(path)) + ". Game no. "+ str(game) +" added to file")
