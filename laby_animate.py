import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy

# Load labyrinth file and remove linefeed characters
with open('maps/laby1.txt', 'r') as f:
    laby = [line.replace('\n', '').split(',') for line in f]

# Set values to int
for i in range(0, len(laby)):
    for j in range(0, len(laby)):
        laby[i][j] = int(laby[i][j])

# Load game file
with open('games/shortest_game.txt', 'r') as f:
    paths = [line.replace('\n', '').split(',') for line in f]

num_list = []

for l in range(0, len(paths)):
    num_str = [item for item in paths[l] if item.isdigit()]
    for i in range(0, len(num_str)):
        num_list.append(int(num_str[i]))

num_array = np.array(num_list).reshape(len(num_list)/7,7)

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
pos = 0


def animate(interval):
    if pos < len(num_array):
        global pos
        pos += 1
    temp_laby = copy.deepcopy(laby)
    temp_laby[num_array[pos][1]][num_array[pos][0]] = 5
    ax1.clear()
    ax1.imshow(255 - np.uint8(np.matrix(temp_laby) * 255 / 9.))


ani = animation.FuncAnimation(fig, animate, interval=200)
plt.show()
