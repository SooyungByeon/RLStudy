#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
# Comments added by Sooyung Byeon
# 09/18/2019

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

matplotlib.use('Agg')

# Environment
WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]
# Discounting factor
DISCOUNT = 0.9

# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
# For the random policy
ACTION_PROB = 0.25


# Environment of the gridworld example
# Input: current state / chosen action
# Output: next state / reward
def step(state, action):
    # If the agent is on the A, take it to the A' and give reward (10)
    if state == A_POS:
        return A_PRIME_POS, 10

    # If the agent is on the B, take it to the B' and give reward (5)
    if state == B_POS:
        return B_PRIME_POS, 5

    # Calculate the next state in terms of position (x,y)
    # tolist(): Return a copy of the array data as a (nested) Python list.
    next_state = (np.array(state) + action).tolist()
    # If the agent hits the wall, stay at the same state and give reward (-1)
    # Otherwise, move the the next state and give reward (0)
    x, y = next_state
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        reward = -1.0
        next_state = state
    else:
        reward = 0
    return next_state, reward


# Image drawing function (has nothing to do with RL...)
# Input: image (numpy array)
# Output: no explicit one
def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')

    # Row and column labels...
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')

    ax.add_table(tb)


# Generating Figure 3.2
# State-value function for the equiprobable random policy
def figure_3_2():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        # keep iteration until convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # bellman equation
                    new_value[i, j] += ACTION_PROB * (reward + DISCOUNT * value[next_i, next_j])
        # if value update is slow enough (converge), break the while loop
        if np.sum(np.abs(value - new_value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            plt.savefig('../images/figure_3_2.png')
            plt.close()
            break
        # Otherwise, keep updating state-value
        value = new_value

# Generating Figure 3.5
# Optimal state-value function
def figure_3_5():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        # keep iteration until convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                values = []
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # value iteration: keep every possible value at each state
                    values.append(reward + DISCOUNT * value[next_i, next_j])
                # Choose the highest (maximum) value at each state
                new_value[i, j] = np.max(values)
        # if value update is slow enough (converge), break the while loop
        if np.sum(np.abs(new_value - value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            plt.savefig('../images/figure_3_5.png')
            plt.close()
            break
        # Otherwise, keep updating state-value
        value = new_value


if __name__ == '__main__':
    figure_3_2()
    figure_3_5()
