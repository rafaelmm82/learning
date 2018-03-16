"""
Application of initial functions, packages and libraries associated with
datascience using python, course:

"Intermediate Python for Data Science"

This project concerns on simulate a random walk through the floors stages on
Empire State Building, by rolling a virtual dice

@codedby: rafael magalhaes (github @rafaelmm82)
"""

# Importing the essentials
import numpy as np
import matplotlib.pyplot as plt

# setting the initial seed to random numbers
np.random.seed(123)

all_walks = []

# Simulate random walk 500 times
for i in range(500):
    random_walk = [0]

    # at each walk trhow 100 times
    for x in range(100):
        step = random_walk[-1]
        dice = np.random.randint(1, 7)

        # to go down once
        if dice <= 2:
            step = max(0, step - 1)

        # to go up once
        elif dice <= 5:
            step = step + 1

        # to go up ramdomly
        else:
            step = step + np.random.randint(1, 7)

        # chances to fallin down to zero
        if np.random.rand() <= 0.001:
            step = 0
        random_walk.append(step)

    # all the simulated walks
    all_walks.append(random_walk)


# Create and plot all walks
np_aw_t = np.transpose(np.array(all_walks))
plt.plot(np_aw_t)
plt.show()

# Select last row from np_aw_t: ends
ends = np_aw_t[-1, :]

# Plot histogram of ends, display plot
plt.clf()
plt.hist(ends)
plt.show()

# probability to be at least at 60th floor
num_60_floor = len(ends[ends >= 60])
probability_60_or_greater = (num_60_floor / len(ends)) * 100
print(probability_60_or_greater)
