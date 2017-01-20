import sys
if sys.version_info[0] < 3:
  print('Running with python version:')
  print(sys.version)
  print('WARNING: You will probably need to run with version 3')

import matplotlib.pyplot as plt
import numpy as np

from Emitter import *

# Make an Emitter instance.
dim = 2
bWidth = 1
res = 128 + 2 * bWidth
emBorder = 1
timestep = 0.0333333
numEmitters = 20
numSteps = 256

# Create a bunch of random emitters.
emitters = []
for e in range(0, numEmitters):
  # NOTE: to test out all these emitter properties you can use
  # scenes/EmitterTest to see a visualization.

  # eRad: controls the size of the emitter.
  eRad = random.randint(1, 3)

  # eVel: speed of movement around the domain.
  eVel = 10 ** random.uniform(0.5, 1.2)  # roughly [0.5, 10]

  # eAmp: the amplitude of the force applied (in the same direction as vel).
  eAmp = 10 ** random.uniform(-0.3, 0.3)  # roughly [0.5, 2]

  # eCurvature and eCurvatureTScale define the amount of movement in the
  # particle's path through the simulation.
  eCurvature = 10 ** random.uniform(-1, 1)
  eCurvatureTScale = 10 ** random.uniform(-1, 1)

  # Create the emitter.
  emitters.append(ForceEmitter(eRad, eVel, eAmp, dim == 3, res, res,
                               res, eCurvature, eCurvatureTScale))

# Move them around the grid.
pos = np.ndarray(shape=(numEmitters, numSteps, 3), dtype='float32')
for t in range(0, numSteps):
  for e in range(0, numEmitters):
    pos[e, t, :] = emitters[e].getPosition()
    emitters[e].update(timestep, timestep * t)

# Plot the positions.
ax = plt.figure()
textStride = int(numSteps / 10)
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
for e in range(0, numEmitters):
  plt.plot(pos[e, :, 0], pos[e, :, 1], '-', color=colors[e % len(colors)],
           linewidth=1.5)
  plt.plot(pos[e, ::textStride, 0], pos[e, ::textStride, 1], '*',
           color=colors[e % len(colors)], linewidth=1.5)
  for i in range(0, numSteps, textStride):
    plt.text(pos[e, i, 0], pos[e, i, 1], '%d' % i, fontsize=8)
plt.grid(True)
plt.show()
