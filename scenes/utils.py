# A collection of utils to share code between _trainingData.py and _testData.py

import gc
from manta import *
import os, shutil, math, sys, random
#from Emitter import *
from voxel_utils import VoxelUtils
import utils
import binvox_rw
import numpy as np

# Some Arguments the user probably should not set.
bWidth = 0
cgAccuracy = 1e-3
cgMaxIterFac = 30.0
precondition = True

def InitDomain(flags, bWidth, dim):
  if bWidth > 0:
    flags.initDomain(boundaryWidth=bWidth)
    flags.fillGrid()
    if bWidth > 0:
      if (dim == 2):
        openBoundStr = "xXyY"
      else:
        openBoundStr = "xXyYzZ"
      setOpenBound(flags, bWidth, openBoundStr, FlagOutflow | FlagFluid)
  else:
    flags.initDomain()
    flags.fillGrid()

def AddSphereGeom(dim, flags, addModelGeometry, gridSize, solver,
                  FlagObstacle):
  spheres = []
  if dim == 2:
    if addModelGeometry:
      numSpheres = int(random.uniform(4, 8))
    else:
      numSpheres = int(random.uniform(8, 16))
  else:
    if addModelGeometry:
      # curse of dimensionality means you need more spheres in 3D.
      numSpheres = int(random.uniform(8, 16))
    else:
      numSpheres = int(random.uniform(16, 32))
  for sid in range(0, numSpheres):
    center = gridSize * vec3(random.uniform(0.1, 0.9), random.uniform(0.1, 0.9),
                             random.uniform(0.1, 0.9))
    radius = gridSize.x * (10 ** random.uniform(-1.39, -0.9))
    sphere = solver.create(Sphere, center=center, radius=radius)
    spheres.append(sphere)
    sphere.applyToGrid(grid=flags, value=FlagObstacle)
  return spheres

def CreateNoiseField(solver):
  # Noise field used to initialize velocity fields.
  fixedSeed = random.randint(0, 512)  # no idea what range this should be in.
  noise = solver.create(NoiseField, fixedSeed=fixedSeed, loadFromFile=True)
  # TODO(tompson): Make this grid size independent.
  noise.posScale = vec3(random.randint(12, 24))
  noise.clamp = True
  noise.clampNeg = 0
  noise.clampPos = 2
  noise.valScale = .25 * (10 ** random.uniform(-0.5, 0.5))
  noise.valOffset = 0.075
  noise.timeAnim = 0.0
  print('  Noise valScale: %f ' % (noise.valScale))
  print('  Noise posScale: %d ' % (noise.posScale.x))

  return noise

def CreateRandomDensity(density):
  # Fills with [0, 1] iid noise.
  density.random(0, 1, random.randint(0, math.pow(2, 31) - 1))

def InitSim(flags, vel, velTmp, noise, density, pressure, bWdith,
            cgAccuracy, precondition, cgMaxIterFac):
  if noise is not None:
    # Note: applyNoiseVec3 DOES NOT handle MAC grids properly, it does
    # not create a divergence free MAC grid (because it does not
    # sample the wavelet noise at 0.5pix offset). More importantly, it
    # does not even fill in the last grid cell with noise (i.e. the
    # dimx + 1 entry for the vel_x component will be garbage).
    # THEREFORE: we need to create a Vec3 noise field and interpolate to
    # a MAC grid.
    scale = random.uniform(2.5, 5.0)
    applyNoiseVec3(flags=flags, target=velTmp, noise=noise,
                   scale=scale)
    getMACFromVec3(target=vel, source=velTmp)

  setWallBcs(flags=flags, vel=vel)
  residue = solvePressure(flags=flags, vel=vel, pressure=pressure,
                          cgMaxIterFac=cgMaxIterFac,
                          cgAccuracy=cgAccuracy, precondition=precondition)
  if residue <= cgAccuracy * 10 and not math.isnan(residue):
    setWallBcs(flags=flags, vel=vel)
  return residue

def CreateRandomGravity(solver, dim):
  # Pick a random direction.
  gravity = vec3(0, 0, 0)
  axis = random.randint(0, dim - 1)
  bStrength = solver.dx() / 4
  if random.uniform(0, 1) < 0.5:
    if axis == 0:
      gravity.x = bStrength
    elif axis == 1:
      gravity.y = bStrength
    elif axis == 2:
      gravity.z = bStrength
    else:
      raise Exception("bad axis value")
  else:
    if axis == 0:
      gravity.x = bStrength
    elif axis == 1:
      gravity.y = bStrength
    elif axis == 2:
      gravity.z = bStrength
    else:
      raise Exception("bad axis value")
  return gravity

