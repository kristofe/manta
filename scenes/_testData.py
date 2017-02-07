# Script to create testing data to validate tfluids routines. Not the script
# to create the test set (this is done in _trainingData.py).
#
# Usage:
#
#    manta ../scenes/_testData.py
#
#    manta ../scenes/_testData.py --help

import argparse
import gc
from manta import *
import os, shutil, math, sys, random
from Emitter import *
from voxel_utils import VoxelUtils
import utils
import binvox_rw
import numpy as np

ap = argparse.ArgumentParser()

# Some arguments the user might want to set.
ap.add_argument("--timeStep", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=1944)
ap.add_argument("--directory", type=str,
                default='../../torch/tfluids/test_data')
ap.add_argument("--nbatch", type=int, default=16)

args = ap.parse_args()
print("\nUsing arguments:")
for k, v in vars(args).items():
  print("  %s: %s" % (k, v))
print("\n")

# Note: We test not only 2D and 3D but also a batched input. Therefore, we need
# to dump many frames.

random.seed(args.seed)

for dim in [2, 3]:
  for batch in range(0, args.nbatch):
    gc.collect()

    # solver params
    gridSize = vec3(16 + 2 * utils.bWidth,
                    17 + 2 * utils.bWidth,
                    9 + 2 * utils.bWidth)  # Make dims not equal.
    if (dim == 2):
      gridSize.z = 1
    
    solver = Solver(name="main", gridSize=gridSize, dim=dim)
    solver.timestep = args.timeStep
    
    if (dim == 3):
      filenamePrefix = "b%d_3d_" % (batch)
    else:
      assert(dim == 2)
      filenamePrefix = "b%d_2d_" % (batch)
    
    def SaveData(name, vel, pressure, density, flags):
      filename = filenamePrefix + name
      path = args.directory + "/" + filename
      writeOutSim(path, vel, pressure, density, flags)
      print('Saved data: ' + path)

    flags = solver.create(FlagGrid)
    vel = solver.create(MACGrid)
    velTmp = solver.create(VecGrid)  # Internally a Grid<Vec3>
    velDiv = solver.create(MACGrid)
    pressure = solver.create(RealGrid)
    density = solver.create(RealGrid)
    densityTmp = solver.create(RealGrid)
    rhs = solver.create(RealGrid)
    
    setDebugLevel(1)
    
    vel.clear()
    velTmp.clear()
    velDiv.clear()
    pressure.clear()
    density.clear()
    
    utils.InitDomain(flags, utils.bWidth, dim)
    
    spheres = utils.AddSphereGeom(dim, flags, False, gridSize, solver,
                                  FlagObstacle)
    
    # Noise field used to initialize velocity fields.
    noise = None
    gc.collect()
    noise = utils.CreateNoiseField(solver)
    
    utils.CreateRandomDensity(density)
    
    # Use a gravity that involves all coordinates but with the same magnitude
    # that we use during training.
    gStrength = solver.dx() / 4
    gravity = vec3(1, 2, 3)
    if dim == 2:
      gravity.z = 0
    gravity = Vec3Utils.normalizeVec3(gravity)
    gravity.x = gravity.x * gStrength
    gravity.y = gravity.y * gStrength
    gravity.z = gravity.z * gStrength
    
    # Note: we wont include emitters or geometry in the test data. Sphere
    # data is good enough to handle all cases.

    residue = utils.InitSim(flags, vel, velTmp, noise, density, pressure,
                            utils.bWidth, utils.cgAccuracy / 100,
                            utils.precondition, utils.cgMaxIterFac * 100)

    if math.isnan(residue):
      # Try again but with the preconditioner off.
      residue = utils.InitSim(flags, vel, velTmp, noise, density, pressure,
                              utils.bWidth, utils.cgAccuracy / 100,
                              False, utils.cgMaxIterFac * 100)

    # We need low residue for the test data!
    assert(residue <= utils.cgAccuracy and not math.isnan(residue))
 
    # Save the initial simulation data.
    SaveData("initial.bin", vel, pressure, density, flags)

    # We don't have to advect density, but it helps debug the plume.
    for openBounds in [False]:
      for order in [1, 2]:
        velDiv.copyFrom(vel)
        densityTmp.copyFrom(density)
        advectSemiLagrange(flags=flags, vel=velDiv, grid=densityTmp,
                           order=order, boundaryWidth=utils.bWidth)
        advectSemiLagrange(flags=flags, vel=velDiv, grid=velDiv, order=order,
                           openBounds=openBounds, boundaryWidth=utils.bWidth)
        SaveData("advect_openBounds_%r_order_%d.bin" % (openBounds, order),
                 velDiv, pressure, densityTmp, flags)

    # Perform the standard advection (to continue the simulation).
    advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2,
                       boundaryWidth=utils.bWidth)
    advectSemiLagrange(flags=flags, vel=vel, grid=vel, order=2,
                       openBounds=True, boundaryWidth=utils.bWidth)
    SaveData("advect.bin", vel, pressure, density, flags)

    setWallBcs(flags=flags, vel=vel)
    SaveData("setWallBcs1.bin", vel, pressure, density, flags)
   
    addBuoyancy(density=density, vel=vel, gravity=gravity, flags=flags)
    SaveData("buoyancy.bin", vel, pressure, density, flags)
    
    vStrength = solver.dx()
    vorticityConfinement(vel=vel, flags=flags, strength=vStrength)
    SaveData("vorticityConfinement.bin", vel, pressure, density, flags)
 
    velDiv.copyFrom(vel)  # Make a copy of the velocity.
    makeRhs(flags=flags, vel=velDiv, rhs=rhs)
    SaveData("makeRhs.bin", velDiv, rhs, density, flags)

    setWallBcs(flags=flags, vel=vel)
    SaveData("setWallBcs2.bin", vel, pressure, density, flags)

    residue = solvePressure(flags=flags, vel=vel, pressure=pressure, 
                            cgMaxIterFac=utils.cgMaxIterFac * 100,
                            cgAccuracy=utils.cgAccuracy / 100,
                            precondition=utils.precondition)
    if math.isnan(residue):
      # Try again but with the preconditioner off.
      residue = solvePressure(flags=flags, vel=vel, pressure=pressure,
                              cgMaxIterFac=utils.cgMaxIterFac * 100,
                              cgAccuracy=utils.cgAccuracy / 100,
                              precondition=False)

    assert(residue <= utils.cgAccuracy and not math.isnan(residue))
    SaveData("solvePressure.bin", vel, pressure, rhs, flags)

    # Use the pressure from the solution above to correct the divergence.
    # Note: solvePressure has already called this on vel (in-place). This is why
    # we saved the divergent velocity.
    correctVelocity(flags=flags, vel=velDiv, pressure=pressure)
    SaveData("correctVelocity.bin", velDiv, pressure, density, flags)

    # Important, must come AFTER write to file.
    setWallBcs(flags=flags, vel=vel)
    SaveData("setWallBcs3.bin", vel, pressure, density, flags)
    
    solver.step()
    
