# Training script for creating 2D or 3D data with or without geometry.
# Usage:
#
#    manta ../scenes/_trainingData.py --help
#
#    manta ../scenes/_trainingData.py --dim 3

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
ap.add_argument("--dim", type=int, default=2)
ap.add_argument("--numFrames", type=int, default=256)
ap.add_argument("--numTraining", type=int, default=320)
ap.add_argument("--numTest", type=int, default=320)
ap.add_argument("--frameStride", type=int, default=4)
ap.add_argument("--timeStep", type=float, default=0.1)
ap.add_argument("--addModelGeometry", type=bool, default=True)
ap.add_argument("--addSphereGeometry", type=bool, default=True)
ap.add_argument("--addNoise", type=bool, default=True)
ap.add_argument("--addBuoyancy", type=bool, default=True)
ap.add_argument("--addVortConf", type=bool, default=True)
ap.add_argument("--voxelPath", type=str, default="../../voxelizer/")
ap.add_argument("--voxelNameRegex", type=str, default=".*_(16|32)\.binvox")
ap.add_argument("--voxelLayoutBoxSize", type=int, default=32)
ap.add_argument("--seed", type=int, default=1945)
ap.add_argument("--datasetNamePostfix", type=str, default="")

# Some Arguments the user probably should not set.
verbose = False
buoyancyProb = 0.1  # Don't always add buoyancy.
vortConfProb = 0.5  # Don't always add VC.

args = ap.parse_args()
print("\nUsing arguments:")
for k, v in vars(args).items():
  print("  %s: %s" % (k, v))
print("\n")

# solver params
if (args.dim == 3):
  baseRes = 64
else:
  baseRes = 128
res = baseRes + 2 * utils.bWidth
gridSize = vec3(res, res, res)
if (args.dim == 2):
  gridSize.z = 1

numSims = args.numTraining + args.numTest

random.seed(args.seed)

solver = Solver(name="main", gridSize=gridSize, dim=args.dim)
solver.timestep = args.timeStep

if (args.dim == 3):
  datasetName = "output_current_3d"
  layoutBoxSize = [args.voxelLayoutBoxSize, args.voxelLayoutBoxSize,
             args.voxelLayoutBoxSize]
else:
  assert(args.dim == 2)
  datasetName = "output_current"
  layoutBoxSize = [args.voxelLayoutBoxSize, args.voxelLayoutBoxSize]

if args.addModelGeometry:
  datasetName = datasetName + "_model"

if args.addSphereGeometry:
  datasetName = datasetName + "_sphere"

datasetName = datasetName + args.datasetNamePostfix

print("Outputting dataset '%s'" % (datasetName))

modelList = []

if args.addModelGeometry:
  print("using " + args.voxelPath + " for voxel data with pattern "
      + args.voxelNameRegex)
  modelListTrain = VoxelUtils.create_voxel_file_list(
      args.voxelPath + "/voxels_train/", args.voxelNameRegex)
  modelListTest = VoxelUtils.create_voxel_file_list(
      args.voxelPath + "/voxels_test/", args.voxelNameRegex)

flags = solver.create(FlagGrid)
vel = solver.create(MACGrid)
velTmp = solver.create(VecGrid)  # Internally a Grid<Vec3>
pressure = solver.create(RealGrid)
density = solver.create(RealGrid)

timings = Timings()

if not verbose:
  setDebugLevel(-1)  # Disable debug printing altogether.
else:
  setDebugLevel(10)  # Print like crazy!

if (GUI):
  gui = Gui()
  gui.show(True)

for simnum in range(numSims):
  print("Simulating sim %d of %d (total)" % (simnum + 1, numSims))
  trainSample = simnum < args.numTraining

  vel.clear()
  velTmp.clear()
  pressure.clear()
  density.clear()

  utils.InitDomain(flags, utils.bWidth, args.dim)

  if args.addModelGeometry:
    if(args.dim == 3):
      inputDims = [res, res, res]
      geom = binvox_rw.Voxels(np.zeros(inputDims), inputDims, [0, 0, 0],
          [1, 1, 1], "xyz", 'cur_geom')
      if trainSample:
        VoxelUtils.create_grid_layout(modelListTrain, layoutBoxSize, geom,
            args.dim)
      else:
        VoxelUtils.create_grid_layout(modelListTest, layoutBoxSize, geom,
            args.dim)

      for i in range(0, geom.dims[0]):
        for j in range(0, geom.dims[1]):
          for k in range(0, geom.dims[2]):
            if geom.data[i, j, k]:
              flags.setObstacle(i, j, k)
    else:
      assert(args.dim == 2)
      geom = np.zeros((res, res))
      if trainSample:
        VoxelUtils.create_grid_layout_2d(modelListTrain, layoutBoxSize, geom,
            args.dim)
      else:
        VoxelUtils.create_grid_layout_2d(modelListTest, layoutBoxSize, geom,
            args.dim)

      for i in range(0, geom.shape[0]):
        for j in range(0, geom.shape[1]):
            if geom[i, j]:
              flags.setObstacle(i, j, 0)


  if args.addSphereGeometry:
    spheres = utils.AddSphereGeom(args.dim, flags, args.addModelGeometry,
                                  gridSize, solver, FlagObstacle)

  # Noise field used to initialize velocity fields.
  noise = None
  gc.collect()
  if args.addNoise:
    noise = utils.CreateNoiseField(solver)

  utils.CreateRandomDensity(density)

  # Sometimes add buoyancy.
  addBuoy = False
  if args.addBuoyancy:
    addBuoy = random.uniform(0, 1) < buoyancyProb or (simnum == 0)
    if addBuoy:
      gravity = utils.CreateRandomGravity(solver, args.dim)
      print('  buoyancy gravity = (%f, %f, %f)' % \
          (gravity.x, gravity.y, gravity.z))
  print('  addBuoy = %r' % (addBuoy))

  # Sometimes add vorticity confinement.
  addVortConf = False
  if args.addVortConf:
    addVortConf = random.uniform(0, 1) < vortConfProb or (simnum == 0)
  print('  addVortConf = %r' % (addVortConf))

  # Random emitters.
  emitters = []
  if args.dim == 2:
    numEmitters = int(random.uniform(10, 20))
  else:
    # Again curse of dimensionality means we probably need more emitters.
    numEmitters = int(random.uniform(32, 64))

  emitterAmps = []
  globalEmitterAmp = random.uniform(0.1, 1)
  for e in range(0, numEmitters):
    # NOTE: to test out all these emitter properties you can use
    # scenes/EmitterTest to see a visualization.

    # eRad: controls the size of the emitter.
    eRad = random.randint(1, 3)

    # eVel: speed of movement around the domain.
    eVel = 10 ** random.uniform(0.5, 0.8)

    # eAmp: the amplitude of the force applied (in the same direction as vel).
    eAmp = 10 ** random.uniform(-0.3, 0.3) * globalEmitterAmp 

    # eCurvature and eCurveTScale define the amount of movement in the
    # particle's path through the simulation.
    eCurvature = 10 ** random.uniform(-1, 1)
    eCurveTScale = 10 ** random.uniform(-1, 1)

    # Create the emitter.
    emitters.append(ForceEmitter(eRad, eVel, eAmp, args.dim == 3, res, res,
                                 res, eCurvature, eCurveTScale, utils.bWidth))
    emitterAmps.append(eAmp)
  print('  Num Emitters: ' + str(len(emitters)))
  print('  Global emitter amp: ' + str(globalEmitterAmp))
  print('  Emitter min amp: ' + str(min(emitterAmps)))
  print('  Emitter max amp: ' + str(max(emitterAmps)))

  residue = 0
  offset = 0  # You can use this to add additional frames.
  for t in range(args.numFrames):
    if trainSample:
      directory = "../../data/datasets/%s/tr/%06d" % (datasetName, simnum +
                                                      offset)
    else:
      directory = "../../data/datasets/%s/te/%06d" % \
          (datasetName, simnum - args.numTraining + offset)

    if (t + 1 != args.numFrames):
      sys.stdout.write('  Simulating %d of %d\r' % (t + 1, args.numFrames))
      sys.stdout.flush()
    else:
      print('  Simulating %d of %d' % (t + 1, args.numFrames))  # keep \n char

    if not os.path.exists(directory):
      os.makedirs(directory)

    if(t == 0):
      residue = utils.InitSim(flags, vel, velTmp, noise, density, pressure,
                              utils.bWidth, utils.cgAccuracy,
                              utils.precondition, utils.cgMaxIterFac)
      if math.isnan(residue):
        # Try again but with the preconditioner off.
        residue = utils.InitSim(flags, vel, velTmp, noise, density, pressure,
                                utils.bWidth, utils.cgAccuracy,
                                False, utils.cgMaxIterFac)
      if residue > utils.cgAccuracy * 10 or math.isnan(residue):
        print("WARNING: Residue (%f) has blown up before starting sim)" % \
            (residue))
        print("--> Starting a new simulation")
        break

    # We don't have to advect density, but it helps debug the plume.
    advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2,
                       boundaryWidth=utils.bWidth)
    advectSemiLagrange(flags=flags, vel=vel, grid=vel, order=2,
                       openBounds=True, boundaryWidth=utils.bWidth)

    setWallBcs(flags=flags, vel=vel)

    for em in emitters:
      em.update(solver.timestep, solver.timestep * t)
      em.addVelocities(vel, flags, utils.bWidth)

    setWallBcs(flags=flags, vel=vel)

    if addBuoy:
      addBuoyancy(density=density, vel=vel, gravity=gravity, flags=flags)

    if addVortConf:
      vStrength = solver.dx()
      vorticityConfinement(vel=vel, flags=flags, strength=vStrength)

    if t % args.frameStride == 0:
      filename = "%06d_divergent.bin" % t
      fullFilename = directory + "/" + filename 
      writeOutSim(fullFilename, vel, pressure, density, flags)

    setWallBcs(flags=flags, vel=vel)  # This will be "in" the model.
    residue = solvePressure(flags=flags, vel=vel, pressure=pressure, 
                            cgMaxIterFac=utils.cgMaxIterFac,
                            cgAccuracy=utils.cgAccuracy,
                            precondition=utils.precondition)
   
    if math.isnan(residue):
      # try again but with the preconditioner off.
      residue = solvePressure(flags=flags, vel=vel, pressure=pressure,
                              cgMaxIterFac=utils.cgMaxIterFac,
                              cgAccuracy=utils.cgAccuracy,
                              precondition=False)

    if residue < utils.cgAccuracy * 10 or not math.isnan(residue):
      setWallBcs(flags=flags, vel=vel)  # This will be "in" the model.

    else:
      # If we hit maxIter, than residue will be higher than our
      # specified accuracy.  This is OK, but we shouldn't let it grow too
      # much.
      #
      # If it does grow (it happens 1 in every ~10k frames), then the best
      # thing to do is just start a new simulation rather than crashing out
      # completely. This ends up being faster than letting the solver
      # arbitrarily increase the number of iterations.
      #
      # Admittedly this is a hack. These high residue frames could be (and
      # probably are) highly correlated with ConvNet failure frames and so are
      # likely examples that we should be trying to incorporate. Unfortunately,
      # they're rare, and they result in significantly longer simulation
      # times when the solver bumps up against the max iter, that it's just
      # not worth it to include them.
      print("WARNING: Residue (%f) has blown up on frame %d" % (residue, t))
      print("--> Starting a new simulation")

      # Remove the last recorded divergent frame.
      if t % args.frameStride == 0:
        filename = "%06d_divergent.bin" % t
        fullFilename = directory + "/" + filename
        os.remove(fullFilename) 

      # Break out of frame loop (and start a new sim).
      break
   
    if t % args.frameStride == 0:
      filename = "%06d.bin" % t
      fullFilename = directory + "/" + filename
      writeOutSim(fullFilename, vel, pressure, density, flags) 

    solver.step()
    
    if verbose:
      timings.display()

