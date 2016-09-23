# Training script for creating 2D or 3D data with or without geometry.
# Usage:
#
#    manta ../scenes/_trainingData.py --help
#
#    manta ../scenes/_trainingData.py --dim 3 --addGeometryType "model"

import argparse
import gc
from manta import *
import os, shutil, math, sys, random
from Emitter import *
from voxel_utils import VoxelUtils
import binvox_rw
import numpy as np

ap = argparse.ArgumentParser()

# Some arguments the user might want to set.
ap.add_argument("--dim", type=int, default=2)
ap.add_argument("--numFrames", type=int, default=256)
ap.add_argument("--numTraining", type=int, default=80)
ap.add_argument("--numTest", type=int, default=80)
ap.add_argument("--frameStride", type=int, default=4)
ap.add_argument("--timeStep", type=float, default=0.033333)
ap.add_argument("--addGeometryType", type=str, default="model")  #"model|sphere|none"
ap.add_argument("--addNoise", type=bool, default=True)
ap.add_argument("--voxelPath", type=str, default="../../voxelizer/")
ap.add_argument("--voxelNameRegex", type=str, default=".*_(16|32)\.binvox")
ap.add_argument("--voxelLayoutBoxSize", type=int, default=32)

# Some Arguments the user probably should not set.
# TODO(tompson,kris): This is still messed up. A 1 here still has a border of 2.
bWidth = 1
emBorder = 1
cgAccuracy = 0.00001
cgMaxIterFac = 30.0
verbose = False
precondition = False  # For now the preconditioner causes segfaults.

args = ap.parse_args()
print("\nUsing arguments:")
for k, v in vars(args).items():
  print("  %s: %s" % (k, v))
print("\n")

# solver params
if (args.dim == 3):
  res = 64 + 2 * bWidth
else:
  res = 128 + 2 * bWidth
gs = vec3(res, res, res)
if (args.dim == 2):
  gs.z = 1

numSims = args.numTraining + args.numTest

random.seed(1945)

sm = Solver(name="main", gridSize=gs, dim=args.dim)
sm.timestep = args.timeStep

if (args.dim == 3):
  datasetName = "output_current_3d"
  layoutBoxSize = [args.voxelLayoutBoxSize, args.voxelLayoutBoxSize,
             args.voxelLayoutBoxSize]
else:
  assert(args.dim == 2)
  datasetName = "output_current"
  layoutBoxSize = [args.voxelLayoutBoxSize, args.voxelLayoutBoxSize]

if args.addGeometryType == "model":
  datasetName = datasetName + "_geom"
elif args.addGeometryType == "spheres":
  datasetName = datasetName + "_spheres"
elif args.addGeometryType != "none":
  raise Exception("BadParams", "addGeometryType is not model, geom or none")
print("Outputting dataset '%s'" % (datasetName))

modelList = []

if (args.addGeometryType == "model"):
  print("using " + args.voxelPath + " for voxel data with pattern "
      + args.voxelNameRegex)
  modelListTrain = VoxelUtils.create_voxel_file_list(
      args.voxelPath + "/voxels_train/", args.voxelNameRegex)
  modelListTest = VoxelUtils.create_voxel_file_list(
      args.voxelPath + "/voxels_test/", args.voxelNameRegex)

flags    = sm.create(FlagGrid)
vel      = sm.create(MACGrid)
velTmp   = sm.create(VecGrid)  # Internally a Grid<Vec3>
pressure = sm.create(RealGrid)
# energy   = sm.create(RealGrid)

timings = Timings()

if not verbose:
  setDebugLevel(-1)  # Disable debug printing altogether.
else:
  setDebugLevel(10)  # Print like crazy!

if (GUI):
  gui = Gui()
  gui.show(True)

totalFrames = numSims * args.numFrames
curFrame = 0

for simnum in range(numSims):
  trainSample = simnum < args.numTraining

  vel.clear()
  velTmp.clear()
  pressure.clear()
  # energy.clear()

  flags.initDomain(boundaryWidth=bWidth)
  flags.fillGrid()
  if bWidth > 0:
    if (args.dim == 2):
      openBoundStr = "xXyY"
    else:
      openBoundStr = "xXyYzZ"
    setOpenBound(flags, bWidth, openBoundStr, FlagOutflow | FlagFluid)

  if args.addGeometryType == "model":
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


  elif args.addGeometryType == "sphere":
    spheres = []
    numSpheres = int(random.uniform(8, 20))
    for sid in range(0, numSpheres):
      center = gs * vec3(random.uniform(0, 1), random.uniform(0, 1),
                         random.uniform(0, 1))
      radius = res * random.uniform(0.02, 0.06)
      sphere = sm.create(Sphere, center=center, radius=radius)
      spheres.append(sphere)
      sphere.applyToGrid(grid=flags, value=FlagObstacle)

  # Noise field used to initialize velocity fields.
  if args.addNoise:
    fixedSeed = random.randint(0, 512)  # no idea what range this should be in.
    noise = None
    gc.collect()
    noise = sm.create(NoiseField, fixedSeed=fixedSeed, loadFromFile=True)
    # TODO(tompson): Make this grid size independent.
    noise.posScale = vec3(random.randint(12, 24))
    noise.clamp = True
    noise.clampNeg = 0
    noise.clampPos = 2
    noise.valScale = .25
    noise.valOffset = 0.075
    noise.timeAnim = 0.0

  # Random emitters.
  emitters = []
  numEmitters = int(random.uniform(1, 6))
  for e in range(0, numEmitters):
    emitters.append(createRandomForceEmitter(args.dim, emBorder, res))

  for t in range(args.numFrames):
    if curFrame % 16 == 0:
      print("Simulating frame %d of %d (total)" % (curFrame + 1,
                                                   totalFrames))
    if trainSample:
      directory = "../../data/datasets/%s/tr/%06d" % (datasetName, simnum)
    else:
      directory = "../../data/datasets/%s/te/%06d" % \
          (datasetName, simnum - args.numTraining)

    if not os.path.exists(directory):
      os.makedirs(directory)
    curt = t * sm.timestep

    if(t == 0):
      if args.addNoise:
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
      if residue > cgAccuracy * 10 or math.isnan(residue):
        print("ERROR: Residue (%f) has blown up" % (residue))
      setWallBcs(flags=flags, vel=vel)

    advectSemiLagrange(flags=flags, vel=vel, grid=vel, order=2,
                       openBounds=True, boundaryWidth=bWidth)
    setWallBcs(flags=flags, vel=vel)    

    # You have to compute the energy explicitly. It is usually used for
    # adding turbulence.
    # computeEnergy(flags=flags, vel=vel, energy=energy)  # Causes SEGFAULT!
   
    for em in emitters:
      em.update(sm.timestep, sm.timestep * t)
      em.addVelocities(vel, flags, bWidth)
    setWallBcs(flags=flags, vel=vel)
  
    #addBuoyancy(density=density, vel=vel, gravity=vec3(0,-1e-3,0), flags=flags)
    # if random.random() > 0.5:
    #   # With prob 0.5 add in vorticity confinement (we want the network to see
    #   # with and without it).
    #   vorticityConfinement(vel=vel, flags=flags, strength=0.3)

    if t % args.frameStride == 0:
      filename = "%06d_divergent.bin" % t
      fullFilename = directory + "/" + filename 
      writeOutSim(fullFilename,t,vel, pressure, flags)

    residue = solvePressure(flags=flags, vel=vel, pressure=pressure, 
                            cgMaxIterFac=cgMaxIterFac,
                            cgAccuracy=cgAccuracy, precondition=precondition)
   
    if t % args.frameStride == 0:
      filename = "%06d.bin" % t
      fullFilename = directory + "/" + filename
      writeOutSim(fullFilename,t,vel, pressure, flags) 

    if residue > cgAccuracy * 10 or math.isnan(residue):
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
      # they're so rare, and they result in significantly longer simulation
      # times when the solver bumps up against the max iter, that it's just
      # not worth it to include them.
      #
      # TODO(tompson): Ideally we should analyze the solver stats, and try
      # and fix it so that we can include these frames.
      print("WARNING: Residue (%f) has blown up on frame %d" % (residue, t))
      print("--> Starting a new simulation")
      break
   
    # Important, must come AFTER write to file.
    setWallBcs(flags=flags, vel=vel)
 
    sm.step()
    
    if verbose:
      timings.display()

    curFrame += 1
