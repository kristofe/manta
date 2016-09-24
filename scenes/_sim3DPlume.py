# Script for generating the manta plume video data.
#
# Usage:
#
#    manta ../scenes/_sim3DPlume.py --help
#
#    manta ../scenes/_sim3DPlume.py --resolution 64 --loadVoxelModel none

import argparse
import gc
from manta import *
import os, shutil, math, sys, random
from voxel_utils import VoxelUtils
import binvox_rw
import numpy as np
import struct

ap = argparse.ArgumentParser()

# Some arguments the user might want to set.
ap.add_argument("--numFrames", type=int, default=256)
ap.add_argument("--timeStep", type=float, default=0.4)
ap.add_argument("--resolution", type=int, default=128)
ap.add_argument("--loadVoxelModel", default="none")

# Some Arguments the user probably should not set.
dim = 3
bWidth = 1
emBorder = 1
cgAccuracy = 0.001
cgMaxIterFac = 30.0
verbose = False
precondition = False  # For now the preconditioner causes segfaults.

args = ap.parse_args()
print("\nUsing arguments:")
for k, v in vars(args).items():
  print("  %s: %s" % (k, v))
print("\n")

baseRes = args.resolution
res = baseRes + 2 * bWidth
gs = vec3(res, res, res)

random.seed(1945)

sm = Solver(name="main", gridSize=gs, dim=dim)
sm.timestep = args.timeStep

flags    = sm.create(FlagGrid)
vel      = sm.create(MACGrid)
pressure = sm.create(RealGrid)
density = sm.create(RealGrid)
geom = sm.create(RealGrid)

timings = Timings()

if not verbose:
  setDebugLevel(-1)  # Disable debug printing altogether.
else:
  setDebugLevel(10)  # Print like crazy!

if (GUI):
  gui = Gui()
  gui.show(True)

totalFrames = args.numFrames
curFrame = 0

flags.initDomain(boundaryWidth=bWidth)
flags.fillGrid()
assert(bWidth == 1)  # Must match training data.
setOpenBound(flags, bWidth, "xXyYzZ", FlagOutflow | FlagFluid)
plumeRad = 0.15  # Should match rad in fluid_net_3d_sim.
plumeScale = 1
plumeUp = True
plumeFace = 'y'
setPlumeBound(flags, density, vel, bWidth, plumeRad, plumeScale, plumeUp,
              plumeFace)

if args.loadVoxelModel == "none":
  outDir = "../../blender/mushroom_cloud_render/"
  # No geometry for the mushroom cloud.
elif args.loadVoxelModel == "arc":
  outDir = "../../blender/arch_render/"
  flags.loadGeomFromVboxFile(outDir + "/geom_output.vbox")
elif args.loadVoxelModel == "bunny":
  outDir = "../../blender/bunny_render/"
  flags.loadGeomFromVboxFile(outDir + "/geom_output.vbox")
else:
  raise Exception("Bad args.loadVoxelModel value")

def writeInt32(fileHandle, val):
  fileHandle.write(struct.pack('i', val))

def writeFloat(fileHandle, val):
  fileHandle.write(struct.pack('f', val))

densityFilename = outDir + "/density_output_manta.vbox"
densityFile = open(densityFilename, "wb")
writeInt32(densityFile, baseRes)
writeInt32(densityFile, baseRes)
writeInt32(densityFile, baseRes)
writeInt32(densityFile, args.numFrames)
densityFile.close()

geomFilename = outDir + "/geom_output_manta.vbox"
geomFile = open(geomFilename, "wb")
writeInt32(geomFile, baseRes)
writeInt32(geomFile, baseRes)
writeInt32(geomFile, baseRes)
writeInt32(geomFile, args.numFrames)
geomFile.close()

for t in range(args.numFrames):
  print("Simulating frame %d of %d (total)" % (curFrame + 1, totalFrames))

  advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2)
  advectSemiLagrange(flags=flags, vel=vel, grid=vel, order=2,
                     openBounds=True, boundaryWidth=bWidth)
  # Note: The plume density wont stay constant throughout the simulation. I'm
  # still not 100% positive how mantaflow BCs work, but there's no such thing
  # as a constant density inflow unit (inflow just sets constant velocity).
  setWallBcs(flags=flags, vel=vel)
  setPlumeBound(flags, density, vel, bWidth, plumeRad, plumeScale, plumeUp,
                plumeFace)
  # Torch buoyancy strength is defined as: density * dt * 0.5470 (no dx term).
  # -> it does not scale by grid size (i.e. we assume a larger grid occupies
  # more space).
  # Manta buoyancy strength is defined as: density * gravity * dt / dx.
  # -> so it does scale by grid size.
  # Assume torch dt matches manta dt (0.4) and back-calculate manta strength
  # to receive the same force.
  bStrength = 0.5470 / (1 / sm.dx())
  print("  Using buoyancy strength: %f" % (bStrength))
  addBuoyancy(density=density, vel=vel, gravity=vec3(0, -bStrength, 0),
              flags=flags)

  # Our vorticity confinement force is multiplied by:
  # -> f_vc * scale * dt
  # It's not obvious what manta does, but I DON'T think they're normalizing
  # by dt, i.e. it's just:
  # -> f_vc * scale
  # Assume we used 0.8 strength in torch (current default).
  vStrength = 0.8 * 0.4 / (1 / sm.dx())
  # Unfortunately, this should work, but it doesn't :-( We need an extra
  # fudge factor (I suspect they don't calculate curl properly).
  vStrength = vStrength * 10
  print("  Using vorticity confinement strength: %f" % (vStrength))
  vorticityConfinement(vel=vel, flags=flags, strength=vStrength) 

  residue = solvePressure(flags=flags, vel=vel, pressure=pressure, 
                          cgMaxIterFac=cgMaxIterFac,
                          cgAccuracy=cgAccuracy, precondition=precondition)
 
  if residue > cgAccuracy * 10 or math.isnan(residue):
    raise Exception("residue is too high!")
 
  # Important, must come AFTER write to file.
  setWallBcs(flags=flags, vel=vel)
  setPlumeBound(flags, density, vel, bWidth, plumeRad, plumeScale, plumeUp,
                plumeFace)

  sm.step()

  print("  writing fame to files %s, %s" % (densityFilename, geomFilename))

  # Write out the sim state.
  density.writeFloatDataToFile(densityFilename, bWidth, True)
  geom.writeFloatDataToFile(geomFilename, bWidth, True)

  if verbose:
    timings.display()

  curFrame += 1
