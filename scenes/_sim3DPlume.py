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
import utils

ap = argparse.ArgumentParser()

# Some arguments the user might want to set.
ap.add_argument("--numFrames", type=int, default=1536)
ap.add_argument("--timeStep", type=float, default=0.1)
ap.add_argument("--outputDecimation", type=int, default=6)
ap.add_argument("--resolution", type=int, default=128)
ap.add_argument("--loadVoxelModel", default="none")
ap.add_argument("--buoyancyScale", type=float, default=0.5)
ap.add_argument("--vorticityConfinementAmp", type=float, default=0.0)
ap.add_argument("--plumeScale", type=float, default=0.25)
ap.add_argument("--advectionOrder", type=int, default=2)
ap.add_argument("--advectionOrderSpace", type=int, default=1)

# Some Arguments the user probably should not set.
verbose = False
dim = 3

args = ap.parse_args()
print("\nUsing arguments:")
for k, v in vars(args).items():
  print("  %s: %s" % (k, v))
print("\n")

utils.bWidth = 1
baseRes = args.resolution
res = baseRes + 2 * utils.bWidth  # Note: bWidth is 0 by default.
gridSize = vec3(res, res, res)

random.seed(1945)

sm = Solver(name="main", gridSize=gridSize, dim=dim)
sm.timestep = args.timeStep

flags = sm.create(FlagGrid)
vel = sm.create(MACGrid)
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

flags.initDomain(boundaryWidth=utils.bWidth)
flags.fillGrid()
setOpenBound(flags, utils.bWidth, "xXyYzZ", FlagOutflow|FlagEmpty)


'''
plumeRad = 0.15  # Should match rad in fluid_net_3d_sim.
plumeScale = args.plumeScale
plumeUp = True
plumeFace = 'y'
#setPlumeBound(flags, density, vel, utils.bWidth, plumeRad, plumeScale, plumeUp,
#              plumeFace)
'''
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

# Do I set the outflow here where it will effect the geom file that is written
# to disk?
#setOpenBound(flags, utils.bWidth, "yY", FlagOutflow|FlagEmpty)
source = sm.create(Cylinder, center=res*vec3(0.5, 0.1, 0.4), radius=res*0.15, z=res*vec3(0, 0.05, 0))

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

  source.applyToGrid(grid=density, value=1)

  advectSemiLagrange(flags=flags, vel=vel, grid=density,
                     order=args.advectionOrder,
                     orderSpace=args.advectionOrderSpace)
  advectSemiLagrange(flags=flags, vel=vel, grid=vel, order=args.advectionOrder,
                     openBounds=True, boundaryWidth=utils.bWidth,
                     orderSpace=args.advectionOrderSpace)
 
  resetOutflow(flags=flags, real=density)
  setWallBcs(flags=flags, vel=vel)

#  setPlumeBound(flags, density, vel, utils.bWidth, plumeRad, plumeScale,
#                plumeUp, plumeFace)

  bStrength = -(sm.dx() / 4) * args.buoyancyScale
  print("  Using buoyancy strength: %f" % (bStrength))
  addBuoyancy(density=density, vel=vel, gravity=vec3(0, bStrength, 0),
              flags=flags)

  vStrength = sm.dx() * args.vorticityConfinementAmp
#  print("  Using vorticity confinement strength: %f" % (vStrength))
  vorticityConfinement(vel=vel, flags=flags, strength=vStrength) 

#  setPlumeBound(flags, density, vel, utils.bWidth, plumeRad, plumeScale,
#                plumeUp, plumeFace)
  setWallBcs(flags=flags, vel=vel)
  resetOutflow(flags=flags, real=density)

  residue = solvePressure(flags=flags, vel=vel, pressure=pressure, 
                          cgMaxIterFac=utils.cgMaxIterFac,
                          cgAccuracy=utils.cgAccuracy,
                          precondition=utils.precondition)
 
  if residue > utils.cgAccuracy * 10 or math.isnan(residue):
    raise Exception("residue is too high!")
 
  # Important, must come AFTER write to file.
  setWallBcs(flags=flags, vel=vel)
#  setPlumeBound(flags, density, vel, utils.bWidth, plumeRad, plumeScale,
#                plumeUp, plumeFace)

  resetOutflow(flags=flags, real=density)
  sm.step()

  # Write out the sim state.
  if (t % args.outputDecimation) == 0:
    print("  writing fame to files %s, %s" % (densityFilename, geomFilename))
    density.writeFloatDataToFile(densityFilename, utils.bWidth, True)
    geom.writeFloatDataToFile(geomFilename, utils.bWidth, True)

  if verbose:
    timings.display()

  curFrame += 1
