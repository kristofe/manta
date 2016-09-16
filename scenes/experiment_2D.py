#
# Simple example scene for a 2D simulation
# Simulation of a buoyant smoke density plume with open boundaries at top & bottom
#
from manta import *
import os
import math
from Emitter import *

# solver params
res = 128
gs = vec3(res,res,1)
s = Solver(name='main', gridSize = gs, dim=2)
s.timestep = 0.033
timings = Timings()

# prepare grids
flags = s.create(FlagGrid)
vel = s.create(MACGrid)
density = s.create(RealGrid)
pressure = s.create(RealGrid)

flags.initDomain()
flags.fillGrid()

setOpenBound(flags, 1,'yY',FlagOutflow|FlagEmpty)

if (GUI):
	gui = Gui()
	gui.show( True )
	gui.pause()

source = s.create(Cylinder, center=gs*vec3(0.5,0.1,0.5), radius=res*0.14, z=gs*vec3(0, 0.02, 0))

directory = "output/"



for simnum in range(5):
	#Emitter(radius, maxVelocity, maxForce, density, duration, variance, is3D)
	em = SmokeEmitter(5.0, 200.0, 0.1, 1.0, 100.0, 0.0, False)
	em.setBounds(vec3(0,0,0), vec3(res - 1, res -1, 0))
	em.activate()
#main loop
	for t in range(100):
		if t<100:
			source.applyToGrid(grid=density, value=1)

		advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2)
		advectSemiLagrange(flags=flags, vel=vel, grid=vel,     order=2, openBounds=True, boundaryWidth=1)
		resetOutflow(flags=flags,real=density)

		setWallBcs(flags=flags, vel=vel)
		#addBuoyancy(density=density, vel=vel, gravity=vec3(0,-4e-3,0), flags=flags)

		#vv = vec3(10.0, 10.0, 0.0)
		#pp = vec3(int(0.5*res), int(0.1*res), 0) #This is in grid cell coordinates. i.e. [0-127] for a grid of 128
		#vel.addDataKDS(pp, vv)
		
		em.update(s.timestep, s.timestep * t)
		
		em.addVelocities(vel)
		'''
		vv = em.getForce()
		pp = em.getIntPosition()
		#vel.setDataKDS(pp, vv)
		vel.addDataKDS(pp, vv)
		'''

		solvePressure(flags=flags, vel=vel, pressure=pressure)

		timings.display()
		directory = "output/sim%03d"%simnum
		if not os.path.exists(directory):
			os.makedirs(directory)

		filename = "%03d.bin" % t
		fullFilename = directory + "/" + filename
		#writeOutSim(fullFilename,t,vel, pressure)
		s.step()
	#reset grid
	vel.clear()
	density.clear()
	pressure.clear()
