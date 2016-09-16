#
# Smoke simulation with wavelet turbulence
# (This is a simpler example, a more generic and automated example can be found in waveletTurbObs.py)
# 

from manta import *
import os, shutil, math, sys, random

# dimension two/three d
dim = 2

# solver params
res = 130
gs = vec3(res,res,res)
if (dim==2): gs.z = 1  # 2D

numFrames = 256

random.seed(1945)

sm = Solver(name='main', gridSize = gs, dim=dim)
sm.timestep = 0.033

directory = "output/"
tt = 0.0

noises = []

if (GUI):
	gui = Gui()
	gui.show(True)

timings = Timings()

# allocate low-res grids
flags    = sm.create(FlagGrid)
vel      = sm.create(MACGrid)
density  = sm.create(RealGrid)
pressure = sm.create(RealGrid)
energy   = sm.create(RealGrid)

bWidth=1
flags.initDomain(boundaryWidth=bWidth)
flags.fillGrid() 
#setOpenBound(flags,bWidth,'xXyY',FlagOutflow|FlagEmpty)
setOpenBound(flags,bWidth,'xXyY',0)

#how large are the noise forces
#scale = random.uniform(1.0, 3.0)
scale = random.uniform(2.5, 5.0)

# main loop
for t in range(numFrames):
	
	curt = t * sm.timestep

	if(False and t == 0):
		f = vec3(0.001,0.1,0.0)
		setForce( flags=flags, vel=vel, force=f) # this sets the force on the entire velocity field	
	
	advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2)    
	advectSemiLagrange(flags=flags, vel=vel, grid=vel,     order=2, openBounds=True, boundaryWidth=bWidth )
	
	
	setWallBcs(flags=flags, vel=vel)    
	#addBuoyancy(density=density, vel=vel, gravity=vec3(0,-1e-3,0), flags=flags)

	vv = vec3(10.0, 10.0, 0.0)
	pp = vec3(100, 100, 0) #This is in grid cell coordinates. i.e. [0-127] for a grid of 128 	
	#vel.setDataKDS(pp, vv)
	vel.addDataKDS(pp, vv) 
	
	
	
	solvePressure(flags=flags, vel=vel, pressure=pressure , cgMaxIterFac=1.0, cgAccuracy=0.0001 )
	setWallBcs(flags=flags, vel=vel)
	
	
	#write everything to disk
	directory = "output/tr/000"
			
	#if not os.path.exists(directory):
	#  os.makedirs(directory)
	
	filename = "%03d.bin" % t
	fullFilename = directory + "/" + filename
	#writeOutSim(fullFilename,t,vel, pressure)
	
	if(tt < numFrames):
		f = "../../fluid_solver/cmake/%03d.bin" % tt;
		#writeOutSim(f,t,vel, pressure)
	tt = tt + 1
	
	sm.step()
	
	timings.display()

	#gui.screenshot( 'waveletTurb_%04d.png' % t );
