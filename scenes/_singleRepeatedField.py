#
# Smoke simulation with wavelet turbulence
# (This is a simpler example, a more generic and automated example can be found in waveletTurbObs.py)
# 

from manta import *
import os, shutil, math, sys, random

# dimension two/three d
dim = 2

# solver params
res = 128
gs = vec3(res,res,res)
if (dim==2): gs.z = 1  # 2D

numFrames = 2
numTraining = 20
numTest = 5
numSims = numTraining + numTest
keepSameNoise = True


random.seed(1945)

sm = Solver(name='main', gridSize = gs, dim=dim)
sm.timestep = 0.0330

directory = "output/"
tt = 0.0

noises = []

for i in range(numSims):
	if(i > 0 and keepSameNoise):
		noises.append(noises[0])
		continue
	
	fixedSeed = random.randint(0,512) #no idea what range this should be in
	noise = sm.create(NoiseField, fixedSeed=fixedSeed, loadFromFile=True)

	#how small are the details/vorticies - the larger the number the more detail
	#4 to 5 is a nice low range
	#20 is a nice high range on a 128 grid
	#noise.posScale = vec3(random.randint(20,20))
	noise.posScale = vec3(random.randint(4,4))
	noise.clamp = True
	noise.clampNeg = 0
	noise.clampPos = 2
	noise.valScale = 1.0
	noise.valOffset = 0.075
	noise.timeAnim = 0.0 #0.3
	
	noises.append(noise)

gui = None

for simnum in range(numSims):
	'''	
	if (GUI):
		if gui == None:
			gui = Gui()
			gui.show(True)
			gui.pause()
	'''
		
	timings = Timings()
	
	# allocate low-res grids
	flags    = sm.create(FlagGrid)
	vel      = sm.create(MACGrid)
	density  = sm.create(RealGrid)
	pressure = sm.create(RealGrid)
	energy   = sm.create(RealGrid)
	
	bWidth=0
	flags.initDomain(boundaryWidth=bWidth)
	flags.fillGrid() 
	#setOpenBound(flags,bWidth,'xXyY',FlagOutflow|FlagEmpty)
	setOpenBound(flags,bWidth,'xXyY',0)

	#noise field used to inialize velocity fields
	noise = noises[simnum]

	#how large are the noise forces
	scale = random.uniform(2.5, 5.0)
	
	# main loop
	for t in range(numFrames):
		
		#Make sure directory we are going to write to exists
		if(simnum < numTraining):
			directory = "output/tr/%03d"%simnum
		else:
			directory = "output/te/%03d"%(simnum - numTraining)
			
		if not os.path.exists(directory):
		  os.makedirs(directory)
		curt = t * sm.timestep

		if(t == 0):
			applyNoiseVec3( flags=flags, target=vel, noise=noise, scale=scale ) # just to test, add everywhere...
			#This can cause an increase in energy if strength is too high
			vorticityStrength = 0.2 #random.uniform(0.03, 0.3)
			vorticityConfinement( vel=vel, flags=flags, strength=vorticityStrength )
		
		#this is just so all of the arrays are instantiated for export
    		'''
		if(t == 0):
			advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2)    
			advectSemiLagrange(flags=flags, vel=vel, grid=vel,     order=2, openBounds=True, boundaryWidth=bWidth )
			setWallBcs(flags=flags, vel=vel)    
			solvePressure(flags=flags, vel=vel, pressure=pressure , cgMaxIterFac=1.0, cgAccuracy=0.0001 )
			setWallBcs(flags=flags, vel=vel)
    		'''		
		advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2)    
		advectSemiLagrange(flags=flags, vel=vel, grid=vel,     order=2, openBounds=True, boundaryWidth=bWidth )
		setWallBcs(flags=flags, vel=vel)    
		solvePressure(flags=flags, vel=vel, pressure=pressure , cgMaxIterFac=1.0, cgAccuracy=0.0001 )
		setWallBcs(flags=flags, vel=vel)

		

		
		tt = tt + 1
		
		sm.step()
		
		timings.display()
	
		#gui.screenshot( 'waveletTurb_%04d.png' % t );
		
	'''	
	if (GUI):
		if gui != None:
			gui.show(False)
			gui = None
	'''
