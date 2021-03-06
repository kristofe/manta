#
# Flip example with "smooth" 2nd order wall boundary conditions
# 
from manta import *

# solver params
secOrderBc = True
dim    = 3
res    = 64
#res    = 124
gs     = vec3(res,res,res)
if (dim==2):
	gs.z=1
s = Solver(name='main', gridSize = gs, dim=dim)
s.timestep = 0.8
minParticles = pow(2,dim)

# size of particles 
radiusFactor = 1.0

# prepare grids and particles
flags     = s.create(FlagGrid)
phi       = s.create(LevelsetGrid)
phiOrg    = s.create(LevelsetGrid)
phiObs    = s.create(LevelsetGrid)

vel       = s.create(MACGrid)
velOld    = s.create(MACGrid)
pressure  = s.create(RealGrid)
fractions = s.create(MACGrid)
tmpVec3   = s.create(VecGrid)
phiWalls  = s.create(LevelsetGrid)

pp       = s.create(BasicParticleSystem) 
pVel     = pp.create(PdataVec3) 
mesh     = s.create(Mesh)

# acceleration data for particle nbs
pindex = s.create(ParticleIndexSystem) 
gpi    = s.create(IntGrid)

# scene setup, 0=breaking dam, 1=drop into pool
bWidth=1
flags.initDomain(boundaryWidth=bWidth, phiWalls=phiWalls )
fluidVel = 0
fluidSetVel = 0
phi.setConst(999.)
phiObs.setConst(999.)

# standing dam
fluidbox = s.create(Box, p0=gs*vec3(0,0,0), p1=gs*vec3(1.0,0.3,1)) 
#fluidbox = s.create(Box, p0=gs*vec3(0,0,0), p1=gs*vec3(1.0,1.0,1)) 
phi.join( fluidbox.computeLevelset() )
# fluidbox2 = s.create(Box, p0=gs*vec3(0.1,0,0), p1=gs*vec3(0.2,0.35,1))  # ok
fluidbox2 = s.create(Box, p0=gs*vec3(0.1,0,0), p1=gs*vec3(0.2,0.75,1)) 
phi.join( fluidbox2.computeLevelset() )

phiObs.join(phiWalls)
if 1:
	sphere = s.create(Sphere, center=gs*vec3(0.66,0.3,0.5), radius=res*0.2)
	phiObs.join( sphere.computeLevelset() )
	#obsbox = s.create(Box, p0=gs*vec3(0.4,0.2,0), p1=gs*vec3(0.7,0.4,1))
	#obsbox = s.create(Box, p0=gs*vec3(0.3,0.2,0), p1=gs*vec3(0.7,0.6,1))
	#phiObs.join( obsbox.computeLevelset() )


flags.updateFromLevelset(phi)
phi.subtract( phiObs );
sampleLevelsetWithParticles( phi=phi, flags=flags, parts=pp, discretization=2, randomness=0.05 )

# testing
phiOrg.copyFrom(phi)

if fluidVel!=0:
	# set initial velocity
	fluidVel.applyToGrid( grid=vel , value=fluidSetVel )
	mapGridToPartsVec3(source=vel, parts=pp, target=pVel )

# also sets boundary flags for phiObs
updateFractions( flags=flags, phiObs=phiObs, fractions=fractions, boundaryWidth=bWidth )
setObstacleFlags(flags=flags, phiObs=phiObs, fractions=fractions)

if 1 and (GUI):
	gui = Gui()
	gui.show()
	gui.pause()

#main loop
for t in range(2500):
	
	# FLIP 
	pp.advectInGrid(flags=flags, vel=vel, integrationMode=IntRK4, deleteInObstacle=False, stopInObstacle=False )
	pushOutofObs( parts=pp, flags=flags, phiObs=phiObs )

	# make sure we have velocities throught liquid region
	mapPartsToMAC(vel=vel, flags=flags, velOld=velOld, parts=pp, partVel=pVel, weight=tmpVec3 ) 
	extrapolateMACFromWeight( vel=vel , distance=2, weight=tmpVec3 )  # note, tmpVec3 could be free'd now...
	markFluidCells( parts=pp, flags=flags, phiObs=phiObs )

	# create approximate surface level set, resample particles
	gridParticleIndex( parts=pp , flags=flags, indexSys=pindex, index=gpi )
	unionParticleLevelset( pp, pindex, flags, gpi, phi , radiusFactor ) 

	# extend levelset somewhat, needed by particle resampling in adjustNumber
	extrapolateLsSimple(phi=phi, distance=4, inside=True); 

	# forces & pressure solve
	addGravity(flags=flags, vel=vel, gravity=(0,-0.001,0))
	if(secOrderBc):
		extrapolateMACSimple( flags=flags, vel=vel , distance=2, intoObs=True )
		setWallBcs(flags=flags, vel=vel, fractions=fractions, phiObs=phiObs)	

		solvePressure(flags=flags, vel=vel, pressure=pressure, phi=phi, fractions=fractions )

		extrapolateMACSimple( flags=flags, vel=vel , distance=4, intoObs=True )
		setWallBcs(flags=flags, vel=vel, fractions=fractions, phiObs=phiObs)
	else:
		setWallBcs(flags=flags, vel=vel)	
		solvePressure(flags=flags, vel=vel, pressure=pressure, phi=phi)
		setWallBcs(flags=flags, vel=vel)
		extrapolateMACSimple( flags=flags, vel=vel )

	# set source grids for resampling, used in adjustNumber!
	pVel.setSource( vel, isMAC=True )
	adjustNumber( parts=pp, vel=vel, flags=flags, minParticles=1*minParticles, maxParticles=2*minParticles, phi=phi, radiusFactor=radiusFactor , exclude=phiObs ) 

	flipVelocityUpdate(vel=vel, velOld=velOld, flags=flags, parts=pp, partVel=pVel, flipRatio=0.97 )

	if (dim==3):
		phi.createMesh(mesh)
	
	#s.printMemInfo()
	s.step()

	# generate data for flip03_gen.py surface generation scene
	#pp.save( 'flipParts_%04d.uni' % t );

	if 0 and (GUI):
		gui.screenshot( 'flip06_%04d.png' % t );


