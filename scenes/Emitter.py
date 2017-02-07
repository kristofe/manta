try:
  from manta import *
except ImportError:
  # This happens when we're running EmitterTest.py.
  print('WARNING: Could not import manta. vec3 class is not defined.')

import os, shutil, math, sys, random

# TODO(tompson): This code is horrible. It started horrible and has only gotten
# worse. The Emitter class should be rewritten at some point to be a lot
# clearer.

# Append matlabnoise path.
sys.path.append(os.getcwd() + "/../../../matlabnoise/")
import matlabnoise

class Vec3Utils:
  @staticmethod
  def clone(a):
    return vec3(a.x, a.y, a.z)
  
  @staticmethod
  def dot(a, b):
    return a.x * b.x + a.y * b.y + a.z * b.z
  
  @staticmethod
  def lengthSquared(v):
    return Vec3Utils.dot(v,v)
  
  @staticmethod
  def length(v):
    return math.sqrt(Vec3Utils.lengthSquared(v))
 
  @staticmethod
  def normalizeVec3(p):
    length = math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
    if length > 1e-6:
      p.x /= length
      p.y /= length
      p.z /= length
    return p

  @staticmethod
  #source is vec3
  def setLength(source, s):
    len = math.sqrt(Vec3Utils.lengthSquared(source))
    if (len > 1e-4):
      source.x /= len
      source.y /= len
      source.z /= len

      source.x *= s
      source.y *= s
      source.z *= s
    return source
  
  @staticmethod
  def rotateZAxis(v, angleInRadians):
    if type(v) is list:
      v[0] = v[0] * math.cos(angleInRadians) - v[1] * math.sin(angleInRadians)
      v[1] = v[0] * math.sin(angleInRadians) + v[1] * math.cos(angleInRadians)
    else:
      v.x = v.x * math.cos(angleInRadians) - v.y * math.sin(angleInRadians)
      v.y = v.x * math.sin(angleInRadians) + v.y * math.cos(angleInRadians)
    return v
  
  @staticmethod
  def rotateXAxis(v, angleInRadians):
    if type(v) is list:
      v[1] = v[1] * math.cos(angleInRadians) - v[2] * math.sin(angleInRadians)
      v[2] = v[1] * math.sin(angleInRadians) + v[2] * math.cos(angleInRadians)
    else:
      v.y = v.y * math.cos(angleInRadians) - v.z * math.sin(angleInRadians)
      v.z = v.y * math.sin(angleInRadians) + v.z * math.cos(angleInRadians)
    return v

  @staticmethod
  def rotateYAxis(v, angleInRadians):
    if type(v) is list:
      v[2] = v[2] * math.cos(angleInRadians) - v[0] * math.sin(angleInRadians)
      v[0] = v[2] * math.sin(angleInRadians) + v[0] * math.cos(angleInRadians)
    else:
      v.z = v.z * math.cos(angleInRadians) - v.x * math.sin(angleInRadians)
      v.x = v.z * math.sin(angleInRadians) + v.x * math.cos(angleInRadians)
    return v
        
class Sphere3D:
  # @param location - python list of length 3.
  # @param radius - scalar value.
  def __init__(self, location, radius):
    self.center = list(location)  # deep copy.
    self.radius = radius
  
  # @param pos - python list of length 3.
  def signedDistance(self, pos):
    dx = pos[0] - self.center[0]
    dy = pos[1] - self.center[1]
    dz = pos[2] - self.center[2]
    len = math.sqrt(dx * dx + dy * dy + dz * dz)
    return len - self.radius
 
  # @param p - python list of length 3. 
  def isInside(self, p):
    dist = signedDistance(p)
    return dist <= 0.0 
    
testSphere = Sphere3D([0.0, 0.0, 0.0], 1.0)
assert(testSphere.signedDistance([2.0, 0.0, 0.0]) == 1.0), \
       "Failed sphere signed distance check"

class MathUtils:
  @staticmethod
  def sign(x):
    if x < 0.0:
        return -1.0
    return 1.0
  
  @staticmethod
  def smoothstep(edge0, edge1, x):
    x = MathUtils.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3 - 2 * x)
  
  @staticmethod
  def clamp(x, a, b):
    if (x < a):
      return a
    if (x > b):
      return b
    return x
  
  @staticmethod
  def sphereForceFalloff(sphere, location):
    signed_dist = sphere.signedDistance(location)
    if (signed_dist < 0.0):
      # We are inside the sphere..
      # This should make the center 1 and edge 0.0.
      t = signed_dist / -sphere.radius
      return MathUtils.smoothstep(0.0, 1.0, t)
    return 0.0
        
  @staticmethod
  def getRand01():
    return random.uniform(0.0, 1.0)
  
  
class ForceEmitter:
  def __init__(self, radius, velocity, amplitude, is3D, xdim, ydim, zdim,
               curvature, curvatureTScale, bWidth): 
    self.is3D_ = is3D
    self.radius_ = radius
    self.amplitude_ = amplitude
    self.curvature_ = curvature
    self.curvatureTScale_ = curvatureTScale
    if is3D:
      self.min_ = [bWidth, bWidth, bWidth]
      self.max_ = [xdim - 1 - bWidth, ydim - 1 - bWidth, zdim - 1 - bWidth]
    else:
      self.min_ = [bWidth, bWidth, 0]
      self.max_ = [xdim - 1 - bWidth, ydim - 1 - bWidth, 0]
    self.updateValidBounds()  # Calculates self.validMin_ and self.validMax_
    self.source_ = Sphere3D([0.0, 0.0, 0.0], self.radius_)
    self.resetPosition()
    self.resetVelocity(velocity)

  def getIntPosition(self):
    return [int(self.source_.center[0]), int(self.source_.center[1]),
            int(self.source_.center[2])]
  
  def getPosition(self):
    return self.source_.center
  
  def setRadius(self, radius):
    self.radius_ = radius
    self.updateValidBounds()
      
  def setBounds(self, pMin, pMax):
    self.min_ = pMin
    self.max_ = pMax
    self.updateValidBounds()
      
  def clipPosition(self, p):
    for i in range(3):
      p[i] = MathUtils.clamp(p[i], self.validMin_[i], self.validMax_[i])
    return p
  
  def isOutsideBounds(self, p):
    for i in range(3):
      if (p[i] < self.min_[i] or p[i] > self.max_[i]):
        return True
    return False

  def isOutsideValidBounds(self, p):
    for i in range(3):
      if (p[i] < self.validMin_[i] or p[i] > self.validMax_[i]):
        return True
    return False
      
  def wrapPosition(self, p):
    if (p[0] > self.validMax_[0]):
      p[0] = self.validMin_[0]
    if (p[1] > self.validMax_[1]):
      p[1] = self.validMin_[1]
    if (p[2] > self.validMax_[2]):
      p[2] = self.validMin_[2]
    
    if (p[0] < self.validMin_[0]):
      p[0] = self.validMax_[0]
    if (p[1] < self.validMin_[1]):
      p[1] = self.validMax_[1]
    if (p[2] < self.validMin_[2]):
      p[2] = self.validMax_[2]
    return p
  
  def updateValidBounds(self):
    self.validMin_ = [self.min_[0] + self.radius_,
                      self.min_[1] + self.radius_,
                      self.min_[2] + self.radius_]
    self.validMax_ = [self.max_[0] - self.radius_,
                      self.max_[1] - self.radius_,
                      self.max_[2] - self.radius_]
    if (self.is3D_ == False):
      self.validMin_[2] = 0
      self.validMax_[2] = 0
      
  def update(self, dt, time):
    newPos = list(self.source_.center)

    for i in range(3):
      newPos[i] += self.velocity_[i] * dt
    
    # If we go outside the bounds then randomly pick between changing
    # direction or wrapping the position.
    if self.isOutsideValidBounds(newPos):
      if random.uniform(0.0, 1.0) > 0.8:
        newPos = self.wrapPosition(newPos)  # Wrap with 20% probability.
      else:
        # Otherwise reflect off the border.
        for i in range(3):
          if newPos[i] < self.validMin_[i] or newPos[i] > self.validMax_[i]:
            self.velocity_[i] *= -1
        # rollback time.
        newPos = list(self.source_.center)
        # Update the timestep again with this new velocity.
        for i in range(3):
          newPos[i] += self.velocity_[i] * dt
        # Just in case something went wrong (float issues).
        newPos = self.clipPosition(newPos)
    
    self.source_.center = newPos

    # Calculate a smoothly varying radial acceleration.
    accel = [0, 0, 0]
    tcurve = time * self.curvatureTScale_ + self.curvatureOffset_
    for i in range(3):
      accel[i] = self.curvature_ * matlabnoise.Perlin2D(tcurve, i * 4)
    if (self.is3D_ == False):
      accel[2] = 0

    # Apply the acceleration to velocity.
    # TODO(tompson): This should be axis angle.
    self.velocity_ = Vec3Utils.rotateZAxis(self.velocity_,
                                           accel[0] * math.pi * 2 * dt)
    if (self.is3D_):
      self.velocity_ = Vec3Utils.rotateXAxis(self.velocity_,
                                             accel[1] * math.pi * 2 * dt)
      self.velocity_ = Vec3Utils.rotateYAxis(self.velocity_,
                                             accel[2] * math.pi * 2 * dt)
    
  def resetPosition(self):
    p = [0, 0, 0]
    for i in range(3):
      p[i] = MathUtils.getRand01() * (self.validMax_[i] - self.validMin_[i]) + \
          self.validMin_[i]
    assert(not self.isOutsideValidBounds(p))
    self.source_.center = p 
 
  def addVelocities(self, vg, flags, borderWidth):  # vg is a MAC grid (manta)
    dims = vg.getSizeVec3i()
    maxX = dims.x - borderWidth
    maxY = dims.y - borderWidth
    maxZ = dims.z - borderWidth
    pos = self.getIntPosition()
    r = int(self.source_.radius)

    # We have to sample the velocity at 3 locations
    offsets = [ [ 0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0] ]
    for x in range(-r, r, 1):
      for y in range(-r, r, 1):
        rz = 1
        if self.is3D_:
          rz = r
        for z in range(-rz, rz, 1):
          for o in range(0, len(offsets)):
            pp = list(pos)  # Clone.
            offset = offsets[o]
            pp[0] += x
            pp[1] += y
            pp[2] += z

            samplePos = list(pp)  # Clone.
            samplePos[0] += offset[0]
            samplePos[1] += offset[1]
            samplePos[2] += offset[2]

            if (self.isOutsideBounds(pp) != True and 
              pp[0] > borderWidth and pp[0] < maxX and
              pp[1] > borderWidth and pp[1] < maxY):
              t = MathUtils.sphereForceFalloff(self.source_, samplePos)
              # Force should be in the direction of velocity, multiplied by
              # the amplitude.

              # Force should be equal to the velocity multiplied by the
              # amplitude
              v = list(self.velocity_)
              v = ForceEmitter.normalizeVec3(v)
              f = vec3(self.amplitude_ * t * v[0],
                       self.amplitude_ * t * v[1],
                       self.amplitude_ * t * v[2])
              pp = vec3(int(pp[0]), int(pp[1]), int(pp[2]))
              if (flags.isFluid(pp) and \
                  not flags.isObstacle(int(pp.x), int(pp.y), int(pp.z))):
                if (o == 0):
                  vg.addAtMACX(pp, f)
                elif (o == 1):
                  vg.addAtMACY(pp, f)
                elif (o == 2):
                  vg.addAtMACZ(pp, f)

  @staticmethod
  def normalizeVec3(p):
    length = math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
    if length > 1e-6:
      for i in range(3):
        p[i] /= length
    return p

  def resetVelocity(self, velocity):
    # Create a unit length direction vector.
    u = MathUtils.getRand01()
    v = MathUtils.getRand01()
    w = MathUtils.getRand01()
    direction = [(u - 0.5) * 2.0, (v - 0.5) * 2.0, (w - 0.5) * 2.0]
    if (self.is3D_ == False):
      direction[2] = 0.0
    direction = ForceEmitter.normalizeVec3(direction)
   
    self.velocity_ = list(direction)

    for i in range(3):
      self.velocity_[i] *= velocity

    self.curvatureOffset_ = MathUtils.getRand01() * 100

  def getSource(self):
    return self.source_

  def getFalloffFromLocation(self,location):
    return MathUtils.sphereForceFalloff(self.source_, self.location)

  def isLocationInsideEmitter(self, location):
    return source_.isInside(location)

