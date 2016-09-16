import numpy as np
import binvox_rw
import math
import random
import os
import re

class VoxelUtils():
  @staticmethod
  def get_voxel_z_slice(srcVoxels, zSlice):
    # Get a z slice (2D array) of the voxel data
    # zSlice is a value between 0 and 1.  It is a normalized coord. 
    srcDims = srcVoxels.dims
    zIndex = int(zSlice * srcDims[2])
    assert(zIndex >= 0 and zIndex < srcDims[2])
    np2DArray = srcVoxels.data[:,:,zIndex]
    return np2DArray

  @staticmethod
  def translate_voxels(trans, target):
    # Shift the voxels.  This can be done by simple index offsets.  But be
    # carefull to bounds check the lookup into the voxel source.
    dims = target.dims
    tmp = np.zeros_like(target.data)
    for i in range(0, dims[0]):
      newi = int(i + trans[0])
      for j in range(0, dims[1]):
        newj = int(j + trans[1])
        for k in range(0, dims[2]):
          newk = int(k + trans[2])
          if (newi < dims[0] and newj < dims[1] and newk < dims[2] and newi >= 0
              and newj >= 0 and newk >= 0):
            tmp[newi, newj, newk] = target.data[i, j, k]
    target.data[:] = tmp

  @staticmethod
  def move_voxels_to_origin(target):
    dims = target.dims
    bbox = VoxelUtils.calculate_bounding_box(target)
    bmin = bbox['min']
    assert(bmin[0] >= 0 and bmin[1] >= 0 and bmin[2] >= 0)
    shift = [-bmin[0], -bmin[1], -bmin[2]]
    VoxelUtils.translate_voxels(shift, target)

    bbox = VoxelUtils.calculate_bounding_box(target)
    bmin = bbox['min']
    assert(bmin[0] == 0 and bmin[1] == 0 and bmin[2] == 0)
    return bbox

  @staticmethod
  def calculate_bounding_box(bvVoxels):
    maxVal = [0, 0, 0]
    minVal = bvVoxels.dims[:]
    count = 0
    for z in range(0, bvVoxels.dims[0]):
      for y in range(0, bvVoxels.dims[1]):
        for x in range(0, bvVoxels.dims[2]):
          if bvVoxels.data[x, y, z] > 0:
            count = count + 1
            if x < minVal[0]:
              minVal[0] = x
            if y < minVal[1]:
              minVal[1] = y
            if z < minVal[2]:
              minVal[2] = z

            if x > maxVal[0]:
              maxVal[0] = x
            if y > maxVal[1]:
              maxVal[1] = y
            if z > maxVal[2]:
              maxVal[2] = z
# Make sure we don't have an empty voxel array
    assert(count > 0)
    return {'min':minVal, 'max':maxVal}

  @staticmethod
  def intersection_test_aabb_to_aabb(a, b):
    assert('min' in a and 'min' in b)
    assert('max' in a and 'max' in b)
    if(a.min[0] > b.max[0] or a.max[0] < b.min[0]):
      return False
    if(a.min[1] > b.max[1] or a.max[1] < b.min[1]):
      return False
    if(a.min[2] > b.max[2] or a.max[2] < b.min[2]):
      return False
    return True

  @staticmethod
  def does_aabb_encompass_other(src, other):
    assert('min' in src and 'min' in other)
    assert('max' in src and 'max' in other)
    srcmin = src['min']
    srcmax = src['max']
    othermin = other['min']
    othermax = other['max']
    if (
      srcmin[0] <= othermin[0] and srcmax[0] >= othermax[0] and
      srcmin[1] <= othermin[1] and srcmax[1] >= othermax[1] and
      srcmin[2] <= othermin[2] and srcmax[2] >= othermax[2]
      ):
      return True
    return False

  @staticmethod
  def expand_aabb_info(a):
    assert('min' in a and 'max' in a)
    amin = a['min']
    amax = a['max']
    dims = [amax[0] - amin[0], amax[1] - amin[1], amax[2] - amin[2]]
    assert(dims[0] >= 0 and dims[1] >= 0 and dims[2] >= 0)

    halfdims = [dims[0] * 0.5, dims[1] * 0.5, dims[2] * 0.5]
    center = [amin[0] + halfdims[0], amin[1] + halfdims[1], amin[2] +
        halfdims[2]]
    assert(
        center[0] == (amax[0] - amin[0]) * 0.5 and
        center[1] == (amax[1] - amin[1]) * 0.5 and
        center[2] == (amax[2] - amin[2]) * 0.5
        )
    a['dims'] = dims
    a['halfdims'] = halfdims
    a['center'] = center
    return a

  @staticmethod
  def move_centroid_to_center(bvVoxels):
    # Calculate Centroid.
    dims = bvVoxels.dims
    voxelCount = 0
    values = [0, 0, 0,]

    for i in range(0, dims[0]):
      for j in range(0, dims[1]):
        for k in range(0, dims[2]):
          if bvVoxels.data[i, j, k] > 0:
            values[0] = values[0] + i
            values[1] = values[1] + j
            values[2] = values[2] + k
            voxelCount = voxelCount + 1

    centroid = [values[0] / voxelCount, values[1] / voxelCount, values[2] /
        voxelCount]
    center = [dims[0] * 0.5, dims[1] * 0.5, dims[2] * 0.5]
    shift = [math.floor(center[0] - centroid[0]), math.floor(center[1] -
        centroid[1]), math.floor(center[2] - centroid[2])]
    VoxelUtils.translate_voxels(shift, bvVoxels)

  @staticmethod
  def move_aabb_to_center(bvVoxels):
    bbox = VoxelUtils.calculate_bounding_box(bvVoxels)
    dims = bvVoxels.dims

    VoxelUtils.expand_aabb_info(bbox)
    centroid = bbox['center']
    center = [dims[0] * 0.5, dims[1] * 0.5, dims[2] * 0.5]
    shift = [math.floor(center[0] - centroid[0]), math.floor(center[1] -
        centroid[1]), math.floor(center[2] - centroid[2])]

    VoxelUtils.translate_voxels(shift, bvVoxels)

  @staticmethod
  def randomly_move_aabb_within_voxels(bvVoxels):
    VoxelUtils.move_voxels_to_origin(bvVoxels)
    
    bbox = VoxelUtils.calculate_bounding_box(bvVoxels)
    dims = bvVoxels.dims
    vmax = bbox['max']
    vmin = bbox['min']

    assert(vmin[0] == 0 and vmin[1] == 0 and vmin[2] == 0)
    assert(vmax[0] <= dims[0] and vmax[1] <= dims[1] and vmax[2] <= dims[2])

    roomPerDim = [dims[0] - vmax[0], dims[1] - vmax[1], dims[2] - vmax[2]]
    assert(roomPerDim[0] > 0 and roomPerDim[1] > 0 and roomPerDim[2] > 0)

    shift = [random.randint(0, roomPerDim[0]), random.randint(0, roomPerDim[1]),
        random.randint(0, roomPerDim[2])]

    VoxelUtils.translate_voxels(shift, bvVoxels)

  @staticmethod
  def uniform_scale(scale, target):
    assert(scale > 0.0)
    dims = target.dims
    bbox = VoxelUtils.move_voxels_to_origin(target)
    bmax = bbox['max']
    bmax[0] = bmax[0] * scale
    bmax[1] = bmax[1] * scale
    bmax[2] = bmax[2] * scale
    assert(bmax[0] < dims[0] and bmax[1] < dims[1] and bmax[2] < dims[2])

    inverseScale = 1.0 / scale
    tmp = np.zeros_like(target.data)
    for i in range(0, dims[0]):
      newi = int(math.floor(i * inverseScale))
      for j in range(0, dims[1]):
        newj = int(math.floor(j * inverseScale))
        for k in range(0, dims[2]):
          newk = int(math.floor(k * inverseScale))
          if (newi < dims[0] and newj < dims[1] and newk < dims[2] and newi >= 0
              and newj >= 0 and newk >= 0):
            tmp[i, j, k] = target.data[newi, newj, newk]

    target.data[:] = tmp

  @staticmethod
  def expand_voxels_to_dims(width, height, depth, target):
    dims = target.dims
    assert(target.axis_order == 'xyz')
    assert(dims[0] <= width and dims[1] <= height and dims[2] <= depth)
    tmp = np.zeros([width, height, depth])
    tmp[0:dims[0], 0:dims[1], 0:dims[2]] = target.data
    target.data = tmp
    target.dims = list(tmp.shape)

  @staticmethod
  def flip_diagonal(bvVoxels, plane):
    dims = bvVoxels.dims
    assert(plane >= 0 and plane <= 2)
    if plane == 0:
      assert(dims[1] == dims[2])
    elif plane == 1:
      assert(dims[0] == dims[2])
    else:
      assert(dims[0] == dims[1])

    tmp = np.zeros_like(bvVoxels.data)
    for i in range(0, dims[0]):
      for j in range(0, dims[1]):
        for k in range(0, dims[2]):
          ii = i
          jj = j
          kk = k
          if plane == 0:
            ii = i
            jj = k
            kk = j
          elif plane == 1:
            ii = k
            jj = j
            kk = i
          else:
            ii = j
            jj = i
            kk = k

          tmp[i, j, k] = bvVoxels.data[ii, jj, kk]
          tmp[ii, jj, kk] = bvVoxels.data[i, j, k]

    bvVoxels.data[:] = tmp

  @staticmethod
  def flip_voxels(bvVoxels, axis):
    dims = bvVoxels.dims
    assert(axis >= 0 and axis <= 2)
    tmp = np.zeros_like(bvVoxels.data)
    for i in range(0, dims[0]):
      for j in range(0, dims[1]):
        for k in range(0, dims[2]):
          ii = i
          jj = j
          kk = k
          if axis == 0:
            ii = (dims[0] - 1) - i
            jj = j
            kk = k
          elif axis == 1:
            ii = i
            jj = (dims[1] - 1)- j
            kk = k
          else:
            ii = i
            jj = j
            kk = (dims[2] - 1) - k
          tmp[i, j, k] = bvVoxels.data[ii, jj, kk]
          tmp[ii, jj, kk] = bvVoxels.data[i, j, k]
    bvVoxels.data[:] = tmp

  @staticmethod
  def blit_voxels_into_2d_target(offset, src2d, target2d):
    dims = target2d.shape
    sdims = src2d.shape
    assert(sdims[0] <= dims[0] and sdims[1] <= dims[1])
    start = [int(offset[0]), int(offset[1])]
    end  = [int(offset[0] + sdims[0]), int(offset[1] + sdims[1])]
    assert(offset[0] >= 0 and offset[1] >= 0)
    assert(sdims[0] >= 0 and sdims[1] >= 0)
    assert(end[0] <= dims[0] and end[1] <= dims[1])

    target2d[start[0]:end[0], start[1]:end[1]] = src2d

  @staticmethod
  def blit_voxels_into_target(offset, src, target):
    dims = target.dims
    sdims = src.dims
    assert(sdims[0] <= dims[0] and sdims[1] <= dims[1] and sdims[2] <= dims[2])
    start = [int(offset[0]), int(offset[1]), int(offset[2])]
    end  = [int(offset[0] + sdims[0]), int(offset[1] + sdims[1]),
        int(offset[2] + sdims[2])]
    assert(offset[0] >= 0 and offset[1] >= 0 and offset[2] >= 0)
    assert(sdims[0] >= 0 and sdims[1] >= 0 and sdims[2] >= 0)
    assert(end[0] <= dims[0] and end[1] <= dims[1] and end[2] <= dims[2])

    target.data[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = src.data

  @staticmethod
  def create_simple_grid_layout(srcList, srcDims, target):
    dims = target.dims
    assert(srcDims[0] <= dims[0] and srcDims[1] <= dims[1] and srcDims[2] <=
       dims[2])

    boxSize = max(srcDims)
    assert(boxSize > 0)
    numSlots = min([dims[0]/boxSize, dims[1]/boxSize, dims[2]/boxSize])
    assert(numSlots > 0)
    tmp = np.zeros_like(target.data)
    for i in range(0, numSlots):
      for j in range(0, numSlots):
        for k in range(0, numSlots):
          offset = [int(boxSize * i), int(boxSize * j), int(boxSize * k)]
          src = random.choice(srcList)
          VoxelUtils.blit_voxels_into_target(offset, src, target)

  @staticmethod
  def create_grid_layout_2d(srcList, srcDimsList, target2d, printMessages=True): 
    dims = target2d.shape
    assert(srcDimsList[0] <= dims[0] and srcDimsList[1] <= dims[1])

    boxSize = max(srcDimsList)
    assert(boxSize > 0)
    numSlots = int(min([dims[0]/boxSize, dims[1]/boxSize]))
    assert(numSlots > 0)

    count = 0
    total = numSlots * numSlots

    if (printMessages):
      print("Creating grid layout with " + str(total) + " slots")

    for i in range(0, numSlots):
      for j in range(0, numSlots):
        count = count + 1
        offset = [int(boxSize * i), int(boxSize * j)]
        src = random.choice(srcList)
        tmp = src.clone()
        VoxelUtils.move_voxels_to_origin(tmp)
        choice = random.randint(0, 3)
        if (choice == 0):
          VoxelUtils.flip_voxels(tmp, random.randint(0,2))
        elif (choice == 1):
          VoxelUtils.flip_diagonal(tmp, random.randint(0,2))
        elif (choice == 2):
          VoxelUtils.flip_voxels(tmp, random.randint(0,2))
          VoxelUtils.flip_diagonal(tmp, random.randint(0,2))
        # Any other choice (in this case it can only be 3 due to the randint()
        # call) means no flips to voxels.

        VoxelUtils.randomly_move_aabb_within_voxels(tmp)

        zSlice = 0.5
        voxelSlice = VoxelUtils.get_voxel_z_slice(tmp, zSlice)
        VoxelUtils.blit_voxels_into_2d_target(offset, voxelSlice, target2d)

      if (printMessages):
        print("%d percent done" % ((float(count)/(total)) * 100 ))

  @staticmethod
  def create_grid_layout(srcList, srcDimsList, target, printMessages=True): 
    dims = target.dims
    assert(srcDimsList[0] <= dims[0] and srcDimsList[1] <= dims[1] and
        srcDimsList[2] <= dims[2])

    boxSize = max(srcDimsList)
    assert(boxSize > 0)
    numSlots = int(min([dims[0]/boxSize, dims[1]/boxSize, dims[2]/boxSize]))
    assert(numSlots > 0)

    count = 0
    total = numSlots * numSlots * numSlots

    if (printMessages):
      print("Creating grid layout with " + str(total) + " slots")

    tmp = np.zeros_like(target.data)
    for i in range(0, numSlots):
      for j in range(0, numSlots):
        for k in range(0, numSlots):
          count = count + 1
          offset = [int(boxSize * i), int(boxSize * j), int(boxSize * k)]
          src = random.choice(srcList)
          tmp = src.clone()
          VoxelUtils.move_voxels_to_origin(tmp)
          choice = random.randint(0, 3)
          if (choice == 0):
            VoxelUtils.flip_voxels(tmp, random.randint(0,2))
          elif (choice == 1):
            VoxelUtils.flip_diagonal(tmp, random.randint(0,2))
          elif (choice == 2):
            VoxelUtils.flip_voxels(tmp, random.randint(0,2))
            VoxelUtils.flip_diagonal(tmp, random.randint(0,2))
          # Choice 3 means no flips to voxels.

          VoxelUtils.randomly_move_aabb_within_voxels(tmp)
          VoxelUtils.blit_voxels_into_target(offset, tmp, target)

        if (printMessages):
          print("%d percent done" % ((float(count)/(total)) * 100 ))

  @staticmethod
  def create_voxel_file_list(folderPath, expression, printMessages=True):
    path = os.getcwd() + "/" + folderPath
    if (printMessages):
      print("Loading voxel files from " + path)

    fileList = []
    for i in os.listdir(path):
      if re.match(expression, i):
        fileList.append(path + i)

    assert(len(fileList) > 0)
    if (printMessages):
      print("Found " + str(len(fileList)) + " files")

    modelList = []
    for filePath in fileList:
      with open(filePath, 'rb') as f:
        geom = binvox_rw.read_as_3d_array(f)
        # Don't append any empty files.
        if (geom.data.max() > 0):
          modelList.append(geom)
        else:
          print(filePath + " was empty!!")
    if (printMessages):
      print("Done loading voxel files.")
    
    return modelList

