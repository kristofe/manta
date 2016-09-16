import unittest
from voxel_utils import VoxelUtils
import binvox_rw
import numpy as np

class TestVoxelUtils(unittest.TestCase):

  def test_calculate_bounding_box(self):
    voxels = self.createTestObject([10, 10, 10])
    voxels.data[0:2, 0:2, 0:2] = 1.0  # Slicing doesn't include the last index.
    bbox = VoxelUtils.calculate_bounding_box(voxels)
    self.assertEqual(bbox['min'], [0, 0, 0])
    self.assertEqual(bbox['max'], [1, 1, 1])

  def test_move_voxels_to_origin(self):
    voxels = self.createTestObject([5, 5, 5])
    voxels.data[3:5, 3:5, 3:5] = 1.0
    VoxelUtils.move_voxels_to_origin(voxels)
    bbox = VoxelUtils.calculate_bounding_box(voxels)
    self.assertEqual(bbox['min'], [0, 0, 0])
    self.assertEqual(bbox['max'], [1, 1, 1])

  def test_translate_voxels(self):
    voxels = self.createTestObject([5, 5, 5])
    voxels.data[3:5, 3:5, 3:5] = 1.0
    shift = [-3, -3, -3]
    VoxelUtils.translate_voxels(shift, voxels)
    bbox = VoxelUtils.calculate_bounding_box(voxels)
    self.assertEqual(bbox['min'], [0, 0, 0])
    self.assertEqual(bbox['max'], [1, 1, 1])
    voxels = self.createTestObject([5, 5, 5])
    voxels.data[0, 0, 0] = 1.0
    shift = [3, 3, 3]
    VoxelUtils.translate_voxels(shift, voxels)
    bbox = VoxelUtils.calculate_bounding_box(voxels)
    self.assertEqual(bbox['min'], [3, 3, 3])
    self.assertEqual(bbox['max'], [3, 3, 3])

  def test_uniform_scale(self):
    voxels = self.createTestObject([10, 10, 10])
    voxels.data[3:5, 3:5, 3:5] = 1.0
    bbox = VoxelUtils.calculate_bounding_box(voxels)
    self.assertEqual(bbox['min'], [3, 3, 3])
    self.assertEqual(bbox['max'], [4, 4, 4])
    scale = 4.0
    VoxelUtils.uniform_scale(scale, voxels)
    bbox = VoxelUtils.calculate_bounding_box(voxels)
    self.assertEqual(bbox['min'], [0, 0, 0])
    self.assertEqual(bbox['max'], [7, 7, 7])

  def test_move_centroid_to_center(self):
    voxels = self.createTestObject([5, 5, 5])
    voxels.data[0:3, 0:3, 0:3] = 1.0 # put centroid at 1, 1, 1
    VoxelUtils.move_centroid_to_center(voxels)
    bbox = VoxelUtils.calculate_bounding_box(voxels)
    self.assertEqual(bbox['min'], [1, 1, 1])
    self.assertEqual(bbox['max'], [3, 3, 3])

  def test_move_aabb_to_center(self):
    voxels = self.createTestObject([5, 5, 5])
    voxels.data[0:3, 0:3, 0:3] = 1.0 # put centroid at 1, 1, 1
    VoxelUtils.move_aabb_to_center(voxels)
    bbox = VoxelUtils.calculate_bounding_box(voxels)
    self.assertEqual(bbox['min'], [1, 1, 1])
    self.assertEqual(bbox['max'], [3, 3, 3])

  def test_expand_voxels_to_dims(self):
    voxels = self.createTestObject([5, 5, 5])
    voxels.data[0:2, 0:2, 0:2] = 1.0
    VoxelUtils.expand_voxels_to_dims(10, 10, 10, voxels)
    self.assertEqual(voxels.data.shape, (10, 10, 10))
    self.assertEqual(voxels.data[0, 0, 0], 1)
    bbox = VoxelUtils.calculate_bounding_box(voxels)
    self.assertEqual(bbox['min'], [0, 0, 0])
    self.assertEqual(bbox['max'], [1, 1, 1])

  def test_flip_diagonal(self):
    # Test rotate on x plane
    voxels = self.createTestObject([3, 3, 3])
    voxels.data[0, 0, 0] = 1.0
    voxels.data[0, 1, 0] = 1.0
    bbox = VoxelUtils.calculate_bounding_box(voxels)
    self.assertEqual(bbox['min'], [0, 0, 0])
    self.assertEqual(bbox['max'], [0, 1, 0])
    VoxelUtils.flip_diagonal(voxels, 0)
    self.assertEqual(voxels.data[0, 0, 0], 1)
    self.assertEqual(voxels.data[0, 0, 1], 1)
    bbox = VoxelUtils.calculate_bounding_box(voxels)
    self.assertEqual(bbox['min'], [0, 0, 0])
    self.assertEqual(bbox['max'], [0, 0, 1])

    # Test rotate on y plane
    voxels = self.createTestObject([3, 3, 3])
    voxels.data[0, 0, 0] = 1.0
    voxels.data[0, 0, 1] = 1.0
    bbox = VoxelUtils.calculate_bounding_box(voxels)
    self.assertEqual(bbox['min'], [0, 0, 0])
    self.assertEqual(bbox['max'], [0, 0, 1])
    VoxelUtils.flip_diagonal(voxels, 1)
    self.assertEqual(voxels.data[1, 0, 0], 1)
    self.assertEqual(voxels.data[0, 0, 0], 1)
    bbox = VoxelUtils.calculate_bounding_box(voxels)
    self.assertEqual(bbox['min'], [0, 0, 0])
    self.assertEqual(bbox['max'], [1, 0, 0])

    # Test rotate on z plane
    voxels = self.createTestObject([3, 3, 3])
    voxels.data[0, 0, 0] = 1.0
    voxels.data[1, 0, 0] = 1.0
    bbox = VoxelUtils.calculate_bounding_box(voxels)
    self.assertEqual(bbox['min'], [0, 0, 0])
    self.assertEqual(bbox['max'], [1, 0, 0])
    VoxelUtils.flip_diagonal(voxels, 2)
    self.assertEqual(voxels.data[0, 0, 0], 1)
    self.assertEqual(voxels.data[0, 1, 0], 1)
    bbox = VoxelUtils.calculate_bounding_box(voxels)
    self.assertEqual(bbox['min'], [0, 0, 0])
    self.assertEqual(bbox['max'], [0, 1, 0])

  def test_flip_voxels(self):
    # Test flip on x axis
    voxels = self.createTestObject([3, 3, 3])
    voxels.data[0, 0, 0] = 1.0
    voxels.data[0, 1, 0] = 1.0
    bbox = VoxelUtils.calculate_bounding_box(voxels)
    self.assertEqual(bbox['min'], [0, 0, 0])
    self.assertEqual(bbox['max'], [0, 1, 0])
    VoxelUtils.flip_voxels(voxels, 0)
    self.assertEqual(voxels.data[2, 0, 0], 1)
    self.assertEqual(voxels.data[2, 1, 0], 1)
    bbox = VoxelUtils.calculate_bounding_box(voxels)
    self.assertEqual(bbox['min'], [2, 0, 0])
    self.assertEqual(bbox['max'], [2, 1, 0])

    # Test flip on y axis
    voxels = self.createTestObject([3, 3, 3])
    voxels.data[0, 0, 0] = 1.0
    voxels.data[0, 1, 0] = 1.0
    bbox = VoxelUtils.calculate_bounding_box(voxels)
    self.assertEqual(bbox['min'], [0, 0, 0])
    self.assertEqual(bbox['max'], [0, 1, 0])
    VoxelUtils.flip_voxels(voxels, 1)
    self.assertEqual(voxels.data[0, 2, 0], 1)
    self.assertEqual(voxels.data[0, 1, 0], 1)
    bbox = VoxelUtils.calculate_bounding_box(voxels)
    self.assertEqual(bbox['min'], [0, 1, 0])
    self.assertEqual(bbox['max'], [0, 2, 0])

    # Test flip on z axis
    voxels = self.createTestObject([3, 3, 3])
    voxels.data[0, 0, 0] = 1.0
    voxels.data[0, 0, 1] = 1.0
    bbox = VoxelUtils.calculate_bounding_box(voxels)
    self.assertEqual(bbox['min'], [0, 0, 0])
    self.assertEqual(bbox['max'], [0, 0, 1])
    VoxelUtils.flip_voxels(voxels, 2)
    self.assertEqual(voxels.data[0, 0, 2], 1)
    self.assertEqual(voxels.data[0, 0, 1], 1)
    bbox = VoxelUtils.calculate_bounding_box(voxels)
    self.assertEqual(bbox['min'], [0, 0, 1])
    self.assertEqual(bbox['max'], [0, 0, 2])

  def test_blit_voxels_into_target(self):
    target = self.createTestObject([10, 10, 10])
    src = self.createTestObject([2, 2, 2])
    src.data[0:2, 0:2, 0:2] = 1.0 
    offset = [2, 2, 2]
    VoxelUtils.blit_voxels_into_target(offset, src, target)
    self.assertEqual(target.data[2, 2, 2], 1)
    self.assertEqual(target.data[3, 2, 2], 1)
    bbox = VoxelUtils.calculate_bounding_box(target)
    self.assertEqual(bbox['min'], [2, 2, 2])
    self.assertEqual(bbox['max'], [3, 3, 3])

  def test_create_simple_grid_layout(self):
    target = self.createTestObject([4, 4, 4])
    src2 = self.createTestObject([2, 2, 2])
    src2.data[0, 0, 0] = 1.0 
    srcList = [src2, src2]
    srcDims = src2.dims
    VoxelUtils.create_simple_grid_layout(srcList, srcDims, target)
    self.assertEqual(target.data[0, 0, 0], 1)
    self.assertEqual(target.data[2, 0, 0], 1)
    self.assertEqual(target.data[0, 2, 0], 1)
    self.assertEqual(target.data[2, 2, 0], 1)

    self.assertEqual(target.data[0, 0, 2], 1)
    self.assertEqual(target.data[2, 0, 2], 1)
    self.assertEqual(target.data[0, 2, 2], 1)
    self.assertEqual(target.data[2, 2, 2], 1)

  def test_create_voxel_file_list(self):
    VoxelUtils.create_voxel_file_list("../../voxelizer/voxels/", ".*[32,64].binvox")

  def printDataArray(voxels):
    for k in range(0, voxels.dims[2]):
      for j in range(0, voxels.dims[1]):
          print( voxels.data[:, j, k])

  def createTestObject(self, dims):
    translate = [0, 0, 0]
    scale = [1, 1, 1]
    axis_order = 'xyz'
    data = np.zeros(dims)
    voxels = binvox_rw.Voxels(data, dims, translate, scale, axis_order, "test")
    return voxels

if __name__ == '__main__':
  unittest.main()

