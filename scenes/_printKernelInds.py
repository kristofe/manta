# A stupid script to verify (via printf) kernel bounds during macro expansion.

for dim in [2, 3]:
  gridSize = vec3(4, 5, 6)
  if dim == 2:
    gridSize.z = 1

  print('Domain size: ')
  print(gridSize)

  solver = Solver(name="main", gridSize=gridSize, dim=dim)

  flags = solver.create(FlagGrid)
  mac = solver.create(MACGrid)
  vec = solver.create(VecGrid)  # Internally a Grid<Vec3>
  real = solver.create(RealGrid)

  printKernelIndsBnd1(real)
  printKernelIndsBnd1(mac)
  printKernelIndsBnd1(vec)

  printKernelIndsIdx(real)
  printKernelIndsIdx(mac)
  printKernelIndsIdx(real)

  printKernelInds(real)
  printKernelInds(mac)
  printKernelInds(real)

