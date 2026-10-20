#!/usr/bin/env python
"""
vdb_to_sdf_bin.py

Usage:
    python vdb_to_sdf_bin.py input.vdb output.bin --dim 128 [--grid-name Density] [--iso 0.0]

- Reads a scalar grid from a .vdb file
- Samples it into an N x N x N dense voxel grid in uvw space, uvw in [0,1]^3
- Computes a signed distance field (SDF) to the > iso surface (default iso = 0)
  in uvw units (so voxel size is 1/N in each dimension).
- For voxels where the sampled value <= iso, the SDF is negative.
- Writes a binary file: int32 dimension, followed by N^3 float32 values (C-order).
"""

import argparse
import struct

import numpy as np
import pyopenvdb  # pip install openvdb or pyopenvdb
from scipy.ndimage import distance_transform_edt


def sample_vdb_to_dense(grid, dim, iso=0.0):
    """
    Sample an OpenVDB FloatGrid onto a dense (dim, dim, dim) numpy array in uvw space.

    uvw ∈ [0,1]^3, voxel centers at ((i+0.5)/dim, (j+0.5)/dim, (k+0.5)/dim)

    We map uvw to the world-space bounding box of the active voxels of the grid.
    """
    # Get bounding box of active voxels in index space
    ijk_min, ijk_max = grid.evalActiveVoxelBoundingBox()  # returns Coord, Coord

    # Convert bbox corners to world space
    # Note: ijk_max is inclusive; add (1,1,1) to get the "upper" corner
    min_world = grid.indexToWorld(ijk_min)
    max_world = grid.indexToWorld(ijk_max + openvdb.IntVector(1, 1, 1))

    # World-space extents
    extent = max_world - min_world

    # Allocate dense array
    dense = np.zeros((dim, dim, dim), dtype=np.float32)

    # For convenience
    ex, ey, ez = extent.x, extent.y, extent.z
    mx, my, mz = min_world.x, min_world.y, min_world.z

    # Triple loop: (k,j,i) -> (w,v,u) -> world -> sample
    # This is not highly optimized but is straightforward and correct.
    for k in range(dim):
        w = (k + 0.5) / dim
        z = mz + w * ez
        for j in range(dim):
            v = (j + 0.5) / dim
            y = my + v * ey
            for i in range(dim):
                u = (i + 0.5) / dim
                x = mx + u * ex

                # Sample the scalar field at this world position
                # grid.sample returns a float
                val = grid.sample((x, y, z))
                dense[k, j, i] = val

    return dense


def compute_sdf_from_scalar(dense, iso=0.0):
    """
    Given a dense scalar field `dense` (shape: (D, H, W)), compute a signed distance
    field to the > iso region in uvw units, assuming the cube size is (1,1,1).

    - Inside: dense > iso  -> SDF > 0
    - Outside/empty: dense <= iso -> SDF < 0

    We use a standard trick:
        dist_outside = EDT(mask_inside == 0)
        dist_inside  = EDT(mask_inside == 1)
        sdf = dist_inside - dist_outside

    We use sampling=(1/dim, 1/dim, 1/dim) to get distances in uvw units instead of voxel units.
    """
    dim_z, dim_y, dim_x = dense.shape
    assert dim_x == dim_y == dim_z, "Grid must be cubic"

    dim = dim_x
    voxel_size = 1.0 / dim

    # Mask of where the scalar is "inside" (> iso)
    inside = dense > iso

    # Distance transform:
    # - dist_outside: distance from each outside voxel to nearest inside voxel
    # - dist_inside:  distance from each inside voxel to nearest outside voxel
    dist_outside = distance_transform_edt(~inside, sampling=(voxel_size, voxel_size, voxel_size))
    dist_inside = distance_transform_edt(inside, sampling=(voxel_size, voxel_size, voxel_size))

    sdf = dist_inside - dist_outside  # >0 inside, <0 outside
    return sdf.astype(np.float32)


def write_sdf_bin(path, dim, sdf):
    """
    Write sdf (dim, dim, dim) float32 array to a binary file:

    - First 4 bytes: int32 dimension
    - Then dim^3 float32 values in C-order (z,y,x) flattening.
    """
    sdf = np.asarray(sdf, dtype=np.float32)
    assert sdf.shape == (dim, dim, dim)

    with open(path, "wb") as f:
        # Write dimension as 32-bit little-endian int
        f.write(struct.pack("<i", dim))

        # Write voxel data as float32
        f.write(sdf.ravel(order="C").tobytes())


def main():
    parser = argparse.ArgumentParser(description="Convert VDB to SDF dense grid and write to binary.")
    parser.add_argument("input_vdb", help="Path to input .vdb file")
    parser.add_argument("output_bin", help="Path to output .bin file")
    parser.add_argument("--dim", type=int, required=True, help="Symmetric voxel dimension (e.g. 128 -> 128^3)")
    parser.add_argument(
        "--grid-name",
        type=str,
        default=None,
        help="Name of the grid to use in the VDB file (if multiple). If not set, uses the first scalar grid."
    )
    parser.add_argument(
        "--iso",
        type=float,
        default=0.0,
        help="Iso-value threshold; > iso is treated as 'inside' (default: 0.0)"
    )

    args = parser.parse_args()

    dim = args.dim
    iso = args.iso

    # 1. Read VDB file
    vdb_file = openvdb.read(args.input_vdb)

    # Choose grid
    if args.grid_name is not None:
        if args.grid_name not in vdb_file.gridNames():
            raise RuntimeError(f"Grid '{args.grid_name}' not found in {args.input_vdb}. "
                               f"Available grids: {vdb_file.gridNames()}")
        grid = vdb_file.grid(args.grid_name)
    else:
        # Take first grid; ideally a FloatGrid
        names = vdb_file.gridNames()
        if not names:
            raise RuntimeError(f"No grids found in VDB file: {args.input_vdb}")
        grid = vdb_file.grid(names[0])

    if not isinstance(grid, openvdb.FloatGrid):
        print(f"Warning: grid '{grid.name}' is of type {type(grid)}, not FloatGrid. "
              f"Sampling may still work if it is scalar.")

    print(f"Using grid: '{grid.name}'")
    print(f"Sampling to dense grid of size {dim}^3...")

    # 2. Sample VDB to dense scalar grid in uvw space
    dense = sample_vdb_to_dense(grid, dim, iso=iso)

    # 3. Compute SDF in uvw units
    print("Computing signed distance field...")
    sdf = compute_sdf_from_scalar(dense, iso=iso)

    # 4. Write to binary file
    print(f"Writing SDF to {args.output_bin} (dim={dim})...")
    write_sdf_bin(args.output_bin, dim, sdf)

    print("Done.")


if __name__ == "__main__":
    main()
