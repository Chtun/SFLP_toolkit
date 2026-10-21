"""Pixel Arrangement module for SFLP method.

This module includes helpful functions for generating the pixel arrangement and decoding the pixel arrangement.
It includes utilities for validation.

Methods:
  golden_ratio: Returns the golden ratio.
  get_SFL_vector: Gets the vector for a particular Spherical Fibonacci Lattice point.
  get_projected_point: Projects a point in geographic coordinates to a point in the unnormalized z-theta plane.
  get_projected_SFL_point: Gets the coordinates in the unnormalized z-theta plane for a particular Spherical Fibonacci Lattice point.
  get_projected_generator_vector: Gets the vertical and horizontal components of a generator vector in the unnormalized z-theta plane.
  generator_vec_geodesic_distance: Gets the generator vector's distance on the sphere from the equator.
  
  find_optimal_generator_vectors: Finds the optimal nearest-neighbor generator vectors.
  
  get_tile_index: Gets the tile index that the given point projected in the z-theta plane falls in.
  get_row_col_index: Gets the row and column index for a particular SFLP index based on the number of SFLP pixels and associated metadata.
  
  generate_SFL_projected_gpu: GPU-accelerated function to project a set of points to the unnormalized z-theta plane.
  get_SFL_geographic_points: GPU-accelerated function that gets all points in the Spherical Fibonacci Lattice in geographic coordinates.

  generate_pixel_arrangement: GPU-accelerated function that generates the pixel arrangement for a single 360-degree frame given a set number of pixels.
  generate_CubeMap_pixel_arrangement:   Generates the pixel arrangement for each a single CMP/EAC frame.

"""


import math
import numpy as np
from typing import Tuple
from pydantic import ValidationError
import cupy as cp
import pandas as pd
from enum import Enum

from SFLP_toolkit.data_types import Vector, GeographicPoint, Point


class SFLPArrangementType(Enum):
    STANDARD = "STANDARD"

    # Experimental
    SEAM_REMOVED = "SEAM_REMOVED"
    ZIGZAG = "ZIGZAG"

#
# region Helper Functions for Vector Mathematics
#

def golden_ratio():
    """
    Returns the golden ratio.
    """
    return (1 + math.sqrt(5)) / 2

def get_SFL_vector(index: int, num_points: int) -> Vector:
  """
  Get the Spherical Fibonacci Lattice vector for a particular index value and number of SFL points.

  Args:
    index: The index of the SFL point.
    num_points: The number of SFL points.

  Returns:
    The Spherical Fibonacci Lattice vector.
  """
  if num_points <= 0:
      raise ValidationError("Number of points must be positive")

  phi = golden_ratio()

  if (num_points % 2 == 0):
      num_points -= 1
      print(f"Warning: The number of points must be odd, so Fibonacci lattice point generation will be done with n={num_points}")

  N = int(np.floor((num_points - 1) / 2))

  # Get the i from the index.
  i = index - N

  lat = np.arcsin(2 * i / (2 * N + 1)) * 180 / np.pi
  lon = (i % phi) * 360 / phi

  # Normalize longitude to [-180, 180]
  lon = ((lon + 180) % 360) - 180

  # Convert to vector and return
  return Vector.from_geographic(lon, lat)

def get_projected_point(geographic_point: GeographicPoint) -> Point:
  """
  Projects a point in geographic coordinates to a point in the unnormalized z-theta plane.

  Args:
    geographic_point: The geographic point.

  Returns:
    The projected point in the unnormalized z-theta plane.
  """
  theta = geographic_point.lon
  phi = 90 - geographic_point.lat

  return Point(theta, math.cos(math.radians(phi)))

def get_projected_SFL_point(index: int, num_points: int) -> Point:
  """
  Gets the projected Spherical Fibonacci Lattice point in the unnormalized projected z-theta plane.

  Args:
    index: The index of the Spherical Fibonacci Lattice point.
    num_points: The number of points in the Spherical Fibonacci Lattice.

  Returns:
    The projected Spherical Fibonacci Lattice point in the unnormalized z-theta plane.
  """
  N = int(np.floor((num_points - 1) / 2))
  i = index - N
  phi = golden_ratio()

  theta = (i % phi) * 360 / phi

  # Normalize theta to [-180, 180]
  theta = ((theta + 180) % 360) - 180

  z = 2 * i / (2 * N + 1)

  return Point(theta, z)

def get_projected_generator_vector(k_index: int, num_points: int) -> Point:
  """
  Get the generator vector in the unnormalized z-theta plane for a particular index hop value and number of SFL points.

  Args:
    index: The difference in index between any two SFL points.
    num_points: The number of FB points.

  Returns:
    The generator vector in the unnormalized z-theta plane.
  """

  if num_points <= 0:
      raise ValidationError("Number of points must be positive")

  N = int(np.floor((num_points - 1) / 2))
  phi = golden_ratio()
  frac = (k_index / phi) % 1.0
  # If the fraction is too large, then it should be in the other direction (for distance calculations).
  if frac > 0.5:
      frac = frac - 1

  x = frac * 360
  y = (2 * k_index) / (2 * N + 1)

  return Point(x, y)

def generator_vec_geodesic_distance(k_index: int, num_points: int) -> Tuple[float, float, float]:
  """
  Get the geodesic distance of two points on the sphere separated by
  the specified generator vector, where one point is at the equator.


  Args:
    k: The k-th index jump for each pixel.
    n: The number of pixels.

  Returns:
    The geodesic distance, unnormalized theta component, and unnormalized z component.
  """

  projected_generator_vector = get_projected_generator_vector(k_index, num_points)

  y_value = projected_generator_vector.y

  # Clamp the value to the range [-1, 1]
  clamped_y = max(-1.0, min(1.0, y_value))

  # Now use the clamped value
  latitude = math.degrees(math.asin(clamped_y))
  longitude =  projected_generator_vector.x

  generator_vector_from_reference = Vector.from_geographic(longitude, latitude)
  reference_vector = Vector(1, 0, 0)

  # Calculate the geodesic distance from the reference (1, 0, 0)
  dot_product = generator_vector_from_reference.dot(reference_vector)

  # Ensure the dot product is within the valid range for arccos due to potential floating-point inaccuracies
  dot_product = max(-1.0, min(1.0, dot_product))

  # Calculate the geodesic distance (angle in radians)
  geodesic_distance = math.acos(dot_product)

  return geodesic_distance, projected_generator_vector.x, projected_generator_vector.y

#
# endregion Helper Functions for Vector Mathematics
#

#
# region Finding the Nearest-Neighbor Generator Vectors
#

def find_optimal_generator_vectors(n, arrangement_type: SFLPArrangementType = SFLPArrangementType.STANDARD, k_max=-1):
  """
  Finds the two optimal generator vectors (i.e. the nearest neighbor generator vectors
  in the normalized z-theta plane) to build the array for, up to a given k_max.

  Args:
    n: The number of SFLP pixels (or SFL points).
    k_max: The maximum k-th index jump to consider.

  Returns:
    A list with containing the following properties of the two optimal generator vectors:
      1. The index hop.
      2. The generator vector magnitude.
      3. The unnormalized theta component [from -180 to 180].
      4. The unnormalized z component [from -1 to 1].

  """
  if k_max < 1:
    k_max = int(3 * math.sqrt(n))

  values = [(k, generator_vec_geodesic_distance(k, n)) for k in range(1, k_max + 1)]
  values = [(k, val[0], val[1], val[2]) for k, val in values]
  values.sort(key=lambda x: x[1])  # Sort by the function value

  nearest_neighbors = values[:4]

  nearest_neighbors.sort(key=lambda x: x[0], reverse=True)

  neighbor_one = nearest_neighbors[0][0]
  neighbor_two = nearest_neighbors[1][0]
  neighbor_three = nearest_neighbors[2][0]

  n1_distance = generator_vec_geodesic_distance(neighbor_one, num_points=n)[0]
  n2n3_distance = generator_vec_geodesic_distance((neighbor_two - neighbor_three), num_points=n)[0]

  if n1_distance/ n2n3_distance > 0.7:
    optimal_k_1 = neighbor_two
    optimal_k_2 = neighbor_three
  else:
    optimal_k_1 = neighbor_one
    optimal_k_2 = neighbor_two

  if arrangement_type == SFLPArrangementType.STANDARD:  
    optimal_k_hops = [optimal_k_1, optimal_k_2]
    optimal_k_hops = sorted(optimal_k_hops)
    optimal_k_1 = optimal_k_hops[1] - optimal_k_hops[0]
    optimal_k_2 = optimal_k_hops[0]

  optimal_k = [(optimal_k_1, generator_vec_geodesic_distance(optimal_k_1, n)), (optimal_k_2, generator_vec_geodesic_distance(optimal_k_2, n))]
  optimal_k = [(k, val[0], val[1], val[2]) for k, val in optimal_k]

  return optimal_k

#
# endregion Finding the Nearest-Neighbor Generator Vectors
#

#
# region SFLP Indexing
#

# region non-GPU compatible

def get_tile_index(projected_point: Point, num_tiles_theta: int, num_tiles_z: int):
  """
  Get the tile index for a given point projected in the unnormalized z-theta plane.

  Args:
    projected_point: The point projected in the unnormalized z-theta plane.
    num_tiles_theta: The number of tiles in the theta direction.
    num_tiles_z: The number of tiles in the z direction.

  Returns:
    A tupe (x, y) for the tile index.
  """

  # Z goes from -1 to 1.
  # Theta goes from -180 to 180.
  tile_x_index = math.floor((projected_point.x + 180) * (num_tiles_theta / 360))
  tile_y_index = math.floor((projected_point.y + 1) * (num_tiles_z / 2))

  if tile_x_index > num_tiles_theta - 1:
    tile_x_index = num_tiles_theta - 1

  if tile_y_index > num_tiles_z - 1:
    tile_y_index = num_tiles_z - 1

  return tile_x_index, tile_y_index

def get_row_index(
    SFLP_index: int,
    col_index: int,
    last_element_SFLP_index: int,
    row_index_hop: int,
    col_index_hop: int,
    col_length: int,
) -> int:
    """
    Gets the row index for a given SFLP index for STANDARD arrangement.

    Args:
      SFLP_index: The SFLP index of the pixel for which to get the row index.
      col_index: The column index of the pixel.
      last_element_SFLP_index: The SFLP index of the last element in the first column.
      row_index_hop: The SFLP index hop from one element in a row to the next.
      col_index_hop: The SFLP index hop from one element in a column to the next.
      col_length: The length of the columns (number of rows).

    """

    bottom_row_index = col_index * row_index_hop + last_element_SFLP_index

    row_index = (SFLP_index - bottom_row_index) // col_index_hop

    index_from_bottom = (col_length - 1) - row_index

    index_from_top = (-1 * row_index) - 1

    if row_index >= 0:
        row_index = index_from_bottom
    else:
        row_index = index_from_top

    return row_index

def get_row_index_seam_removed(projected_pixel_center: Point, col_generator_vector: Point, num_tiles_x: int, num_tiles_y: int, tile_x_index: int, tile_y_index: int, col_length: int):
  """
  Gets the row index for a given pixel center (projected in the unnormalized z-theta plane) in the seam-removed SFLP arrangement.

  Args:
    projected_pixel_center: The projected pixel center projected in the unnormalized z-theta plane.
    col_generator_vector: The generator vector for the column (gets from one row index to the next).
    num_tiles_x: The number of tiles horizontally.
    num_tiles_y: The number of tiles vertically.
    tile_x_index: The x index of the tile.
    tile_y_index: The y index of the tile.
    col_length: The length of every column in the table.

  Returns:
    A tuple (row_index, is_bottom_row) with the row index and if it is in or not in the bottom row in projected z-theta plane.
  """

  tile_height = 2 / num_tiles_y
  tile_width = 360 / num_tiles_x

  hop_vector_goes_left = col_generator_vector.x < 0

  # Z diff is from the bottom of the tile.
  z_diff = projected_pixel_center.y - (tile_y_index * tile_height - 1)

  # Handle the case when the hop vector goes to the left.
  if hop_vector_goes_left:
    # theta diff is from the right side of the tile.
    theta_diff = ((tile_x_index + 1) * tile_width - 180) - projected_pixel_center.x

  # Handle the case when the hop vector goes to the right.
  else:
    # theta diff is from the left side of the tile.
    theta_diff = projected_pixel_center.x - (tile_x_index * tile_width - 180)

  hops_horizontal = math.floor(theta_diff / abs(col_generator_vector.x))
  hops_vertical = math.floor(z_diff / abs(col_generator_vector.y))

  if hops_horizontal < hops_vertical:
    is_bottom_row = False

    z_diff_to_top = ((tile_y_index + 1) * tile_height - 1) - projected_pixel_center.y

    row_index = math.floor(z_diff_to_top / abs(col_generator_vector.y))
    row_index = col_length - 1 - row_index
  else:
    row_index = hops_vertical
    is_bottom_row = True

  # Row begins at the bottom of image and moves up, so need to flip around.
  row_index = (col_length - 1) - row_index

  return row_index, is_bottom_row

def get_col_index(first_SFLP_index_in_col: int, reference_element_SFLP_index: int, first_index_hop: int, second_index_hop: int, col_generator_vector: Point):
  """
  Gets the column index given the SFLP index of the first element in the column,
  the element of the first column's SFLP index, and the optimal index hops.
  This assumes that the first and second index hops are consecutive fibonacci numbers!

  Args:
    first_SFLP_index_in_col: The SFLP index of the first element in the column.
    reference_element_SFLP_index: The element of the first column's SFLP index.
    first_index_hop: The first optimal index hop.
    second_index_hop: The second optimal index hop.
    col_generator_vector: The generator vector for the column.

  Returns:
    The column index.
  """


  multiplier = int(reference_element_SFLP_index - first_SFLP_index_in_col)

  if col_generator_vector.x > 0:
    multiplier *= -1

  if first_index_hop > second_index_hop:
    prev_fb_number = first_index_hop - second_index_hop
    larger_fb_number = first_index_hop
  else:
    prev_fb_number = second_index_hop - first_index_hop
    larger_fb_number = second_index_hop

  col_index = ((prev_fb_number % larger_fb_number) * (multiplier % larger_fb_number)) % larger_fb_number

  return col_index

def get_zag_col_index(first_SFLP_index_in_col: int, reference_element_SFLP_index: int, zag_row_index_hop: int, zig_row_index_hop: int, row_generator_vector: Point): 
  """
  Gets the column index (for a zag line) given the SFLP index of the first element in the column,
  the element of the first column's SFLP index, and the optimal index hops.
  This assumes that the first and second index hops are fibonacci numbers separated by one index,
  i.e. F_n and F_(n+2)!

  Args:
    first_SFLP_index_in_col: The SFLP index of the first element in the column.
    reference_element_SFLP_index: The element of the first column's SFLP index.
    zag_row_index_hop: The index hop of the zag line.
    zig_row_index_hop: The index hop of the zig line.
    col_index_hop: The index hop that moves from one row to the next.
    row_generator_vector: The generator vector to get from one row to the next.

  Returns:
    The column index.
  """


  multiplier = int(reference_element_SFLP_index - first_SFLP_index_in_col)

  if row_generator_vector.x > 0:
    multiplier *= -1

  larger_fb_number = zag_row_index_hop + zig_row_index_hop

  col_index = ((larger_fb_number % zag_row_index_hop) * (multiplier % zag_row_index_hop)) % zag_row_index_hop

  return col_index

def get_row_col_index_standard(SFLP_index: int, num_points: int, optimal_index_hops: list, last_element_SFLP_index: int, row_length: int, col_length: int):
  """
  GPU-compatible function that gets the row and column index in a frame that is in the STANDARD arrangement,
  given the Fibonacci Lattice pixel indicies, the number of pixels, the optimal index hops,
  and the reference element's SFLP index and column index.

  Args:
    indices: The pixel indices to get row and column indices for.
    num_points: The number of pixels.
    optimal_index_hops: The two optimal k-th index hops.
    last_element_SFLP_index: The SFLP index of the last element in the first column.
    row_length, col_length: row and column length of the frame/image.

  Returns:
    A tuple (x, y) with the row and column index.
  """

  optimal_index_hops = sorted(optimal_index_hops, key=lambda x: x[3])
  row_index_hop = int(optimal_index_hops[0][0])
  col_index_hop = int(optimal_index_hops[1][0])

  col_generator_vector = get_projected_generator_vector(col_index_hop, num_points)

  col_index = get_col_index(
    SFLP_index,
    last_element_SFLP_index,
    row_index_hop,
    col_index_hop,
    col_generator_vector,
  )

  row_index = get_row_index(
    SFLP_index,
    col_index,
    last_element_SFLP_index,
    row_index_hop,
    col_index_hop,
    col_length,
  )

  return row_index, col_index

def get_row_col_index_seam_removed(SFLP_index: int, num_points: int, optimal_index_hops: list, num_tiles_x: int, num_tiles_y: int, first_element_SFLP_index: int, last_element_SFLP_index: int, row_length: int, col_length: int):
  """
  Gets the row and column index in a tile that is in the seam-removed arrangement,
  given the Fibonacci Lattice pixel index, the number of pixels, the optimal index hops,
  and the number of tiles in each direction, and the first element in the first row's SFLP index.

  Args:
    index: The pixel index.
    num_points: The number of pixels.
    optimal_index_hops: The two optimal k-th index hops.
    num_tiles_x: The number of tiles in the x direction.
    num_tiles_y: The number of tiles in the y direction.
    first_element_SFLP_index: The first element of the first column in the SFLP.
    last_element_SFLP_index: The last element of the first column in the SFLP.

  Returns:
    A tuple (x, y) with the row and column index.
  """

  optimal_index_hops = sorted(optimal_index_hops, key=lambda x: x[3])
  col_index_hop = optimal_index_hops[0][0]
  row_index_hop = optimal_index_hops[1][0]

  projected_pixel_center = get_projected_SFL_point(SFLP_index, num_points)
  row_generator_vector = get_projected_generator_vector(row_index_hop, num_points)

  tile_x_index, tile_y_index = get_tile_index(projected_pixel_center, num_tiles_x, num_tiles_y)

  row_index, is_bottom_row = get_row_index_seam_removed(
    projected_pixel_center,
    row_generator_vector,
    num_tiles_x,
    num_tiles_y,
    tile_x_index,
    tile_y_index,
    col_length
  )

  first_SLFP_index = SFLP_index - row_index * row_index_hop

  if is_bottom_row:
    reference_element_SFLP_index = last_element_SFLP_index 
  else:
    reference_element_SFLP_index = first_element_SFLP_index

  col_index = get_col_index(
    first_SLFP_index,
    reference_element_SFLP_index,
    col_index_hop,
    row_index_hop,
    row_generator_vector,
  )

  return row_index, col_index

def get_row_col_index_zigzag(SFLP_index: int, num_points: int, optimal_index_hops: list, num_tiles_x: int, num_tiles_y: int, first_element_SFLP_index: int, last_element_SFLP_index: int, row_length: int, col_length: int):
  """
  Gets the row and column index in a tile for a zigzag arrangement,
  given the Fibonacci Lattice pixel index, the number of pixels,the optimal index hops,
  and the number of tiles in each direction, and the first element in the first row's SFLP index.

  Args:
    index: The pixel index.
    num_points: The number of pixels.
    optimal_index_hops: The two optimal k-th index hops.
    num_tiles_x: The number of tiles in the x direction.
    num_tiles_y: The number of tiles in the y direction.
    first_element_SFLP_index: The first element of the first column in the SFLP.
    last_element_SFLP_index: The last element of the first column in the SFLP.

  Returns:
    A tuple (x, y) with the row and column index.
  """

  optimal_index_hops = sorted(optimal_index_hops, key=lambda x: x[3])
  zig_row_index_hop = optimal_index_hops[0][0]
  zag_row_index_hop = optimal_index_hops[1][0]
  col_index_hop = abs(zag_row_index_hop - zig_row_index_hop)
  combined_row_index_hop = zig_row_index_hop + zag_row_index_hop

  projected_pixel_center = get_projected_SFL_point(SFLP_index, num_points)
  combined_row_generator_vector = get_projected_generator_vector(combined_row_index_hop, num_points)
  zig_row_generator_vector = get_projected_generator_vector(zig_row_index_hop, num_points)
  zag_row_generator_vector = get_projected_generator_vector(zag_row_index_hop, num_points)

  tile_x_index, tile_y_index = get_tile_index(projected_pixel_center, num_tiles_x, num_tiles_y)
  tile_height = 2 / num_tiles_y

  # Determine whether the last element of first column is on an alternate row or not.
  top_row_starts_alternate = (first_element_SFLP_index - last_element_SFLP_index) % combined_row_index_hop != 0

  # Grab the combined row index first to figure out where the first pixel is in the combined generator line.
  combined_row_index, is_bottom_row = get_row_index_seam_removed(
    projected_pixel_center,
    combined_row_generator_vector,
    num_tiles_x,
    num_tiles_y,
    tile_x_index,
    tile_y_index,
    col_length
  )

  # Figure out whether the first SFLP combined index is on the alternate row or not.
  if is_bottom_row:
    reference_element_SFLP_index = last_element_SFLP_index
    first_SFLP_combined_index = SFLP_index - ((col_length - 1) - combined_row_index) * combined_row_index_hop
  else:
    reference_element_SFLP_index = first_element_SFLP_index 
    first_SFLP_combined_index = SFLP_index + (combined_row_index) * combined_row_index_hop

  # print(f"SFLP index: {SFLP_index}")

  # Obtaining the col index of the zig generator line and the zag generator line
  # helps determine where the actual col index is, as well as the first SFLP index in the column.
  zig_col_index = get_col_index(
    first_SFLP_combined_index,
    reference_element_SFLP_index,
    col_index_hop,
    zig_row_index_hop,
    zig_row_generator_vector
    )
  
  zag_col_index = get_zag_col_index(
    first_SFLP_combined_index,
    reference_element_SFLP_index,
    zag_row_index_hop,
    zig_row_index_hop,
    zag_row_generator_vector
  )

  # Correct for when on the same zig line, but not on the first element.
  if is_bottom_row and zig_col_index == 0 and first_SFLP_combined_index > reference_element_SFLP_index:
    zig_col_index += zig_row_index_hop
  # This corrects for when the top row starts alternate,
  # so that our current row is adjusted to the fact that it is being compared to an alternate row
  elif not is_bottom_row and top_row_starts_alternate:
    zag_col_index -= 1

  # Alternate row happens when zig_col and zag_col indices do not add to an even number.
  alternate_row = (zig_col_index + zag_col_index) % 2 == 1

  # The row index is twice the combined row index,
  # plus one if the first SFLP in the combined row can do a zig or zag hop back once more.
  if is_bottom_row:
    row_index = (col_length - 1) - 2 * ((col_length - 1) - combined_row_index)
  else:
    row_index = 2 * (combined_row_index)

  projected_first_SFLP_combined = get_projected_SFL_point(first_SFLP_combined_index, num_points)
  z_diff = projected_first_SFLP_combined.y - (tile_y_index * tile_height - 1)

  bottom_first_SFLP_combined_index = -1

  # Handle if the first SFLP in column is not where we are now (it appears on regular row).
  if alternate_row and is_bottom_row and z_diff >= zig_row_generator_vector.y:
    row_index -= 1

    # We need to check again if the SFLP index is in the bottom row or not based on combined line from the bottom row.
    regular_line_point = get_projected_SFL_point(SFLP_index - zig_row_index_hop, num_points)

    regular_line_combined_row_index, is_bottom_row = get_row_index_seam_removed(
      regular_line_point,
      combined_row_generator_vector,
      num_tiles_x,
      num_tiles_y,
      tile_x_index,
      tile_y_index,
      col_length
    )

    if not is_bottom_row:
      row_index = 2 * (regular_line_combined_row_index) - 1
      first_SFLP_combined_index = SFLP_index - zig_row_index_hop + (regular_line_combined_row_index) * combined_row_index_hop
      projected_first_SFLP_combined = get_projected_SFL_point(first_SFLP_combined_index, num_points)
      alternate_row = False

  elif alternate_row and not is_bottom_row:
    regular_line_point = get_projected_SFL_point(SFLP_index - zig_row_index_hop, num_points)

    regular_line_combined_row_index, is_bottom_row = get_row_index_seam_removed(
      regular_line_point,
      combined_row_generator_vector,
      num_tiles_x,
      num_tiles_y,
      tile_x_index,
      tile_y_index,
      col_length
    )

    # If the regular line is from the bottom, then check if first SFLP index starts on regular row.
    if is_bottom_row:
      # Grab the first SFLP combined index that begins at the bottom.
      bottom_first_SFLP_combined_index = SFLP_index - zig_row_index_hop - ((col_length - 1) - regular_line_combined_row_index) * combined_row_index_hop
      projected_bottom_first_SFLP_combined = get_projected_SFL_point(bottom_first_SFLP_combined_index, num_points)
      bottom_z_diff = projected_bottom_first_SFLP_combined.y - (tile_y_index * tile_height - 1)

      # If the first SFLP combined index from the bottom is the first SFLP index in column, then set the correct settings.
      if not bottom_z_diff >= zag_row_generator_vector.y:
        combined_row_index = regular_line_combined_row_index
        row_index = (col_length - 1) - 2 * ((col_length - 1) - combined_row_index) - 1
      else:
        is_bottom_row = False


  # Handle if the first SFLP in column is not where we are now (it appears on alternate row).
  elif not alternate_row and is_bottom_row and z_diff >= zag_row_generator_vector.y:
    row_index -= 1

    # We need to check again if the SFLP index is in the bottom row or not based on combined line from the bottom row.
    alternate_line_point = get_projected_SFL_point(SFLP_index - zag_row_index_hop, num_points)

    alternate_line_combined_row_index, is_bottom_row = get_row_index_seam_removed(
      alternate_line_point,
      combined_row_generator_vector,
      num_tiles_x,
      num_tiles_y,
      tile_x_index,
      tile_y_index,
      col_length
    )

    if not is_bottom_row:
      row_index = 2 * (alternate_line_combined_row_index) - 1
      first_SFLP_combined_index = SFLP_index - zag_row_index_hop + (alternate_line_combined_row_index) * combined_row_index_hop
      projected_first_SFLP_combined = get_projected_SFL_point(first_SFLP_combined_index, num_points)
      alternate_row = True

  elif not alternate_row and not is_bottom_row:
    alternate_line_point = get_projected_SFL_point(SFLP_index - zag_row_index_hop, num_points)

    alternate_line_combined_row_index, is_bottom_row = get_row_index_seam_removed(
      alternate_line_point,
      combined_row_generator_vector,
      num_tiles_x,
      num_tiles_y,
      tile_x_index,
      tile_y_index,
      col_length
    )

    # If the regular line is from the bottom, then check if first SFLP index starts on regular row.
    if is_bottom_row:
      # Grab the first SFLP combined index that begins at the bottom.
      bottom_first_SFLP_combined_index = SFLP_index - zag_row_index_hop - ((col_length - 1) - alternate_line_combined_row_index) * combined_row_index_hop
      projected_bottom_first_SFLP_combined = get_projected_SFL_point(bottom_first_SFLP_combined_index, num_points)
      bottom_z_diff = projected_bottom_first_SFLP_combined.y - (tile_y_index * tile_height - 1)

      # If the first SFLP combined index from the bottom is the first SFLP index in column, then set the correct settings.
      if not bottom_z_diff >= zig_row_generator_vector.y:
        combined_row_index = alternate_line_combined_row_index
        row_index = (col_length - 1) - 2 * ((col_length - 1) - combined_row_index) - 1
      else:
        is_bottom_row = False

  # Finally, if still in top row, check if the current top SFLP index can go up once more.
  if not is_bottom_row:
    z_diff_from_top = ((tile_y_index + 1) * tile_height - 1) - projected_first_SFLP_combined.y
    if not alternate_row and z_diff_from_top >= zig_row_generator_vector.y:
      row_index += 1
    elif alternate_row and z_diff_from_top >= zag_row_generator_vector.y:
      row_index += 1
      
  # The column index is half way between zig and zag lines.
  # If on an alternate row, then round up.
  col_index = math.ceil((zig_col_index + zag_col_index) / 2) % row_length

  return row_index, col_index

def get_row_col_index(SFLP_index: int, num_points: int, optimal_index_hops: list, num_tiles_x: int, num_tiles_y: int, ref_one: int, ref_two: int, row_length: int, col_length: int, arrangement_type: SFLPArrangementType):
  """
  Gets the row and column index in a tile given the SFLP arrangement type, Fibonacci Lattice pixel index, the number of pixels, the optimal index hops, and the number of tiles in each direction, and the first element in the first row's SFLP index.

  Args:
    index: The pixel index.
    num_points: The number of pixels.
    optimal_index_hops: The two optimal k-th index hops.
    num_tiles_x: The number of tiles in the x direction.
    num_tiles_y: The number of tiles in the y direction.
    first_element_SFLP_index: The first element of the first column in the SFLP.
    last_element_SFLP_index: The last element of the first column in the SFLP.
    arrangement_type: The arrangement type for the SFLP frame.

  Returns:
    A tuple (x, y) with the row and column index.
  """

  if arrangement_type == SFLPArrangementType.STANDARD:
    return get_row_col_index_standard(SFLP_index, num_points, optimal_index_hops, ref_two, row_length, col_length)
  elif arrangement_type == SFLPArrangementType.SEAM_REMOVED:
    return get_row_col_index_seam_removed(SFLP_index, num_points, optimal_index_hops, num_tiles_x, num_tiles_y, ref_one, ref_two, row_length, col_length)
  elif arrangement_type == SFLPArrangementType.ZIGZAG:
    return get_row_col_index_zigzag(SFLP_index, num_points, optimal_index_hops, num_tiles_x, num_tiles_y, ref_one, ref_two, row_length, col_length)
  else:
    raise ValueError("Arrangement type is not supported!")

# endregion non-GPU compatible

# region GPU compatible

def get_tile_index_gpu(projected_points: cp.ndarray, num_tiles_x: int, num_tiles_y: int):
    """
    GPU-compatible version of get_tile_index for multiple projected points.

    Args:
        projected_points: (N, 2) or (N, 3) CuPy array of (theta, z) or (theta, z, _)
        num_tiles_x: Number of tiles in the theta direction
        num_tiles_y: Number of tiles in the z direction

    Returns:
        tile_x_indices: (N,) CuPy array
        tile_y_indices: (N,) CuPy array
    """
    theta, z = projected_points

    tile_x = cp.floor((theta + 180.0) * (num_tiles_x / 360.0)).astype(cp.int32)
    tile_y = cp.floor((z + 1.0) * (num_tiles_y / 2.0)).astype(cp.int32)

    tile_x = cp.clip(tile_x, 0, num_tiles_x - 1)
    tile_y = cp.clip(tile_y, 0, num_tiles_y - 1)

    return tile_x, tile_y

def get_row_index_seam_removed_gpu(
    projected_pixel_centers: cp.ndarray,  # (N, 2) array: (x, y)
    col_generator_vector: Tuple[float, float],          # tuple (x, y), scalars
    num_tiles_x: int,
    num_tiles_y: int,
    tile_x_indices: cp.ndarray,           # (N,)
    tile_y_indices: cp.ndarray,           # (N,)
    col_length: int
):
    """
    GPU-parallelized get_row_index for multiple inputs,
    with col_generator_vector as tuple (x, y).
    """
    tile_height = 2.0 / num_tiles_y
    tile_width = 360.0 / num_tiles_x

    theta, z = projected_pixel_centers

    left_oriented = col_generator_vector[0] < 0
    left_oriented = cp.bool_(left_oriented)

    abs_d_theta = abs(col_generator_vector[0])
    abs_d_z = abs(col_generator_vector[1])

    z_diff = z - (tile_y_indices * tile_height - 1)

    theta_diff = cp.where(
        left_oriented,
        ((tile_x_indices + 1) * tile_width - 180.0) - theta,
        theta - (tile_x_indices * tile_width - 180.0)
    )

    hops_horizontal = cp.floor(theta_diff / abs_d_theta)
    hops_vertical = cp.floor(z_diff / abs_d_z)

    is_bottom_row = hops_horizontal >= hops_vertical

    z_diff_to_top = ((tile_y_indices + 1) * tile_height - 1.0) - z
    top_row_index = cp.floor(z_diff_to_top / abs_d_z).astype(cp.int32)

    bottom_row_index = hops_vertical.astype(cp.int32)
    bottom_row_index = (col_length - 1) - bottom_row_index

    row_index = cp.where(is_bottom_row, bottom_row_index, top_row_index)

    return row_index.astype(cp.int32), is_bottom_row

def get_row_index_gpu(
    SFLP_indices: cp.ndarray,  # (N,) array
    col_indices: cp.ndarray,
    last_element_SFLP_index: int,
    row_index_hop: int,
    col_index_hop: int,
    col_length: int,
):
    """
    GPU-parallelized get_row_index for STANDARD arrangement for multiple inputs.

    Args:
      SFLP_indices: The pixel indices to get row and column indices for.
      col_indices: The column index of each pixel.
      last_element_SFLP_index: The SFLP index of the last element in the first column.
      row_index_hop: The SFLP index hop from one element in a row to the next.
      col_index_hop: The SFLP index hop from one element in a column to the next.
      col_length: The length of the columns (number of rows).

    """

    bottom_row_index = col_indices * row_index_hop + last_element_SFLP_index

    row_index = (SFLP_indices - bottom_row_index) // col_index_hop

    index_from_bottom = (col_length - 1) - row_index

    index_from_top = (-1 * row_index) - 1

    row_index = cp.where(row_index >= 0, index_from_bottom, index_from_top).astype(cp.int32)

    return row_index

def get_col_index_gpu(
    first_SFLP_index_in_col: cp.ndarray,         # (N,)
    reference_element_SFLP_index: cp.ndarray,    # (N,)
    first_index_hop: int,
    second_index_hop: int,
    col_generator_vector: Tuple[float, float]                   # tuple (x, y)
):
    """
    GPU-parallelized get_col_index for multiple inputs,
    with col_generator_vector as tuple (x, y).
    """
    multiplier = reference_element_SFLP_index - first_SFLP_index_in_col
    if col_generator_vector[0] > 0:
        multiplier = -multiplier

    if first_index_hop > second_index_hop:
        prev_fb_number = first_index_hop - second_index_hop
        larger_fb_number = first_index_hop
    else:
        prev_fb_number = second_index_hop - first_index_hop
        larger_fb_number = second_index_hop


    mod_larger = multiplier % larger_fb_number

    col_index = ((prev_fb_number % larger_fb_number) * mod_larger) % larger_fb_number

    return col_index.astype(cp.int32)

def get_zag_col_index_gpu(
    first_SFLP_index_in_col: cp.ndarray,      # (N,)
    reference_element_SFLP_index: cp.ndarray, # (N,)
    zag_row_index_hop: int,
    zig_row_index_hop: int,
    row_generator_vector: Tuple[float, float]                # tuple (x, y)
):
    """
    GPU-parallelized version of the zag column index calculation.

    Args:
        first_SFLP_index_in_col: (N,) CuPy array
        reference_element_SFLP_index: (N,) CuPy array
        zag_row_index_hop: int
        zig_row_index_hop: int
        row_generator_vector: tuple (x, y)

    Returns:
        (N,) CuPy array of column indices
    """
    multiplier = reference_element_SFLP_index - first_SFLP_index_in_col
    if row_generator_vector[0] > 0:
        multiplier = -multiplier

    larger_fb_number = zag_row_index_hop + zig_row_index_hop

    col_index = ((larger_fb_number % zag_row_index_hop) * (multiplier % zag_row_index_hop)) % zag_row_index_hop

    return col_index.astype(cp.int32)

def get_row_col_index_gpu(SFLP_indices: cp.ndarray, num_points: int, optimal_index_hops: list, last_element_SFLP_index: int, row_length: int, col_length: int):
  """
  GPU-compatible function that gets the row and column index in a frame that is in the STANDARD arrangement,
  given the Fibonacci Lattice pixel indicies, the number of pixels, the optimal index hops,
  and the reference element's SFLP index and column index.

  Args:
    indices: The pixel indices to get row and column indices for.
    num_points: The number of pixels.
    optimal_index_hops: The two optimal k-th index hops.
    last_element_SFLP_index: The SFLP index of the last element in the first column.
    row_length, col_length: row and column length of the frame/image.

  Returns:
    A tuple (x, y) with the row and column index.
  """

  optimal_index_hops = sorted(optimal_index_hops, key=lambda x: x[3])
  row_index_hop = int(optimal_index_hops[0][0])
  col_index_hop = int(optimal_index_hops[1][0])

  col_generator_vector = get_projected_generator_vector(col_index_hop, num_points)

  col_generator_vector = (col_generator_vector.x, col_generator_vector.y)

  col_indices = get_col_index_gpu(
    SFLP_indices,
    last_element_SFLP_index,
    row_index_hop,
    col_index_hop,
    col_generator_vector,
  )

  row_indices = get_row_index_gpu(
    SFLP_indices,
    col_indices,
    last_element_SFLP_index,
    row_index_hop,
    col_index_hop,
    col_length,
  )

  return cp.stack([row_indices, col_indices], axis=1)

def get_row_col_index_seam_removed_gpu(SFLP_indices: cp.ndarray, num_points: int, optimal_index_hops: list, num_tiles_x: int, num_tiles_y: int, first_element_SFLP_index: int, last_element_SFLP_index: int, row_length: int, col_length: int):
  """
  GPU-compatible function that gets the row and column index in a tile that is in the seam-removed arrangement,
  given the Fibonacci Lattice pixel indicies, the number of pixels, the optimal index hops,
  and the number of tiles in each direction, and the first element in the first row's SFLP index.

  Args:
    indices: The pixel indices to get row and column indices for.
    num_points: The number of pixels.
    optimal_index_hops: The two optimal k-th index hops.
    num_tiles_x: The number of tiles in the x direction.
    num_tiles_y: The number of tiles in the y direction.
    first_element_SFLP_index: The first element of the first column in the SFLP.
    last_element_SFLP_index: The last element of the first column in the SFLP.
    row_length, col_length: row and column length of the tile/image.

  Returns:
    A tuple (x, y) with the row and column index.
  """

  optimal_index_hops = sorted(optimal_index_hops, key=lambda x: x[3])
  row_index_hop = int(optimal_index_hops[0][0])
  col_index_hop = int(optimal_index_hops[1][0])

  projected_pixel_centers = get_SFL_projected_gpu(SFLP_indices, num_points)
  col_generator_vector = get_projected_generator_vector(col_index_hop, num_points)

  col_generator_vector = (col_generator_vector.x, col_generator_vector.y)

  tile_x_index, tile_y_index = get_tile_index_gpu(projected_pixel_centers, num_tiles_x, num_tiles_y)

  row_indices, is_bottom_row = get_row_index_seam_removed_gpu(
    projected_pixel_centers,
    col_generator_vector,
    num_tiles_x,
    num_tiles_y,
    tile_x_index,
    tile_y_index,
    col_length
  )

  first_SFLP_index = SFLP_indices - (col_length - 1 - row_indices) * col_index_hop

  reference_element_SFLP_index = cp.where(
    is_bottom_row,
    last_element_SFLP_index,
    first_element_SFLP_index
  )

  col_indices = get_col_index_gpu(
    first_SFLP_index,
    reference_element_SFLP_index,
    col_index_hop,
    row_index_hop,
    col_generator_vector,
  )

  return cp.stack([row_indices, col_indices], axis=1)

def get_row_col_index_zigzag_gpu(
    SFLP_indices: cp.ndarray,
    num_points: int,
    optimal_index_hops: list,
    num_tiles_x: int,
    num_tiles_y: int,
    first_element_SFLP_index: int,
    last_element_SFLP_index: int,
    row_length: int,
    col_length: int
) -> cp.ndarray:
    """
    GPU-parallelized version of get_row_col_index_zigzag for multiple SFLP indices.
    Each index's result is computed independently across the GPU.

    Args:
        SFLP_indices: (N,) CuPy array of indices
        num_points: Total number of SFLP points
        optimal_index_hops: The two optimal k-th index hops.
        num_tiles_x, num_tiles_y: tile counts
        first_element_SFLP_index: The SFLP index of first element of first column in the pixel arrangement table.
        last_element_SFLP_index: The SFLP index of last element of first column in the pixel arrangement table.
        row_length, col_length: row and column length of the tile/image.

    Returns:
        (N, 2) CuPy array of (row_idx, col_idx) for each SFLP index.
    """
    N = SFLP_indices.shape[0]

    # Sort hops
    optimal_index_hops = sorted(optimal_index_hops, key=lambda x: x[3])
    zig_row_index_hop = int(optimal_index_hops[0][0])
    zag_row_index_hop = int(optimal_index_hops[1][0])
    col_index_hop = abs(zag_row_index_hop - zig_row_index_hop)
    combined_row_index_hop = zig_row_index_hop + zag_row_index_hop

    # Projected center points
    projected = get_SFL_projected_gpu(SFLP_indices, num_points)  # shape (N, 3)

    combined_vec = get_projected_generator_vector(combined_row_index_hop, num_points)
    zig_vec = get_projected_generator_vector(zig_row_index_hop, num_points)
    zag_vec = get_projected_generator_vector(zag_row_index_hop, num_points)
    combined_vec = (combined_vec.x, combined_vec.y)
    zig_vec = (zig_vec.x, zig_vec.y)
    zag_vec = (zag_vec.x, zag_vec.y)

    tile_x, tile_y = get_tile_index_gpu(projected, num_tiles_x, num_tiles_y)  # (N,), (N,)
    tile_height = 2.0 / num_tiles_y

    # Top row alternate check
    top_row_alt = (first_element_SFLP_index - last_element_SFLP_index) % combined_row_index_hop != 0

    combined_idx, is_bot = get_row_index_seam_removed_gpu(
        projected, combined_vec, num_tiles_x, num_tiles_y, tile_x, tile_y, col_length
    )

    ref_idx = cp.where(is_bot, last_element_SFLP_index, first_element_SFLP_index)
    first_combined = cp.where(
        is_bot,
        SFLP_indices - ((col_length - 1) - combined_idx) * combined_row_index_hop,
        SFLP_indices + combined_idx * combined_row_index_hop
    )

    zig_col = get_col_index_gpu(first_combined, ref_idx, col_index_hop, zig_row_index_hop, zig_vec)
    zag_col = get_zag_col_index_gpu(first_combined, ref_idx, zag_row_index_hop, zig_row_index_hop, zag_vec)

    adjust_1 = (is_bot & (zig_col == 0) & (first_combined > ref_idx)).astype(cp.int32)
    adjust_2 = (~is_bot & top_row_alt)

    zig_row_index_hop_32 = cp.int32(zig_row_index_hop)
    zig_col += adjust_1 * zig_row_index_hop_32
    zag_col -= adjust_2.astype(cp.int32)

    alt_row = (zig_col + zag_col) % 2 == 1

    row = cp.where(
        is_bot,
        (col_length - 1) - 2 * ((col_length - 1) - combined_idx),
        2 * combined_idx
    )

    theta, z = get_SFL_projected_gpu(first_combined, num_points)
    z_diff = z - (tile_y * tile_height - 1)

    original_is_bot = cp.copy(is_bot)
    original_alt_row = cp.copy(alt_row)

    # ===
    # Complex logic implemented for GPU, read non-GPU method for more documentation.
    # ===

    # Handle if the first SFLP in column is not where we are now (it appears on regular row at bottom).
    check_on_reg_row_from_bottom = original_alt_row & original_is_bot & (z_diff >= zig_vec[1])
    row -= (check_on_reg_row_from_bottom).astype(cp.int32)
    regular_line_points = get_SFL_projected_gpu(SFLP_indices - zig_row_index_hop, num_points)
    regular_line_combined_index, regular_is_bottom_row = get_row_index_seam_removed_gpu(
        regular_line_points, combined_vec, num_tiles_x, num_tiles_y, tile_x, tile_y, col_length
    )

    is_bot = cp.where(
      check_on_reg_row_from_bottom,
      regular_is_bottom_row,
      is_bot
    )
    should_update_bot_reg = check_on_reg_row_from_bottom & (~regular_is_bottom_row)
    row = cp.where(
      should_update_bot_reg,
      2 * (regular_line_combined_index) - 1,
      row
    )
    new_first_combined = SFLP_indices - zig_row_index_hop + (regular_line_combined_index * combined_row_index_hop)
    first_combined = cp.where(
      should_update_bot_reg,
      new_first_combined,
      first_combined
    )
    alt_row = cp.where(
      should_update_bot_reg,
      False,
      alt_row
    )


    # Handle if the first SFLP in column is not where we are now (it appears on regular row at top).
    check_on_reg_row_from_top = original_alt_row & (~original_is_bot)

    regular_line_points = get_SFL_projected_gpu(SFLP_indices - zig_row_index_hop, num_points)
    regular_line_combined_index, regular_is_bottom_row = get_row_index_seam_removed_gpu(
        regular_line_points, combined_vec, num_tiles_x, num_tiles_y, tile_x, tile_y, col_length
    )

    is_bot = cp.where(
      check_on_reg_row_from_top,
      regular_is_bottom_row,
      is_bot
    )

    bottom_first_combined = (
        SFLP_indices - zig_row_index_hop
        - ((col_length - 1) - regular_line_combined_index) * combined_row_index_hop
    )
    proj_bottom_first_combined = get_SFL_projected_gpu(bottom_first_combined, num_points)
    bottom_z_diff = proj_bottom_first_combined[1] - (tile_y * tile_height - 1)

    # Check if we should reset to regular row state
    should_update_top_reg = check_on_reg_row_from_top & regular_is_bottom_row
    on_reg_with_bottom = (bottom_z_diff < zag_vec[1])

    first_combined = cp.where(
      should_update_top_reg & on_reg_with_bottom,
      regular_line_combined_index,
      first_combined
    )
    new_row = (col_length - 1) - 2 * ((col_length - 1) - first_combined) - 1
    row = cp.where(
      should_update_top_reg & on_reg_with_bottom,
      new_row,
      row
    )

    is_bot = cp.where(should_update_top_reg & ~on_reg_with_bottom, False, is_bot)


    # If the first SFLP combined index from the bottom is the first SFLP index in column, then set the correct settings.
    combined_idx = cp.where(
      should_update_top_reg & on_reg_with_bottom,
      regular_line_combined_index,
      combined_idx
    )
    row = cp.where(
      should_update_top_reg & on_reg_with_bottom,
      (col_length - 1) - 2 * ((col_length - 1) - combined_idx) - 1,
      row
    )
    # Else, then we are still in top row (is_bot is false).
    is_bot = cp.where(
      should_update_top_reg & ~on_reg_with_bottom,
      False,
      is_bot
    )


    # Handle if the first SFLP in column is not where we are now (it appears on alternate row).
    check_on_alt_row_from_bottom = (~original_alt_row) & original_is_bot & (z_diff >= zag_vec[1])
    row -= check_on_alt_row_from_bottom.astype(cp.int32)

    alternate_line_points = get_SFL_projected_gpu(SFLP_indices - zag_row_index_hop, num_points)
    alternate_line_combined_index, alternate_is_bottom_row = get_row_index_seam_removed_gpu(
        alternate_line_points, combined_vec, num_tiles_x, num_tiles_y, tile_x, tile_y, col_length
    )

    is_bot = cp.where(
      check_on_alt_row_from_bottom,
      alternate_is_bottom_row,
      is_bot
    )

    should_update_bot_alt = check_on_alt_row_from_bottom & (~alternate_is_bottom_row)

    alt_row_val = 2 * alternate_line_combined_index - 1
    row = cp.where(should_update_bot_alt, alt_row_val, row)

    new_first_combined_alt = SFLP_indices - zag_row_index_hop + (alternate_line_combined_index * combined_row_index_hop)
    first_combined = cp.where(should_update_bot_alt, new_first_combined_alt, first_combined)

    combined_idx = cp.where(should_update_bot_alt, alternate_line_combined_index, combined_idx)

    alt_row = cp.where(should_update_bot_alt, True, alt_row)
    is_bot = cp.where(should_update_bot_alt, False, is_bot)

    # Handle if the first SFLP in column is not where we are now (it appears on alternate row on top)
    check_on_alt_row_from_top = (~original_alt_row) & (~original_is_bot)

    alternate_line_points = get_SFL_projected_gpu(SFLP_indices - zag_row_index_hop, num_points)
    alternate_line_combined_index, alternate_is_bottom_row = get_row_index_seam_removed_gpu(
        alternate_line_points, combined_vec, num_tiles_x, num_tiles_y, tile_x, tile_y, col_length
    )

    is_bot = cp.where(
      check_on_alt_row_from_top,
      alternate_is_bottom_row,
      is_bot
    )

    bottom_first_combined_index = SFLP_indices - zag_row_index_hop - ((col_length - 1 - alternate_line_combined_index) * combined_row_index_hop)
    bottom_proj = get_SFL_projected_gpu(bottom_first_combined_index, num_points)
    bottom_z_diff = bottom_proj[1] - (tile_y * tile_height - 1)

    should_update_top_alt = check_on_alt_row_from_top & alternate_is_bottom_row
    on_alt_with_bottom = (bottom_z_diff < zig_vec[1])

    # Recalculate row and indexes
    first_combined = cp.where(should_update_top_alt & on_alt_with_bottom, alternate_line_combined_index, first_combined)
    new_row = (col_length - 1) - 2 * ((col_length - 1) - alternate_line_combined_index) - 1
    row = cp.where(should_update_top_alt & on_alt_with_bottom, new_row, row)
    
    # alt_row remains false, is_bot may flip
    is_bot = cp.where(should_update_top_alt & ~on_alt_with_bottom, False, is_bot)

    # Top row correction
    theta, z = get_SFL_projected_gpu(first_combined, num_points)
    z_diff_top = ((tile_y + 1) * tile_height - 1) - z
    add_top_zig = (~alt_row & ~is_bot & (z_diff_top >= zig_vec[1])).astype(cp.int32)
    add_top_zag = (alt_row & ~is_bot & (z_diff_top >= zag_vec[1])).astype(cp.int32)

    row += add_top_zig + add_top_zag

    col = cp.ceil((zig_col + zag_col) / 2.0).astype(cp.int32) % row_length

    return cp.stack([row, col], axis=1)

# endregion GPU compatible

#
# endregion SFLP Indexing
#

#
# region GPU-parallelized Functions to For Point Generation/Manipulation
#

def golden_ratio_cp():
  return (1 + cp.sqrt(5)) / 2

def geographic_to_vector_gpu(lat_deg: cp.ndarray, lon_deg: cp.ndarray):
  """
  Convert a set of geographic coordinates to vectors on a sphere using GPU.

  Args:
    lat_deg: The latitudinal component of geographic coordinates, in degrees.
    lon_deg: The longitudinal component of geographic coordinates, in degrees.

  Returns:
    (x, y, z): The x, y and z components of the vector on a unit sphere.

  """
  theta = cp.radians(lon_deg)
  phi = cp.radians(90 - lat_deg)

  x = cp.sin(phi) * cp.cos(theta)
  y = cp.sin(phi) * cp.sin(theta)
  z = cp.cos(phi)

  return x, y, z

def generate_SFL_projected_gpu(num_points: int):
    """
    Generates Spherical Fibonacci Lattice points projected in the unnormalized z-theta plane using GPU.

    Args:
        num_points: Number of vectors to generate (must be odd).

    Returns:
        theta, z: CuPy arrays of shape (num_points), coordinates in the unnormalized z-theta plane.
    """
    if num_points % 2 == 0:
        raise ValueError("Number of points must be odd")

    phi = golden_ratio_cp()
    N = int(np.floor((num_points - 1) / 2))
    i = cp.arange(-N, N + 1)  # shape: (num_points)

    # Calculate theta coordinate
    theta = (i % phi) * 360 / phi

    # Normalize theta to [-180, 180]
    theta = ((theta + 180) % 360) - 180

    # Calculate z coordinate
    z = 2 * i / (2 * N + 1)

    return theta, z

def get_SFL_projected_gpu(indices: cp.ndarray, num_points: int):
    """
    Gets Fibonacci lattice points projected in the unnormalized z-theta plane for a set of indices using GPU.

    Args:
      indices: The indices to get Fibonacci lattice points for.
      num_points: Number of vectors to generate (must be odd).

    Returns:
        theta, z: CuPy arrays of shape (indices), coordinates in the unnormalized z-theta plane.
    """
    if num_points % 2 == 0:
        raise ValueError("Number of points must be odd")

    phi = golden_ratio_cp()
    N = int(np.floor((num_points - 1) / 2))
    i = indices - N

    # Calculate theta coordinate
    theta = (i % phi) * 360 / phi

    # Normalize theta to [-180, 180]
    theta = ((theta + 180) % 360) - 180

    # Calculate z coordinate
    z = 2 * i / (2 * N + 1)

    return theta, z

def get_SFL_geographic_points(indices: cp.ndarray, num_points: int):
    """
    Vectorized GPU version of Fibonacci lattice lat/lon generation in degrees.

    Args:
        indices: CuPy array of indices (e.g. [0, 1, ..., num_points-1])
        num_points: Total number of FB points (must be odd)

    Returns:
        lat: CuPy array in degrees, shape (N,), values in [-90, 90]
        lon: CuPy array in degrees, shape (N,), values in [-180, 180]
    """
    if num_points % 2 == 0:
        raise ValueError("Number of points must be odd")

    N = (num_points - 1) // 2
    i = indices - N

    phi = golden_ratio_cp()

    lat_rad = cp.arcsin(2 * i / (2 * N + 1))  # range: [-π/2, π/2]
    lon_deg = ((i % phi) * 360.0 / phi)      # range: [0, 360)

    # Latitude range: [-90, 90]
    lat_deg = cp.degrees(lat_rad)

    # Longitude range: [-180, 180]
    lon_deg = ((lon_deg + 180) % 360) - 180


    return lat_deg, lon_deg

#
# endregion GPU-parallelized Functions to For Point Generation/Manipulation
#

#
# region Pixel Arrangement Functions
#

def generate_SFLP_pixel_arrangement(num_pixels: int, arrangement_type: SFLPArrangementType, padding_size: int = 1, gpu_enabled=True):
  """
  Generates the pixel arrangement for a single frame based on the arrangement type chosen.

  Args:
    num_pixels: The number of pixels.
    arrangement_type: The type of arrangement to use for SFLP.
    padding_size: This value determines the amount of padding of the column length necessary to ensure that the padded column length is divisible by padding_size. Only used for standard arrangement type.
    gpu_enabled: True if using GPU, False otherwise.

  Returns:
    A Pandas DataFrame with the specified pixel arrangement.
  """

  if arrangement_type == SFLPArrangementType.STANDARD:
    return generate_SFLP_pixel_arrangement_standard(num_pixels=num_pixels, padding_size=padding_size, gpu_enabled=gpu_enabled)
  elif arrangement_type == SFLPArrangementType.SEAM_REMOVED:
    return generate_SFLP_pixel_arrangement_seam_removed(num_pixels=num_pixels, gpu_enabled=gpu_enabled)
  elif arrangement_type == SFLPArrangementType.ZIGZAG:
    return generate_SFLP_pixel_arrangement_zigzag(num_pixels=num_pixels, gpu_enabled=gpu_enabled)
  else:
    raise ValueError("Arrangement type is not supported!")

def generate_SFLP_pixel_arrangement_standard(num_pixels: int, padding_size: int, gpu_enabled: bool = True):
  """
  Generates the pixel arrangement that preserve angles for a single frame using spiral lines (standard).

  Args:
    num_pixels: The number of pixels.
    padding_size: The size of the padding to add to the longest column length to ensure all pixels fit in the arrangement.
    gpu_enabled: If CUDA is enabled for accelerated arrangement.
    
  Returns:
    A Pandas DataFrame with the pixel arrangement.
  """

  optimal_index_hops = find_optimal_generator_vectors(num_pixels, SFLPArrangementType.STANDARD, 3 * math.floor(math.sqrt(num_pixels)))
  row_index_hop = optimal_index_hops[0][0]
  col_index_hop = optimal_index_hops[1][0]

  num_columns = col_index_hop
  longest_column_length = math.floor(num_pixels / col_index_hop)

  padded_amount = (padding_size - (longest_column_length % padding_size))

  longest_column_length = longest_column_length + padded_amount

  if gpu_enabled:
    rows, cols = cp.meshgrid(cp.arange(longest_column_length), cp.arange(num_columns), indexing='ij')

    pixel_indices = (longest_column_length - 1 - rows) * col_index_hop + cols * row_index_hop
    
    fallback_indices = cols * row_index_hop - ((rows + 1) * col_index_hop)
    final_padded_indices = (cols * row_index_hop % col_index_hop)

    mask_too_large = pixel_indices >= num_pixels
    pixel_indices = cp.where(mask_too_large, fallback_indices, pixel_indices)

    mask_still_invalid = (pixel_indices < 0) & mask_too_large
    pixel_indices = cp.where(mask_still_invalid, final_padded_indices, pixel_indices)

    boundary = cp.sum(mask_too_large, axis=0)
    half_pad = padded_amount // 2

    start_pad_row = boundary - half_pad
    padding_mask = (rows >= start_pad_row) & (rows < boundary)

    edge_values = pixel_indices[boundary, cp.arange(num_columns)]

    final_indices = cp.where(padding_mask, edge_values[cols], pixel_indices)

    df_filled = pd.DataFrame(cp.asnumpy(final_indices)).astype(int)

  else:
    rows, cols = np.meshgrid(np.arange(longest_column_length), np.arange(num_columns), indexing='ij')

    pixel_indices = (longest_column_length - 1 - rows) * col_index_hop + cols * row_index_hop

    fallback_indices = cols * row_index_hop - ((rows + 1) * col_index_hop)
    final_padded_indices = (cols * row_index_hop % col_index_hop)

    mask_too_large = pixel_indices >= num_pixels
    pixel_indices = np.where(mask_too_large, fallback_indices, pixel_indices)

    mask_still_invalid = (pixel_indices < 0) & mask_too_large
    pixel_indices = np.where(mask_still_invalid, final_padded_indices, pixel_indices)

    boundary = np.argmax(mask_too_large == True, axis=0)
    boundary = np.where(np.all(~mask_too_large, axis=0), longest_column_length, boundary)

    half_pad = padded_amount // 2
    start_pad_row = boundary - half_pad
    padding_mask = (rows >= start_pad_row) & (rows < boundary)

    safe_boundary = np.clip(boundary, 0, longest_column_length - 1)
    edge_values = pixel_indices[safe_boundary, np.arange(num_columns)]

    final_indices = np.where(padding_mask, edge_values[cols], pixel_indices)
    df_filled = pd.DataFrame(final_indices).astype(int)

  return df_filled

def generate_SFLP_pixel_arrangement_seam_removed(num_pixels: int, gpu_enabled: bool = True):
  """
  Generates the pixel arrangement for a single frame using the seam-removed layout.

  Args:
    num_pixels: The number of pixels.
    gpu_enabled: If CUDA is enabled for accelerated arrangement.
    
  Returns:
    A Pandas DataFrame with the pixel arrangement.
  """

  if gpu_enabled:
    theta, z = generate_SFL_projected_gpu(num_pixels)
    
    theta_np = cp.asnumpy(theta)
    z_np = cp.asnumpy(z)
  else:
    theta = []
    z = []
    for i in range(num_pixels):
      point = get_projected_SFL_point(i, num_pixels)
      theta.append(point.x)
      z.append(point.y)
    
    theta_np = np.asarray(theta)
    z_np = np.asarray(z)

  projected_pixel_centers = {i: Point(theta_val, z_val) for i, (theta_val, z_val) in enumerate(zip(theta_np, z_np))}

  # Get the two best optimal k hops for developing the array.
  optimal_index_hops = find_optimal_generator_vectors(num_pixels, k_max=3 * math.floor(math.sqrt(num_pixels)))
  optimal_index_hops = sorted(optimal_index_hops, key=lambda x: x[3])
  row_index_hop = optimal_index_hops[1][0]

  generator_vector = get_projected_generator_vector(k_index=row_index_hop, num_points=num_pixels)
  is_left_oriented = generator_vector.x < 0

  # Generate the arrays of pixels needed for each the frame.
  sorted_pixel_centers = sorted(projected_pixel_centers.items(), key=lambda x: x[1].y)

  bottom_row_pixel_centers = {}
  top_row_pixel_centers = {}

  for pixel_index, pixel_center in sorted_pixel_centers:
    if pixel_index - row_index_hop not in projected_pixel_centers:
      bottom_row_pixel_centers[pixel_index] = pixel_center
    elif pixel_index + row_index_hop not in projected_pixel_centers:
      num_hops = math.floor(pixel_index / row_index_hop)

      if is_left_oriented and pixel_center.x / 180 - generator_vector.x * num_hops / 180 > 1:
        top_row_pixel_centers[pixel_index] = pixel_center
      elif not is_left_oriented and pixel_center.x / 180 - generator_vector.x * num_hops / 180 < -1:
        top_row_pixel_centers[pixel_index] = pixel_center

  if is_left_oriented:
    sorted_bottom_row_pixels = sorted(bottom_row_pixel_centers.items(), key=lambda x: x[1].x)
    sorted_top_row_pixels = sorted(top_row_pixel_centers.items(), key=lambda x: x[1].x)
  else:
    sorted_bottom_row_pixels = sorted(bottom_row_pixel_centers.items(), key=lambda x: x[1].x, reverse=True)
    sorted_top_row_pixels = sorted(top_row_pixel_centers.items(), key=lambda x: x[1].x, reverse=True)

  arranged_pixels_bottom_row = {}
  arranged_pixels_top_row = {}

  for pixel_index, pixel_center in sorted_bottom_row_pixels:
    arranged_pixels_bottom_row[pixel_index] = [pixel_index]
    curr_index = pixel_index + row_index_hop
    prev_pixel = pixel_center

    while curr_index in projected_pixel_centers:
      pixel = projected_pixel_centers[curr_index]
      if is_left_oriented and prev_pixel.x < pixel.x:
        break
      elif not is_left_oriented and prev_pixel.x > pixel.x:
        break
      prev_pixel = pixel
      arranged_pixels_bottom_row[pixel_index].append(curr_index)
      curr_index += row_index_hop


  for pixel_index, pixel_center in sorted_top_row_pixels:
    arranged_pixels_top_row[pixel_index] = [pixel_index]
    curr_index = pixel_index - row_index_hop
    prev_pixel = pixel_center

    while curr_index in projected_pixel_centers:
      pixel = projected_pixel_centers[curr_index]
      if is_left_oriented and prev_pixel.x > pixel.x:
        break
      elif not is_left_oriented and prev_pixel.x < pixel.x:
        break
      prev_pixel = pixel
      arranged_pixels_top_row[pixel_index].append(curr_index)
      curr_index -= row_index_hop


  bottom_row_keys_list = list(arranged_pixels_bottom_row.keys())
  top_row_keys_list = list(arranged_pixels_top_row.keys())

  num_rows = len(bottom_row_keys_list)
  longest_bottom_row_length = max(len(arranged_pixels_bottom_row[k]) for k in bottom_row_keys_list)

  if len(top_row_keys_list) > 0:
    longest_top_row_length = max(len(arranged_pixels_top_row[top_row_keys_list[i]]) + len(arranged_pixels_bottom_row[bottom_row_keys_list[i]]) for i, k in enumerate(top_row_keys_list))
  else:
    longest_top_row_length = 0

  longest_row_length = max(longest_top_row_length, longest_bottom_row_length)

  padded_array = np.full((num_rows, longest_row_length), np.nan)

  padded_array = np.full((num_rows, longest_row_length), np.nan)

  # Fill in bottom rows
  for i, k in enumerate(bottom_row_keys_list):
      row = arranged_pixels_bottom_row[k]
      padded_array[i, :len(row)] = row

      padded_array[i, len(row):] = row[-1]
      

  # Overwrite with reversed top row values (from the right)
  for i, k in enumerate(top_row_keys_list):
      top_row = arranged_pixels_top_row.get(k)
      if top_row:
          padded_array[i, -len(top_row):] = top_row[::-1]

  rotated_array = np.rot90(padded_array, k=1)

  df_filled = pd.DataFrame(rotated_array).fillna(-1).astype(int)

  return df_filled

def generate_SFLP_pixel_arrangement_zigzag(num_pixels: int, gpu_enabled: bool = True):
  """
  Generates the pixel arrangement for a single frame zig-zagging between spiral lines.

  Args:
    num_pixels: The number of pixels.
    gpu_enabled: If CUDA is enabled for accelerated arrangement.

  Returns:
    A Pandas DataFrame with the pixel arrangement.
  """
  if gpu_enabled:
    theta, z = generate_SFL_projected_gpu(num_pixels)
    
    theta_np = cp.asnumpy(theta)
    z_np = cp.asnumpy(z)
  else:
    theta = []
    z = []
    for i in range(num_pixels):
      point = get_projected_SFL_point(i, num_pixels)
      theta.append(point.x)
      z.append(point.y)
    
    theta_np = np.asarray(theta)
    z_np = np.asarray(z)

  projected_pixel_centers = {i: Point(theta_val, z_val) for i, (theta_val, z_val) in enumerate(zip(theta_np, z_np))}

  # Get the two best optimal k hops for developing the array.
  optimal_index_hops = find_optimal_generator_vectors(num_pixels, k_max=3 * math.floor(math.sqrt(num_pixels)))
  optimal_index_hops = sorted(optimal_index_hops, key=lambda x: x[3])
  column_first_index_hop = optimal_index_hops[0][0]
  column_second_index_hop = optimal_index_hops[1][0]
  row_index_hop = abs(column_second_index_hop - column_first_index_hop)
  combined_generator_index_hop = column_first_index_hop + column_second_index_hop

  combined_generator_vector = get_projected_generator_vector(k_index=combined_generator_index_hop, num_points=num_pixels)
  is_left_oriented = combined_generator_vector.x < 0

  # Generate the arrays of pixels needed for each the frame.
  sorted_pixel_centers = sorted(projected_pixel_centers.items(), key=lambda x: x[1].x)

  bottom_row_pixel_centers = {}
  top_row_pixel_centers = {}

  for pixel_index, pixel_center in sorted_pixel_centers:
    if pixel_index - column_second_index_hop not in projected_pixel_centers:
      bottom_row_pixel_centers[pixel_index] = pixel_center
    elif pixel_index + column_second_index_hop not in projected_pixel_centers:
      num_hops = math.floor(pixel_index / combined_generator_index_hop)

      if is_left_oriented and pixel_center.x - combined_generator_vector.x * num_hops > 180:
        top_row_pixel_centers[pixel_index] = pixel_center
      elif not is_left_oriented and pixel_center.x - combined_generator_vector.x * num_hops < -180:
        top_row_pixel_centers[pixel_index] = pixel_center

  if is_left_oriented:
    sorted_bottom_row_pixels = sorted(bottom_row_pixel_centers.items(), key=lambda x: x[1].x)
  else:
    sorted_bottom_row_pixels = sorted(bottom_row_pixel_centers.items(), key=lambda x: x[1].x, reverse=True)

  # Extract all of the pixels on the bottom row that start a zig-zag line.
  first_bottom_row_pixel = sorted_bottom_row_pixels[0][0]

  prev_pixel = projected_pixel_centers[first_bottom_row_pixel]
  curr_pixel_index = first_bottom_row_pixel
  curr_spiral_index = first_bottom_row_pixel + row_index_hop
  
  extracted_bottom_row_pixels = [(curr_pixel_index, False)]
  
  while curr_pixel_index in bottom_row_pixel_centers:
    candidate_pixel_index = curr_spiral_index % combined_generator_index_hop

    alternate = False
    if candidate_pixel_index >= column_second_index_hop:
      candidate_pixel_index -= column_second_index_hop
      alternate = True

    curr_pixel_index = candidate_pixel_index
    curr_pixel = projected_pixel_centers[curr_pixel_index]

    if is_left_oriented and curr_pixel.x < prev_pixel.x:
      break
    elif not is_left_oriented and curr_pixel.x > prev_pixel.x:
      break

    extracted_bottom_row_pixels.append((curr_pixel_index, alternate))

    curr_spiral_index += row_index_hop
    prev_pixel = curr_pixel


  arranged_pixels_bottom_row = {}
  last_incomplete_bottom_row_pixel = -1 # Used to figure out what bottom row pixel connects to the first top row pixel.
  is_last_incomplete_alternate = False

  # Arrange the pixels in each column for the bottom row.
  for idx, (pixel_index, alternate) in enumerate(extracted_bottom_row_pixels):
    arranged_pixels_bottom_row[pixel_index] = [pixel_index]
    original_alternate = alternate

    if not alternate:
      curr_index = pixel_index + column_first_index_hop
      alternate = True
    else:
      curr_index = pixel_index + column_second_index_hop
      alternate = False
    
    prev_pixel = projected_pixel_centers[pixel_index]
    

    while curr_index in projected_pixel_centers:
      curr_pixel = projected_pixel_centers[curr_index]

      # Only check on when not on the alternate so that the line for the combined vector is used to determine
      # whether the cross over the lowest theta value has occurred.
      if alternate is original_alternate:

        if is_left_oriented and curr_pixel.x - prev_pixel.x > 180:
          last_incomplete_bottom_row_pixel = pixel_index
          is_last_incomplete_alternate = original_alternate
          break
        elif not is_left_oriented and curr_pixel.x - prev_pixel.x < -180:
          last_incomplete_bottom_row_pixel = pixel_index
          is_last_incomplete_alternate = original_alternate
          break
        
        prev_pixel = curr_pixel

      arranged_pixels_bottom_row[pixel_index].append(curr_index)
      
      if not alternate:
        curr_index += column_first_index_hop
        alternate = True
      else:
        curr_index += column_second_index_hop
        alternate = False

  # Extract all pixels in the top row that start a zig-zag line.
  last_top_row_pixel = last_incomplete_bottom_row_pixel + math.floor((num_pixels - last_incomplete_bottom_row_pixel) / combined_generator_index_hop) * combined_generator_index_hop
  is_top_alternate = is_last_incomplete_alternate

  if last_top_row_pixel <= num_pixels - column_second_index_hop:
    is_top_alternate = not is_last_incomplete_alternate
    if is_last_incomplete_alternate:      
      last_top_row_pixel += column_second_index_hop
    else:
      last_top_row_pixel += column_first_index_hop
  
  if last_top_row_pixel not in top_row_pixel_centers:
    top_row_pixel_centers[last_top_row_pixel] = projected_pixel_centers[last_top_row_pixel]
  
  
  prev_pixel = projected_pixel_centers[last_top_row_pixel]
  curr_pixel_index = last_top_row_pixel
  curr_spiral_index = last_top_row_pixel - row_index_hop

  extracted_top_row_pixels = [(curr_pixel_index, is_top_alternate)]

  while curr_pixel_index in top_row_pixel_centers:

    inverse_spiral_index = (num_pixels - 1) - curr_spiral_index

    candidate_inverse_pixel_index = inverse_spiral_index % combined_generator_index_hop

    alternate = is_top_alternate

    if alternate and candidate_inverse_pixel_index >= column_second_index_hop:
      candidate_inverse_pixel_index -= column_second_index_hop
      alternate = not is_top_alternate
    elif not alternate and candidate_inverse_pixel_index >= column_first_index_hop:
      candidate_inverse_pixel_index -= column_first_index_hop
      alternate = not is_top_alternate

    curr_pixel_index = (num_pixels - 1) - candidate_inverse_pixel_index
    curr_pixel = projected_pixel_centers[curr_pixel_index]

    # Check to see if this the last pixel center to add (the one that matches up with the first bottom row pixel center).
    if curr_pixel_index not in top_row_pixel_centers:
      if (not alternate and (curr_pixel_index - first_bottom_row_pixel) % combined_generator_index_hop == 0
      or alternate and (curr_pixel_index - first_bottom_row_pixel - column_first_index_hop) % combined_generator_index_hop == 0):
        # Append the pixel index for this row, and if it is on an alternate row or not.
        extracted_top_row_pixels.append((curr_pixel_index, alternate))

      break

    # Append the pixel index for this row, and if it is on an alternate row or not.
    extracted_top_row_pixels.append((curr_pixel_index, alternate))

    curr_spiral_index -= row_index_hop
    prev_pixel = curr_pixel


  arranged_pixels_top_row = {}

  # Arrange the pixels in each column for the top row.
  for idx, (pixel_index, alternate) in enumerate(reversed(extracted_top_row_pixels)):
    arranged_pixels_top_row[pixel_index] = [pixel_index]

    if not alternate:
      curr_index = pixel_index - column_second_index_hop
      alternate = True
    else:
      curr_index = pixel_index - column_first_index_hop
      alternate = False
    
    prev_pixel = projected_pixel_centers[pixel_index]

    # Determine whether the connected bottom row is alternate to check when the cross over the lowest theta value occurs.
    bottom_alternate = extracted_bottom_row_pixels[idx][1]

    while curr_index in projected_pixel_centers:
      curr_pixel = projected_pixel_centers[curr_index]


      # Only check on when not on the alternate so that the line for the combined vector is used to determine
      # whether the cross over the lowest theta value has occurred.
      if alternate is bottom_alternate:

        if is_left_oriented and curr_pixel.x - prev_pixel.x < -180:
          arranged_pixels_top_row[pixel_index].pop()
          break
        elif not is_left_oriented and curr_pixel.x - prev_pixel.x > 180:
          arranged_pixels_top_row[pixel_index].pop()
          break
        
        prev_pixel = curr_pixel

      arranged_pixels_top_row[pixel_index].append(curr_index)
      
      if not alternate:
        curr_index -= column_second_index_hop
        alternate = True
      else:
        curr_index -= column_first_index_hop
        alternate = False

  # Line up the bottom row and top row to make the entire table.
  bottom_row_keys_list = list(arranged_pixels_bottom_row.keys())
  top_row_keys_list = list(arranged_pixels_top_row.keys())

  num_rows = len(bottom_row_keys_list)
  longest_bottom_row_length = max(len(arranged_pixels_bottom_row[k]) for k in bottom_row_keys_list)
  longest_top_row_length = max(len(arranged_pixels_top_row[top_row_keys_list[i]]) + len(arranged_pixels_bottom_row[bottom_row_keys_list[i]]) for i, k in enumerate(top_row_keys_list))

  longest_row_length = max(longest_top_row_length, longest_bottom_row_length)

  padded_array = np.full((num_rows, longest_row_length), np.nan)

  # Fill in bottom rows
  for i, k in enumerate(bottom_row_keys_list):
      row = arranged_pixels_bottom_row[k]
      padded_array[i, :len(row)] = row

      padded_array[i, len(row):] = row[-1]
      

  # Overwrite with reversed top row values (from the right)
  for i, k in enumerate(top_row_keys_list):
      top_row = arranged_pixels_top_row.get(k)
      if top_row:
          padded_array[i, -len(top_row):] = top_row[::-1]

  # Rotate the array to put it in the correct orientation.
  rotated_array = np.rot90(padded_array, k=1)

  df_filled = pd.DataFrame(rotated_array).fillna(-1).astype(int)

  return df_filled

def get_CubeMap_face_layout():
  face_names = {
    (0, 0): "X=+1",
    (0, 1): "Y=+1",
    (0, 2): "X=-1",
    (1, 0): "Z=-1",
    (1, 1): "Y=-1",
    (1, 2): "Z=+1"
  }

  return face_names

def generate_CubeMap_pixel_arrangement(num_pixels_horizontal: int, num_pixels_vertical: int) -> pd.DataFrame:
  """
  Generates the pixel arrangement for each a single CMP/EAC frame.

  Args:
    num_pixels: The number of pixels.
    num_tiles_x: The number of tiles in the x direction.
    num_tiles_y: The number of tiles in the y direction.

  Returns:
    A dictionary of Pandas DataFrames with the pixel arrangement for the CMP/EAC frame.
  """

  height = num_pixels_vertical * 2
  width = num_pixels_horizontal * 3

  # Face layout mapping
  face_names = get_CubeMap_face_layout()

  # Initialize array with 'invalid'
  cube_frame_arr = np.full((height, width), 'invalid', dtype=object)

  # Fill face names
  for (tile_row, tile_col), face_name in face_names.items():
    row_start = tile_row * num_pixels_vertical
    row_end = (tile_row + 1) * num_pixels_vertical
    col_start = tile_col * num_pixels_horizontal
    col_end = (tile_col + 1) * num_pixels_horizontal

    face_rows = np.arange(num_pixels_vertical).reshape(-1, 1)
    face_cols = np.arange(num_pixels_horizontal).reshape(1, -1)


    labels = np.char.add(
        np.char.add(face_name + "_", face_rows.astype(str)),
        np.char.add("_", face_cols.astype(str))
    )

    cube_frame_arr[row_start:row_end, col_start:col_end] = labels

  cube_frame_df = pd.DataFrame(cube_frame_arr)

  return cube_frame_df

#
# endregion Pixel Arrangement Functions
#