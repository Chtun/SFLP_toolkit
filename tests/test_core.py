"""Tests for SFLP layout encoding and decoding using CPU."""
import time

from SFLP_toolkit.data_types import Point, GeographicPoint, Vector
from SFLP_toolkit.utilities.pixel_arrangement_utils import *
from SFLP_toolkit.utilities.file_conversion_utils import *

def test_point_creation():
    """Test Point class creation."""
    point = Point(x=200, y=-100)
    assert point.x == 200
    assert point.y == -100

def test_geographic_point_creation():
    """Test GeographicPoint class creation."""
    point = GeographicPoint(lon=45.0, lat=30.0)
    assert point.lon == 45.0
    assert point.lat == 30.0

def test_vector_creation():
    """Test Vector class creation."""
    vector = Vector(x=1.0, y=2.0, z=3.0)
    assert vector.x == 1.0
    assert vector.y == 2.0
    assert vector.z == 3.0

def test_standard_layout_encoding_decoding():
    """Tests that the standard layout for a single frame is generated and decoded properly."""
    num_points_list = [647, 1001, 6401, 12799]
    num_tiles_x = 1
    num_tiles_y = 1

    arrangement_type = SFLPArrangementType.STANDARD

    for num_points in num_points_list:
        arranged_pixels = generate_SFLP_pixel_arrangement(num_points, arrangement_type, gpu_enabled=False)

        optimal_k_hops = find_optimal_generator_vectors(num_points, arrangement_type=arrangement_type)
        optimal_k_hops = sorted(optimal_k_hops, key=lambda x: x[3])
        print(f"Optimal K found for {num_points}.")

        number_right = 0
        col_length = arranged_pixels.shape[0]
        row_length = arranged_pixels.shape[1]
        first_element_SFLP_index = arranged_pixels[0][0]
        last_element_SFLP_index = arranged_pixels[0][col_length - 1]

        
        failed_indices = []

        for SFLP_index in range(num_points):
            row_index, column_index = get_row_col_index(SFLP_index=SFLP_index,
                                                        num_points=num_points,
                                                        optimal_index_hops=optimal_k_hops,
                                                        num_tiles_x=num_tiles_x,
                                                        num_tiles_y=num_tiles_y,
                                                        ref_one=first_element_SFLP_index,
                                                        ref_two=last_element_SFLP_index,
                                                        row_length=row_length,
                                                        col_length=col_length,
                                                        arrangement_type=arrangement_type
                                                        )
            

            if row_index < 0 or row_index >= col_length or column_index < 0 or column_index >= row_length:
                print(f"Row or column index out of bounds for SFLP index {SFLP_index}.")
                print()
                failed_indices.append(SFLP_index)
                continue

            SFLP_index_in_arrangement = arranged_pixels[column_index][row_index]

            if SFLP_index_in_arrangement != SFLP_index:
                print(f"SFLP index {SFLP_index} is not in the arrangement ({SFLP_index_in_arrangement} found instead).")
                print(f"row: {row_index}, column: {column_index}")
                print()
                failed_indices.append(SFLP_index)
            else:
                number_right += 1

        print(failed_indices)

        assert number_right == num_points

        print()

test_standard_layout_encoding_decoding()