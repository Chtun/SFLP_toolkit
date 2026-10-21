"""Tests for SFLP layout encoding and decoding using GPU."""
import time

from SFLP_toolkit.utilities.pixel_arrangement_utils import *
from SFLP_toolkit.utilities.file_conversion_utils import *

def test_standard_layout_encoding_decoding_gpu():
    """Tests that the standard layout for a single frame is generated and decoded properly using GPU-compatible functions."""
    num_points_list = [647, 1001, 12799, 20001, 1000001, 18000001]

    arrangement_type = SFLPArrangementType.STANDARD
    padding_size = 16

    for num_points in num_points_list:
        arranged_pixels = generate_SFLP_pixel_arrangement(num_points, arrangement_type, padding_size=padding_size, gpu_enabled=True)

        optimal_k_hops = find_optimal_generator_vectors(num_points, arrangement_type=arrangement_type)
        optimal_k_hops = sorted(optimal_k_hops, key=lambda x: x[3])
        print(f"Optimal K found for {num_points}.")

        col_length = arranged_pixels.shape[0]
        row_length = arranged_pixels.shape[1]
        last_element_SFLP_index = arranged_pixels[0][col_length - 1]

        SFLP_indices = cp.arange(num_points)

        print(col_length)
        print(row_length)

        start = time.time()

        indices = get_row_col_index_gpu(
            SFLP_indices=SFLP_indices,
            num_points=num_points,
            optimal_index_hops=cp.asarray(optimal_k_hops),
            last_element_SFLP_index=last_element_SFLP_index,
            row_length=row_length,
            col_length=col_length
        )

        end = time.time()
        print("Indexing with GPU took:", end - start, "seconds.")

        print()

        row_indices = cp.asnumpy(indices[:, 0])
        col_indices = cp.asnumpy(indices[:, 1])

        cp_rows = cp.asarray(row_indices)
        cp_cols = cp.asarray(col_indices)
        cp_arranged = cp.asarray(arranged_pixels)
        expected_indices = cp.arange(num_points)
        found_indices = cp_arranged[cp_rows, cp_cols]

        failed_indices = []
        number_right = 0

        mismatch_mask = found_indices != expected_indices
    
        number_right = int(cp.sum(~mismatch_mask))
        
        if cp.any(mismatch_mask):
            failed_indices = expected_indices[mismatch_mask].get().tolist()
            
            for i in failed_indices:
                print(f"SFLP index {i} is not in the arrangement ({found_indices[i]} found instead).")
                print(f"row: {cp_rows[i]}, column: {cp_cols[i]}\n")
        else:
            failed_indices = []

        print(failed_indices)
        print(f"Correct: {number_right}/{num_points}")

        assert number_right == num_points

def test_seam_removed_layout_encoding_decoding_gpu():
    """Tests that the seam-removed layout for a single frame is generated and decoded properly using GPU-compatible functions."""
    num_points_list = [1001, 12799, 20001, 100001]
    num_tiles_x = 1
    num_tiles_y = 1

    arrangement_type = SFLPArrangementType.SEAM_REMOVED

    for num_points in num_points_list:
        arranged_pixels = generate_SFLP_pixel_arrangement(num_points, arrangement_type)

        optimal_k_hops = find_optimal_generator_vectors(num_points)
        optimal_k_hops = sorted(optimal_k_hops, key=lambda x: x[3])
        print(f"Optimal K found for {num_points}.")

        col_length = arranged_pixels.shape[0]
        row_length = arranged_pixels.shape[1]
        first_element_SFLP_index = arranged_pixels[0][0]
        last_element_SFLP_index = arranged_pixels[0][col_length - 1]

        SFLP_indices = cp.arange(num_points)

        start = time.time()

        indices = get_row_col_index_seam_removed_gpu(
            SFLP_indices=SFLP_indices,
            num_points=num_points,
            optimal_index_hops=cp.asarray(optimal_k_hops),
            num_tiles_x=num_tiles_x,
            num_tiles_y=num_tiles_y,
            first_element_SFLP_index=first_element_SFLP_index,
            last_element_SFLP_index=last_element_SFLP_index,
            row_length=row_length,
            col_length=col_length
        )

        end = time.time()
        print("Indexing with GPU took:", end - start, "seconds.")

        print()

        row_indices = cp.asnumpy(indices[:, 0])
        col_indices = cp.asnumpy(indices[:, 1])

        failed_indices = []
        number_right = 0

        for i in range(num_points):
            row_index = row_indices[i]
            col_index = col_indices[i]

            SFLP_index_in_arrangement = arranged_pixels[col_index][row_index]

            if SFLP_index_in_arrangement != i:
                print(f"SFLP index {i} is not in the arrangement ({SFLP_index_in_arrangement} found instead).")
                print(f"row: {row_index}, column: {col_index}")
                print()
                failed_indices.append(i)
            else:
                number_right += 1

        print(failed_indices)
        print(f"Correct: {number_right}/{num_points}")

        assert number_right == num_points


def test_zigzag_layout_encoding_decoding_gpu():
    """Tests that the zigzag layout for a single frame is generated and decoded properly using GPU-compatible functions."""
    num_points_list = [1001, 12799, 20001, 100001]
    num_tiles_x = 1
    num_tiles_y = 1

    arrangement_type = SFLPArrangementType.ZIGZAG

    for num_points in num_points_list:
        arranged_pixels = generate_SFLP_pixel_arrangement(num_points, arrangement_type)

        optimal_k_hops = find_optimal_generator_vectors(num_points)
        optimal_k_hops = sorted(optimal_k_hops, key=lambda x: x[3])
        print(f"Optimal K found for {num_points}.")

        col_length = arranged_pixels.shape[0]
        row_length = arranged_pixels.shape[1]
        first_element_SFLP_index = arranged_pixels[0][0]
        last_element_SFLP_index = arranged_pixels[0][col_length - 1]

        SFLP_indices = cp.arange(num_points)

        start = time.time()

        indices = get_row_col_index_zigzag_gpu(
            SFLP_indices,
            num_points=num_points,
            optimal_index_hops=cp.asarray(optimal_k_hops),
            num_tiles_x=num_tiles_x,
            num_tiles_y=num_tiles_y,
            first_element_SFLP_index=first_element_SFLP_index,
            last_element_SFLP_index=last_element_SFLP_index,
            row_length=row_length,
            col_length=col_length
        )

        end = time.time()
        print("Indexing with GPU took:", end - start, "seconds.")

        print()

        row_indices = cp.asnumpy(indices[:, 0])
        col_indices = cp.asnumpy(indices[:, 1])

        failed_indices = []
        number_right = 0

        for i in range(num_points):
            row_index = row_indices[i]
            col_index = col_indices[i]

            SFLP_index_in_arrangement = arranged_pixels[col_index][row_index]

            if SFLP_index_in_arrangement != i:
                print(f"SFLP index {i} is not in the arrangement ({SFLP_index_in_arrangement} found instead).")
                print(f"row: {row_index}, column: {col_index}")
                print()
                failed_indices.append(i)
            else:
                number_right += 1

        print(failed_indices)
        print(f"Correct: {number_right}/{num_points}")

        assert number_right == num_points

        print()