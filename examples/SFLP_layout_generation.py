from pathlib import Path
from SFLP_toolkit.utilities.pixel_arrangement_utils import generate_SFLP_pixel_arrangement, SFLPArrangementType
import pandas as pd

output_dir = Path("../output/layouts")
input_dir = Path("../input")
output_dir.mkdir(parents=True, exist_ok=True)

num_points = 1001

arrangement_type = SFLPArrangementType.STANDARD
padding_size = 2

arranged_pixel_indices_df = generate_SFLP_pixel_arrangement(num_points, padding_size=padding_size, arrangement_type=arrangement_type)
arranged_pixel_indices_df.to_csv(output_dir / f"SFLP_arrangement_{arrangement_type.name}-360_No_Tiling-{num_points}.csv", index=False, header=False)

H, W = arranged_pixel_indices_df.shape
pad_H = (H + padding_size - 1) // padding_size * padding_size   # round up to multiple of padding_size
pad_W = (W + padding_size - 1) // padding_size * padding_size   # round up to multiple of padding_size

if pad_H != H or pad_W != W:
    # Create a padded DataFrame
    padded_df = arranged_pixel_indices_df.copy()

    # Pad columns (extend last column to the right)
    if pad_W > W:
        last_col = arranged_pixel_indices_df.iloc[:, -1:]
        for _ in range(W, pad_W):
            padded_df[_] = last_col.values

    # Pad rows (extend last row downward)
    if pad_H > H:
        last_row = padded_df.iloc[-1:, :].copy()
        for _ in range(H, pad_H):
            padded_df = pd.concat([padded_df, last_row], axis=0)

    arranged_pixel_indices_df = padded_df.reset_index(drop=True)

output_path = output_dir / f"SFLP_arrangement_{arrangement_type.name}-360_No_Tiling-{num_points}_padded.csv"

# Save padded CSV
arranged_pixel_indices_df.to_csv(
    output_path,
    index=False, header=False
)

print("SFLP layout saved to:", output_path)

indices_found = {}

for i in range(num_points):
    indices_found[i] = 0

arr = arranged_pixel_indices_df.to_numpy()

for row in arr:
    for val in row:
        if val != -1:
            indices_found[val] += 1

multiple_copies = 0

for val in indices_found:
    if indices_found[val] == 0:
        print(f"{val} was not found in the table!")
    elif indices_found[val] > 1:
        # print(f"{val} was found multiple times in the table!")
        multiple_copies += indices_found[val] - 1

print(multiple_copies)