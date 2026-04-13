#!/usr/bin/env python3
"""
Extract all cardiac structures from XCAT phantom and save as CSVs.
Run this once (or after re-growth) to generate domain + reference anatomy.

Label 23 is NOT aorta — it's body surface/tissue (18.9M voxels, spans entire body).
Great vessels near the heart are label 28 (IVC/SVC/great vessels).
"""
import numpy as np
import csv
import os
from datetime import datetime

PHANTOM_PATH = "/home/molloi-lab/smb_mount/shared_drive/shu_nie/PVAT_Analysis/digital phantoms/vmale50_1600x1400x500_8bit_little_endian_act_1.raw"
NX, NY, NZ = 1600, 1400, 500
VOXEL_CM = 0.02
OUTDIR = os.path.dirname(os.path.abspath(__file__))

# Groups for extraction
# NOTE: label 23 is body surface, NOT aorta. Label 28 = great vessels near heart.
GROUPS = {
    "domain":              [15, 16, 17, 18],   # myocardium — growth domain
    "chambers":            [19, 20, 21, 22],   # blood pools (heart cavities)
    "great_vessels":       [28],               # IVC/SVC/great vessels (near heart)
    "coronary_arteries":   [26],               # existing XCAT coronary arteries
    "pericardium":         [29],               # pericardium (outer heart envelope)
}

# Stride per group
STRIDES = {
    "domain": 1,
    "chambers": 6,
    "great_vessels": 4,
    "coronary_arteries": 2,
    "pericardium": 4,
}


def extract_points(phantom, labels, stride):
    """Extract points for given label set with stride sampling."""
    xs, ys, zs = [], [], []
    label_set = set(labels)
    for k in range(0, NZ, stride):
        for j in range(0, NY, stride):
            for i in range(0, NX, stride):
                if phantom[i, j, k] in label_set:
                    xs.append(round((i + 0.5) * VOXEL_CM, 4))
                    ys.append(round((j + 0.5) * VOXEL_CM, 4))
                    zs.append(round((k + 0.5) * VOXEL_CM, 4))
    return xs, ys, zs


def save_csv(xs, ys, zs, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_cm", "y_cm", "z_cm"])
        for x, y, z in zip(xs, ys, zs):
            writer.writerow([x, y, z])


def main():
    print("Loading phantom...")
    phantom = np.fromfile(PHANTOM_PATH, dtype=np.uint8).reshape((NX, NY, NZ), order='F')

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for group_name, label_list in GROUPS.items():
        stride = STRIDES.get(group_name, 4)
        print(f"Extracting {group_name} (labels {label_list}, stride={stride})...")
        xs, ys, zs = extract_points(phantom, label_list, stride)
        if not xs:
            print(f"  → 0 points, skipping")
            continue
        print(f"  → {len(xs):,} points, "
              f"x=[{min(xs):.2f},{max(xs):.2f}] "
              f"y=[{min(ys):.2f},{max(ys):.2f}] "
              f"z=[{min(zs):.2f},{max(zs):.2f}]")

        ts_path = os.path.join(OUTDIR, f"{group_name}_{ts}.csv")
        latest_path = os.path.join(OUTDIR, f"{group_name}_points.csv")
        save_csv(xs, ys, zs, ts_path)
        save_csv(xs, ys, zs, latest_path)
        print(f"  Saved: {latest_path}")
        print(f"  Saved: {ts_path}")

    print("Done.")


if __name__ == "__main__":
    main()
