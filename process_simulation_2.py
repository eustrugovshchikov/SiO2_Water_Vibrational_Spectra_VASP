#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified simulation processing script.
Command-line arguments:
   1) configuration index (line number in config-out.dat, 1-indexed)
   2) start frame (first simulation step in XDATCAR)
   3) finish frame (last simulation step in XDATCAR)
Outputs:
   output_<config_index>.dat and geo_<config_index>.dat

Performance improvements (I/O unchanged):
  - Stream XDATCAR (no full readlines()).
  - Parse only the 2–3 atom coordinate lines needed per frame.
  - Read only header for lattice + total atom count.
  - Metric tensor for distance.
  - Robust frame alignment by scanning for "configuration" markers.
"""

import numpy as np
import sys
import time


def _wrap_frac_diff(diff_frac: np.ndarray) -> np.ndarray:
    return diff_frac - np.round(diff_frac)


def _metric_tensor(lattice: np.ndarray) -> np.ndarray:
    return lattice @ lattice.T


def compute_distance_frac(f1: np.ndarray, f2: np.ndarray, G: np.ndarray) -> float:
    diff = _wrap_frac_diff(f1 - f2)
    d2 = float(diff @ G @ diff)
    if d2 < 0.0 and d2 > -1e-14:
        d2 = 0.0
    return float(np.sqrt(d2))


def angle_between_points_pbc_frac(f1: np.ndarray, f2: np.ndarray, f3: np.ndarray, lattice: np.ndarray) -> float:
    diff1 = _wrap_frac_diff(f1 - f2)
    diff2 = _wrap_frac_diff(f3 - f2)
    cart1 = diff1 @ lattice
    cart2 = diff2 @ lattice

    dot = float(cart1 @ cart2)
    norm1 = float(np.linalg.norm(cart1))
    norm2 = float(np.linalg.norm(cart2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    cos_angle = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def read_xdatcar_header(filename: str = "XDATCAR"):
    with open(filename, "r") as f:
        if not f.readline():
            raise ValueError("XDATCAR appears empty.")
        scale_line = f.readline()
        if not scale_line:
            raise ValueError("XDATCAR missing scaling factor line.")
        scaling_factor = float(scale_line.strip())

        a_vector = np.fromstring(f.readline(), sep=" ")[:3] * scaling_factor
        b_vector = np.fromstring(f.readline(), sep=" ")[:3] * scaling_factor
        c_vector = np.fromstring(f.readline(), sep=" ")[:3] * scaling_factor
        lattice = np.array([a_vector, b_vector, c_vector], dtype=float)

        # element symbols line
        if not f.readline():
            raise ValueError("XDATCAR missing element symbols line.")
        # counts line
        counts_line = f.readline()
        if not counts_line:
            raise ValueError("XDATCAR missing atom counts line.")
        counts = list(map(int, counts_line.split()))
        total_atoms = int(sum(counts))

        # coordinate type line ("Direct"/"Cartesian")
        if not f.readline():
            raise ValueError("XDATCAR missing coordinate type line.")

    return lattice, total_atoms


def _read_config_line(config_index: int, filename: str = "config-out.dat") -> str:
    if config_index < 1:
        raise ValueError("config_index must be >= 1.")
    current = 0
    with open(filename, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            current += 1
            if current == config_index:
                return s
    raise ValueError("Invalid configuration index.")


def _seek_next_configuration_line(f) -> str:
    """
    Advance file handle until a line containing "configuration" (case-insensitive) is found.
    Returns that line. Raises EOFError if not found.
    """
    for line in f:
        if "configuration" in line.lower():
            return line
    raise EOFError("Reached end of XDATCAR while seeking a configuration line.")


def process_simulation():
    if len(sys.argv) != 4:
        print("Usage: python process_simulation.py <config_index> <start_frame> <finish_frame>")
        sys.exit(1)

    config_index = int(sys.argv[1])
    start_step = int(sys.argv[2])
    finish_step = int(sys.argv[3])

    try:
        config_line = _read_config_line(config_index, "config-out.dat")
    except Exception as e:
        print("Error reading config-out.dat:", e)
        sys.exit(1)

    tokens = config_line.split()
    try:
        num_points = int(tokens[0])
        indices = [int(x) - 1 for x in tokens[1:]]
    except Exception as e:
        print(f"Error parsing configuration on line {config_index}:", e)
        sys.exit(1)

    if num_points not in (2, 3) or len(indices) != num_points:
        print(f"Error: configuration {config_index} is malformed: '{config_line}'")
        sys.exit(1)

    mode = "angle" if num_points == 3 else "bond"
    print(f"Processing configuration {config_index} with mode: {mode} and atom indices: {[i+1 for i in indices]}")

    try:
        lattice, total_atoms = read_xdatcar_header("XDATCAR")
    except Exception as e:
        print("Error reading XDATCAR header:", e)
        sys.exit(1)

    for idx in indices:
        if idx < 0 or idx >= total_atoms:
            print(f"Error: atom index {idx+1} out of range for total_atoms={total_atoms}.")
            sys.exit(1)

    total_steps = finish_step - start_step
    if total_steps <= 0:
        print("Invalid simulation step range.")
        sys.exit(1)

    needed_set = set(indices)

    G = _metric_tensor(lattice)
    measurements = []
    dt = 0.5e-15

    try:
        with open("XDATCAR", "r") as f:
            # Skip the fixed 8-line header region (family assumption)
            for _ in range(8):
                if not f.readline():
                    raise ValueError("XDATCAR ended while reading header.")

            # Anchor to the first configuration line robustly
            _seek_next_configuration_line(f)

            # Skip frames up to start_step
            for _ in range(start_step):
                # discard Natoms coordinate lines
                for _atom in range(total_atoms):
                    if not f.readline():
                        raise EOFError("XDATCAR ended while skipping frames (coordinates).")
                # find next configuration header (skip blank lines safely)
                _seek_next_configuration_line(f)

            # Process frames
            for step in range(total_steps):
                got = {}
                for atom_i in range(total_atoms):
                    line = f.readline()
                    if not line:
                        raise EOFError(f"XDATCAR ended unexpectedly at step {start_step + step}.")
                    if atom_i in needed_set:
                        frac = np.fromstring(line, sep=" ")[:3]
                        if frac.size < 3:
                            raise ValueError(f"Bad coordinate line at step {start_step + step}, atom {atom_i+1}: '{line.strip()}'")
                        got[atom_i] = frac

                if len(got) != len(indices):
                    missing = [i + 1 for i in indices if i not in got]
                    raise ValueError(f"Missing needed atom coords at step {start_step + step}: {missing}")

                if mode == "angle":
                    value = angle_between_points_pbc_frac(got[indices[0]], got[indices[1]], got[indices[2]], lattice)
                else:
                    value = compute_distance_frac(got[indices[0]], got[indices[1]], G)

                measurements.append(value)

                # Move to next configuration line unless this was the last step
                if step != total_steps - 1:
                    _seek_next_configuration_line(f)

    except Exception as e:
        print("Error while processing XDATCAR:", e)
        sys.exit(1)

    if measurements:
        if mode == "angle":
            print(f"Angle at first step for configuration {config_index}: {measurements[0]:.2f} deg")
        else:
            print(f"Bond length at first step for configuration {config_index}: {measurements[0]:.2f} A")

    geo_filename = f"geo_{config_index}.dat"
    try:
        with open(geo_filename, "w") as geo_f:
            geo_f.write("# Step\tMeasurement\n")
            for i, meas in enumerate(measurements):
                geo_f.write(f"{start_step + i}\t{meas:.6e}\n")
        print(f"Geometry data for configuration {config_index} written to {geo_filename}")
    except Exception as e:
        print(f"Error writing {geo_filename}:", e)

    N_points = len(measurements)
    fgrid = np.fft.fftfreq(N_points, dt)
    f_converted = fgrid * 3.33e-11
    FFT_result = np.fft.fft(measurements)

    output_filename = f"output_{config_index}.dat"
    try:
        with open(output_filename, "w") as out_f:
            out_f.write("# Fourier Transform results\n")
            out_f.write("# Frequency (cm^-1)    FFT_Amplitude\n")
            for i in range(N_points):
                out_f.write(f"{f_converted[i]:.6e} {abs(FFT_result[i]):.6e}\n")
        print(f"Fourier Transform data for configuration {config_index} written to {output_filename}")
    except Exception as e:
        print(f"Error writing {output_filename}:", e)


if __name__ == "__main__":
    start_time = time.time()
    process_simulation()
    end_time = time.time()
    print("Total processing time:", end_time - start_time)
