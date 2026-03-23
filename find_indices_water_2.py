#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified find_indices.py script for functionalized SiO2 surfaces.
It uses H atoms (from the XDATCAR header) as starting points.
Based on the command-line argument, it outputs:
  1) Mode 1 (only OH): Bonds: H–O and O–Si, Angle: H–O–Si.
  2) Mode 2 (only CH3): Bonds: C–H (for each H in CH3) and C–Si;
       Angles: all H–C–H combinations and H–C–Si angles.
  3) Mode 3 (only surface angles): Angles at the Si center between any two functional groups.
  4) Mode 4 (all): Both groups’ bonds/angles (OH & CH3) plus the surface angles.
  5) Mode 5 (only water): Bonds: H–O (only once per bond) and angle: H–O–H.
All indices are output in 1-indexed format.

Changes implemented (preserving the same I/O and behavior except where noted):
  - Water detection now requires an O–H cutoff (default 1.2 Å) for the second H near O.
    This affects ONLY water vs OH classification (no other bond/angle inference uses cutoffs).
  - Performance improvements:
      * Candidate lists: neighbor searches are restricted to relevant element indices.
      * Metric tensor + squared distances: avoids repeated Cartesian conversions and sqrt in tight loops.
"""

import numpy as np
import sys


WATER_OH_CUTOFF_A = 1.2  # Å (used ONLY to decide whether an O has a second covalently bonded H)


def get_element_list(xdatcar_lines):
    """
    Constructs a list of element symbols for each atom based on the header.
    Assumes line 6 holds element symbols (e.g., "Si O C H") and line 7 holds counts.
    """
    elements = xdatcar_lines[5].split()
    counts = list(map(int, xdatcar_lines[6].split()))
    elem_list = []
    for e, count in zip(elements, counts):
        elem_list.extend([e] * count)
    return elem_list


def get_H_indices(xdatcar_lines, element_list):
    """Returns the indices (0-indexed) of all atoms that are hydrogen."""
    return [i for i, e in enumerate(element_list) if e.upper() == 'H']


def read_xdatcar(filename="XDATCAR"):
    """
    Reads the first configuration from XDATCAR.
    Assumes:
      - Line 1: comment
      - Line 2: scaling factor
      - Lines 3-5: lattice vectors
      - Line 6: element symbols
      - Line 7: atom counts
      - Line 8: coordinate type
    Then a configuration block starts (identified by a line containing "configuration").
    Returns a list of fractional coordinate numpy arrays (one per atom) and the lattice (3x3 array).
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    scaling_factor = float(lines[1].strip())
    a_vector = np.array(list(map(float, lines[2].split()))) * scaling_factor
    b_vector = np.array(list(map(float, lines[3].split()))) * scaling_factor
    c_vector = np.array(list(map(float, lines[4].split()))) * scaling_factor
    lattice = np.array([a_vector, b_vector, c_vector])

    counts = list(map(int, lines[6].split()))
    total_atoms = sum(counts)

    start = None
    for i, line in enumerate(lines):
        if "configuration" in line.lower():
            start = i + 1
            break
    if start is None:
        raise ValueError("Could not find configuration block in XDATCAR.")

    positions = []
    for i in range(total_atoms):
        pos_line = lines[start + i].strip().split()
        if pos_line:
            pos = [float(x) for x in pos_line[:3]]
            positions.append(np.array(pos, dtype=float))

    return positions, lattice


def _wrap_frac_diff(diff_frac):
    """Minimum-image wrap in fractional space."""
    return diff_frac - np.round(diff_frac)


def _metric_tensor(lattice):
    """
    Metric tensor G for squared distances in fractional coordinates:
        r_cart = f @ lattice  (with lattice rows as basis vectors)
        |r_cart|^2 = f @ (lattice @ lattice.T) @ f^T
    """
    return lattice @ lattice.T


def distance2(i, j, positions, G):
    """Returns squared Cartesian distance between atoms i and j using metric tensor."""
    diff = positions[j] - positions[i]      # fractional difference
    diff = _wrap_frac_diff(diff)
    # d^2 = diff^T * G * diff
    return float(diff @ G @ diff)


def distance(i, j, positions, lattice, G=None):
    """
    Returns the Cartesian distance between atoms i and j.
    Kept for compatibility with existing call sites; uses metric tensor if provided.
    """
    if G is None:
        # Fallback to original, slower path if needed
        diff = positions[j] - positions[i]
        diff = _wrap_frac_diff(diff)
        cart_diff = np.dot(diff, lattice)
        return float(np.linalg.norm(cart_diff))
    return float(np.sqrt(distance2(i, j, positions, G)))


def find_closest_neighbor(target_index, positions, G, exclude=None, candidates=None):
    """
    Finds the closest neighbor of atom at target_index among positions (or among 'candidates'),
    excluding any indices in the optional iterable 'exclude'.
    Returns (closest_index, distance).
    """
    exclude_set = set(exclude) if exclude is not None else None
    search_indices = candidates if candidates is not None else range(len(positions))

    min_d2 = float('inf')
    closest_index = None

    for i in search_indices:
        if exclude_set is not None and i in exclude_set:
            continue
        if i == target_index:
            continue
        d2 = distance2(target_index, i, positions, G)
        if d2 < 1e-16:  # extremely close / self
            continue
        if d2 < min_d2:
            min_d2 = d2
            closest_index = i

    if closest_index is None:
        return None, None
    return closest_index, float(np.sqrt(min_d2))


def find_closest_neighbor_of_type(target_index, positions, G, element_list, target_element, exclude=None, element_indices=None):
    """
    Finds the closest neighbor of atom at target_index that has element matching target_element.
    Returns (closest_index, distance) or (None, None) if not found.

    If element_indices is provided, uses it to restrict search to just those indices.
    """
    exclude_set = set(exclude) if exclude is not None else None
    target_el = target_element.upper()

    if element_indices is not None:
        search_indices = element_indices.get(target_el, [])
    else:
        search_indices = range(len(positions))

    min_d2 = float('inf')
    closest_index = None

    for i in search_indices:
        if exclude_set is not None and i in exclude_set:
            continue
        if i == target_index:
            continue
        # If we weren't given indices, we must filter by element here
        if element_indices is None and element_list[i].upper() != target_el:
            continue
        d2 = distance2(target_index, i, positions, G)
        if d2 < 1e-16:
            continue
        if d2 < min_d2:
            min_d2 = d2
            closest_index = i

    if closest_index is None:
        return None, None
    return closest_index, float(np.sqrt(min_d2))


def main():
    # Get group selection from command-line argument.
    if len(sys.argv) < 2:
        print("Usage: python find_indices.py <group_mode>")
        print("   group_mode: 1=OH only, 2=CH3 only, 3=Surface angles only, 4=All (excluding water), 5=Water only")
        sys.exit(1)

    try:
        group_mode = int(sys.argv[1])
    except ValueError:
        print("Invalid group_mode. Must be an integer (1,2,3,4,5).")
        sys.exit(1)

    if group_mode not in (1, 2, 3, 4, 5):
        print("group_mode must be 1, 2, 3, 4, or 5.")
        sys.exit(1)

    positions, lattice = read_xdatcar("XDATCAR")
    G = _metric_tensor(lattice)

    with open("XDATCAR", "r") as f:
        lines = f.readlines()

    element_list = get_element_list(lines)
    h_indices = get_H_indices(lines, element_list)

    # Precompute element index lists for performance (candidate lists)
    element_indices = {}
    for idx, el in enumerate(element_list):
        element_indices.setdefault(el.upper(), []).append(idx)

    O_indices = element_indices.get('O', [])
    C_indices = element_indices.get('C', [])
    Si_indices = element_indices.get('SI', [])
    # For H we already have h_indices, but also keep in dict
    element_indices['H'] = h_indices

    # For H's first neighbor, we only care about O or C (since other types are ignored anyway)
    H_first_candidates = O_indices + C_indices

    output_lines = []
    si_attachments = {}  # key: Si index, value: list of (group_type, group_atom)
    ch3_groups = {}      # key: C index, value: list of H indices (for CH3)
    oh_groups = []       # list of tuples (H, O, Si) for OH groups
    water_groups = []    # list of tuples (H1, O, H2) for water molecules

    cutoff2 = WATER_OH_CUTOFF_A * WATER_OH_CUTOFF_A

    # Process each H atom as starting point.
    for H in h_indices:
        neighbor, _d = find_closest_neighbor(
            H, positions, G,
            exclude=[H],
            candidates=H_first_candidates
        )
        if neighbor is None:
            continue

        neighbor_el = element_list[neighbor].upper()

        if neighbor_el == 'O':
            # Check if this is a water molecule:
            # Look for a second H (H2) that is the closest H to the O (excluding the current H and the O itself).
            H2, _d_H2 = find_closest_neighbor_of_type(
                neighbor, positions, G, element_list, 'H',
                exclude=[H, neighbor],
                element_indices=element_indices
            )

            # NEW: Apply cutoff ONLY for water detection.
            # If H2 exists but O–H2 is not within cutoff, treat as OH (fall through).
            if H2 is not None:
                d2_OH2 = distance2(neighbor, H2, positions, G)
                if d2_OH2 <= cutoff2:
                    water_tuple = (min(H, H2), neighbor, max(H, H2))
                    if water_tuple not in water_groups:
                        water_groups.append(water_tuple)
                    continue  # Skip processing as an OH group.

            # Otherwise, treat it as an OH group.
            Si, _d2 = find_closest_neighbor_of_type(
                neighbor, positions, G, element_list, 'Si',
                exclude=[H, neighbor],
                element_indices=element_indices
            )
            if Si is None:
                continue
            oh_groups.append((H, neighbor, Si))
            si_attachments.setdefault(Si, []).append(('O', neighbor))

        elif neighbor_el == 'C':
            # Assume CH3 group.
            C = neighbor
            ch3_groups.setdefault(C, []).append(H)
        else:
            continue

    # Process CH3 groups: only keep groups with at least 3 H atoms.
    ch3_groups_processed = []
    for C, H_list in ch3_groups.items():
        if len(H_list) < 3:
            print(f"Warning: CH3 group with C index {C+1} has only {len(H_list)} H atoms. Skipping.")
            continue

        if len(H_list) > 3:
            # Keep the 3 closest H to C (use squared distances for speed)
            H_list = sorted(H_list, key=lambda H: distance2(C, H, positions, G))[:3]

        exclude_list = H_list + [C]
        Si, _d3 = find_closest_neighbor_of_type(
            C, positions, G, element_list, 'Si',
            exclude=exclude_list,
            element_indices=element_indices
        )
        if Si is None:
            continue

        ch3_groups_processed.append((C, H_list, Si))
        si_attachments.setdefault(Si, []).append(('C', C))

    # Depending on group_mode, output different sets of bonds/angles.
    # Mode 1: Only OH groups.
    if group_mode in (1, 4):
        for (H, O, Si) in oh_groups:
            output_lines.append(f"2 {H+1} {O+1}")          # Bond: H-O
            output_lines.append(f"2 {O+1} {Si+1}")         # Bond: O-Si
            output_lines.append(f"3 {H+1} {O+1} {Si+1}")   # Angle: H-O-Si

    # Mode 2: Only CH3 groups.
    if group_mode in (2, 4):
        for (C, H_list, Si) in ch3_groups_processed:
            for H in H_list:
                output_lines.append(f"2 {C+1} {H+1}")      # Bond: C-H
            output_lines.append(f"2 {C+1} {Si+1}")         # Bond: C-Si

            # H-C-H angles (all combinations)
            if len(H_list) == 3:
                output_lines.append(f"3 {H_list[0]+1} {C+1} {H_list[1]+1}")
                output_lines.append(f"3 {H_list[0]+1} {C+1} {H_list[2]+1}")
                output_lines.append(f"3 {H_list[1]+1} {C+1} {H_list[2]+1}")

            # H-C-Si angles for each H.
            for H in H_list:
                output_lines.append(f"3 {H+1} {C+1} {Si+1}")

    # Mode 3: Only surface angles (Si at the center).
    if group_mode in (3, 4):
        for Si, attachments in si_attachments.items():
            if len(attachments) < 2:
                continue
            n = len(attachments)
            for i in range(n):
                for j in range(i + 1, n):
                    output_lines.append(f"3 {attachments[i][1]+1} {Si+1} {attachments[j][1]+1}")

    # Mode 5: Only water molecules.
    if group_mode == 5:
        for (H1, O, H2) in water_groups:
            output_lines.append(f"2 {H1+1} {O+1}")         # Bond: H1-O
            output_lines.append(f"2 {O+1} {H2+1}")         # Bond: O-H2
            output_lines.append(f"3 {H1+1} {O+1} {H2+1}")  # Angle: H1-O-H2

    with open("config-out.dat", "w") as f:
        for line in output_lines:
            f.write(line + "\n")

    print("Processing complete. Output written to config-out.dat.")


if __name__ == "__main__":
    main()
