#!/usr/bin/env python3
"""
slicer.py

Simple 2D‐perimeter slicer for STL → G‑code (per layer outlines only).

Dependencies:
  pip install trimesh shapely numpy
"""

import sys
import os
import math
import argparse
import trimesh
import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

def slice_mesh(stl_path, layer_height=0.2):
    """
    Slice mesh into horizontal contours.

    Returns:
      z_levels: list of Z heights
      layers:   list of lists of 2D loops (each loop is Nx2 numpy array)
    """
    # 1) load, force a Trimesh if it came back as a Scene
    mesh = trimesh.load(stl_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        # try extracting and concatenating all geometries
        if isinstance(mesh, trimesh.Scene):
            # dump() gives a list of Trimesh
            parts = list(mesh.dump().values()) if hasattr(mesh, 'geometry') \
                    else mesh.dump()
            mesh = trimesh.util.concatenate(parts)
        else:
            raise TypeError(f"Cannot slice object of type {type(mesh)}")

    # 2) get a flat bounds array of length 6
    b = np.array(mesh.bounds).flatten()
    if b.size != 6:
        raise ValueError(f"Unexpected bounds shape: {mesh.bounds}")

    z_min, z_max = float(b[2]), float(b[5])

    # 3) build slicing heights
    z_levels = np.arange(z_min + layer_height/2, z_max, layer_height)
    layers = []
    for z in z_levels:
        section = mesh.section(
            plane_origin=[0,0,z], plane_normal=[0,0,1]
        )
        if section is None or len(section.entities) == 0:
            layers.append([])
            continue
        proj, _ = section.to_planar()
        lines = []
        for ent in proj.entities:
            coords = proj.vertices[ent.points]
            lines.append(LineString(coords))
        merged = unary_union(lines)
        polys = list(polygonize(merged))
        loops = [np.array(poly.exterior.coords) for poly in polys]
        layers.append(loops)

    return z_levels.tolist(), layers

def generate_gcode(z_levels, layers,
                   feedrate=1500,
                   extrusion_per_mm=0.05,
                   travel_feed=3000):
    """
    Turn sliced loops into a simple G‑code string.

    Only prints perimeters, no infill or supports.
    """
    e_pos = 0.0
    lines = []
    # G‑code header
    lines += [
        "; Simple slicer output",
        "G21         ; units = mm",
        "G90         ; absolute coordinates",
        "M82         ; absolute extrusion",
        "G28         ; home",
        "G1 Z5 F3000 ; lift",
    ]

    for z, loops in zip(z_levels, layers):
        if not loops:
            continue
        lines.append(f"; Layer at Z={z:.3f}")
        lines.append(f"G1 Z{z:.3f} F{travel_feed}")
        for loop in loops:
            # move to loop start
            x0, y0 = loop[0]
            lines.append(f"G1 X{x0:.3f} Y{y0:.3f} F{travel_feed}")
            # extrude along loop
            for (x1, y1), (x2, y2) in zip(loop, loop[1:]):
                d = math.hypot(x2-x1, y2-y1)
                e_pos += d * extrusion_per_mm
                lines.append(f"G1 X{x2:.3f} Y{y2:.3f} E{e_pos:.5f} F{feedrate}")
            # close loop
            x1, y1 = loop[-1]
            x2, y2 = loop[0]
            d = math.hypot(x2-x1, y2-y1)
            e_pos += d * extrusion_per_mm
            lines.append(f"G1 X{x2:.3f} Y{y2:.3f} E{e_pos:.5f} F{feedrate}")
    # footer
    lines += [
        "M104 S0     ; turn off extruder",
        "M140 S0     ; turn off bed",
        "G1 X0 Y0    ; park head",
        "M84         ; disable motors",
    ]
    return "\n".join(lines)

def main():
    p = argparse.ArgumentParser(
        description="Slice an STL into simple perimeter G‑code")
    p.add_argument("input", help="Input STL file")
    p.add_argument("output", help="Output G‑code file")
    p.add_argument("--layer-height", "-l", type=float, default=0.2)
    p.add_argument("--extrusion", "-e", type=float, default=0.05,
                   help="E per mm")
    p.add_argument("--feed", "-f", type=int, default=1500,
                   help="Printing feedrate (F)")
    p.add_argument("--travel-feed", type=int, default=3000,
                   help="Travel moves feedrate")
    args = p.parse_args()

    z_levels, layers = slice_mesh(args.input, args.layer_height)
    gcode = generate_gcode(z_levels, layers,
                           feedrate=args.feed,
                           extrusion_per_mm=args.extrusion,
                           travel_feed=args.travel_feed)
    with open(args.output, "w") as fp:
        fp.write(gcode)
    print(f"Wrote {len(z_levels)} layers → {args.output}")

if __name__ == "__main__":
    main()
