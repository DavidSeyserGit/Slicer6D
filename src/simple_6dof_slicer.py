#!/usr/bin/env python3
"""
Simple 6DOF Slicer - Easy-to-use command-line tool for 6DOF 3D printing

Usage:
    python simple_6dof_slicer.py model.stl                  # Default orientation (up)
    python simple_6dof_slicer.py model.stl -o output.gcode # Custom output
    python simple_6dof_slicer.py model.stl -d 0 0 1       # Print direction
    python simple_6dof_slicer.py model.stl -d 1 0 0.5     # Angled print
    python simple_6dof_slicer.py --help                   # Full options
"""

import argparse
import sys
import os

try:
    import trimesh
    import numpy as np
    from shapely.geometry import LineString, Polygon, MultiLineString, GeometryCollection
    from shapely.ops import polygonize, unary_union
    from slicing.slicing import (
        generate_gcode_6dof,
        slice_mesh_at_orientation,
        get_bounds_cached
    )
except ImportError as e:
    print(f"Error: Missing dependencies. Run: pip install -r requirements.txt")
    print(f"Details: {e}")
    sys.exit(1)


def simple_slice(input_file, output_file=None, direction=(0, 0, 1), 
                 layer_height=0.2, extrusion=0.05, feedrate=1500, travel=3000):
    """Simple 6DOF slicing - just give it an STL and direction."""
    
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        return False
    
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + "_6dof.gcode"
    
    print(f"Loading: {input_file}")
    mesh = trimesh.load(input_file, force="mesh")
    
    direction = np.array(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        direction = direction / norm
    
    print(f"Slicing at direction: [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]")
    print(f"Layer height: {layer_height}mm")
    
    distances, layer_polygons, rot_matrix = slice_mesh_at_orientation(
        mesh, layer_height, direction.tolist()
    )
    
    if not distances or not layer_polygons:
        print("Error: Slicing produced no layers. Check your model.")
        return False
    
    print(f"Generated {len(distances)} layers")
    
    print("Generating 6DOF G-code...")
    gcode, paths = generate_gcode_6dof(
        distances, layer_polygons, 
        direction.tolist(),
        feedrate=feedrate,
        extrusion_per_mm=extrusion,
        travel_feed=travel,
        infill_spacing=2.0,
        infill_angle=45.0
    )
    
    with open(output_file, "w") as f:
        f.write(gcode)
    
    print(f"Done! G-code saved to: {output_file}")
    print(f"Toolpath segments: {len(paths)}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Simple 6DOF Slicer - Print at any angle, no supports needed!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s model.stl                        # Basic slice
    %(prog)s model.stl -d 0 0 1               # Print straight up (default)
    %(prog)s model.stl -d 1 0 0                # Print sideways
    %(prog)s model.stl -d 0.5 0.5 1            # Angled print
    %(prog)s model.stl -o my_print.gcode      # Custom output name
    %(prog)s model.stl -l 0.1                 # Finer layer height
        """
    )
    
    parser.add_argument("input", help="Input STL file")
    parser.add_argument("-o", "--output", help="Output G-code file (default: input_6dof.gcode)")
    parser.add_argument("-d", "--direction", nargs=3, type=float, default=[0, 0, 1],
                        help="Print direction as X Y Z (default: 0 0 1 = print upward)")
    parser.add_argument("-l", "--layer-height", type=float, default=0.2,
                        help="Layer height in mm (default: 0.2)")
    parser.add_argument("-e", "--extrusion", type=float, default=0.05,
                        help="Extrusion per mm (default: 0.05)")
    parser.add_argument("-f", "--feedrate", type=int, default=1500,
                        help="Print feedrate mm/min (default: 1500)")
    parser.add_argument("-t", "--travel", type=int, default=3000,
                        help="Travel feedrate mm/min (default: 3000)")
    
    args = parser.parse_args()
    
    success = simple_slice(
        args.input,
        args.output,
        args.direction,
        args.layer_height,
        args.extrusion,
        args.feedrate,
        args.travel
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
