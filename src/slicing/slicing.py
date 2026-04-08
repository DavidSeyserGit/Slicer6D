#!/usr/bin/env python3
"""
slicing.py

Slicer for STL -> G-code with perimeters and basic line infill.

Dependencies:
  pip install trimesh shapely numpy
"""

import sys
import math
import trimesh
import numpy as np
from shapely.geometry import (LineString, Polygon, MultiPolygon, MultiLineString,
                              GeometryCollection)
from shapely.ops import polygonize, unary_union
from shapely.affinity import rotate, translate

# --- Constants ---
# Small buffer distance to help with floating point issues in Shapely
BUFFER_EPS = 1e-6
# Small distance to offset infill from perimeter slightly (optional)
INFILL_CLIP_OFFSET = -0.1 # Negative for inset

# --- Helper Functions ---
def get_bounds(geometries):
    """Calculate the total bounds of a list/collection of Shapely geometries."""
    if not geometries:
        return None
    # Ensure we handle single geometries too
    if not isinstance(geometries, (list, tuple, GeometryCollection)):
        geometries = [geometries]

    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    has_bounds = False
    for geom in geometries:
        if geom and not geom.is_empty:
            b = geom.bounds
            if len(b) == 4:
                min_x = min(min_x, b[0])
                min_y = min(min_y, b[1])
                max_x = max(max_x, b[2])
                max_y = max(max_y, b[3])
                has_bounds = True
    return (min_x, min_y, max_x, max_y) if has_bounds else None

# --- Core Slicing Logic ---

def slice_mesh_to_polygons(stl_path, layer_height=0.2):
    """
    Slice mesh into horizontal contours represented as Shapely Polygons.

    Returns:
      z_levels: list of Z heights
      layer_polygons: list of lists of shapely.geometry.Polygon objects
                      (Each inner list represents one layer, potentially with multiple islands)
    """
    print(f"Loading mesh: {stl_path}...")
    mesh = trimesh.load(stl_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        if isinstance(mesh, trimesh.Scene):
            print("Scene detected, attempting to concatenate geometry...")
            parts = list(mesh.dump().values()) if hasattr(mesh, 'geometry') else mesh.dump()
            if not parts: raise TypeError("Scene contains no mesh geometry.")
            mesh = trimesh.util.concatenate(parts)
            if not isinstance(mesh, trimesh.Trimesh): raise TypeError(f"Concatenated object not Trimesh: {type(mesh)}")
        else:
            raise TypeError(f"Cannot slice object of type {type(mesh)}")
    print(f"Mesh loaded: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")

    b = np.array(mesh.bounds).flatten()
    if b.size != 6: raise ValueError(f"Unexpected bounds shape: {mesh.bounds}")
    z_min, z_max = float(b[2]), float(b[5])
    print(f"Z bounds: min={z_min:.3f}, max={z_max:.3f}")

    start_z = z_min + layer_height / 2.0
    num_layers = math.ceil((z_max - start_z) / layer_height)
    if num_layers <= 0:
        print("Warning: Mesh height too small for layer height.")
        return [], []
    z_levels = np.arange(start_z, z_max, layer_height)

    print(f"Generating {len(z_levels)} layers...")
    all_layer_polygons = []
    for i, z in enumerate(z_levels):
        layer_loops = []
        try:
            section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
            if section is None or len(section.entities) == 0:
                all_layer_polygons.append([])
                continue

            # Project to 2D - handle potential errors
            try:
                proj, _ = section.to_planar()
            except Exception as e:
                 print(f"  Layer {i+1} at Z={z:.3f}: Error during projection: {e}", file=sys.stderr)
                 all_layer_polygons.append([])
                 continue

            # Convert trimesh paths to shapely lines
            lines = []
            for ent in proj.entities:
                if hasattr(ent, 'points') and len(ent.points) >= 2:
                     coords = proj.vertices[ent.points]
                     if len(coords) >= 2:
                         lines.append(LineString(coords))

            if not lines:
                all_layer_polygons.append([])
                continue

            # Polygonize - Use buffer to help close small gaps if needed
            # merged = unary_union([line.buffer(BUFFER_EPS) for line in lines]) # Option 1: Buffer lines
            # merged_boundary = merged.boundary # Option 2: Use boundary of buffered lines
            merged = unary_union(lines) # Option 3: Direct union (often works)

            # Polygonize the merged lines
            polygons_on_layer = list(polygonize(merged))

            # Validate and clean polygons (remove tiny artifacts, ensure validity)
            valid_polygons = []
            for poly in polygons_on_layer:
                 if isinstance(poly, Polygon):
                     # Optional: buffer(0) is a trick to fix some invalid geometries
                     cleaned_poly = poly.buffer(0)
                     if cleaned_poly.is_valid and cleaned_poly.area > 1e-6: # Check area threshold
                         valid_polygons.append(cleaned_poly)

            all_layer_polygons.append(valid_polygons)

        except Exception as e:
            print(f"  Layer {i+1} at Z={z:.3f}: Unexpected error during slicing: {e}", file=sys.stderr)
            # import traceback # Uncomment for debugging
            # traceback.print_exc() # Uncomment for debugging
            all_layer_polygons.append([]) # Add empty list for this layer on error

    print("Slicing complete.")
    return z_levels.tolist(), all_layer_polygons


def generate_infill(polygons, spacing, angle_degrees, bounds):
    """
    Generates clipped line infill for a list of polygons on a single layer.

    Args:
      polygons: List of shapely.geometry.Polygon for the layer.
      spacing: Distance between infill lines.
      angle_degrees: Rotation angle for the infill pattern.
      bounds: Tuple (min_x, min_y, max_x, max_y) for the layer geometry.

    Returns:
      List of shapely.geometry.LineString representing the infill paths.
    """
    if not polygons or spacing <= 0 or bounds is None:
        return []

    min_x, min_y, max_x, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        return []

    # Combine polygons for efficient clipping (optional, can clip per polygon too)
    layer_area = unary_union(polygons)
    if layer_area.is_empty:
        return []

    # Inset the area slightly for clipping infill (prevents touching perimeter)
    # Adjust INFILL_CLIP_OFFSET as needed, 0 for no offset
    try:
        clip_area = layer_area.buffer(INFILL_CLIP_OFFSET)
        if clip_area.is_empty: return [] # Return if buffer eliminates area
    except Exception: # Buffer might fail on complex shapes
        clip_area = layer_area # Fallback to original area

    # Calculate diagonal length to ensure lines cover area after rotation
    diagonal = math.sqrt(width**2 + height**2)
    center_x = min_x + width / 2
    center_y = min_y + height / 2

    # Generate horizontal lines covering the extended bounds
    num_lines = int(diagonal / spacing) + 2 # Add buffer lines
    lines = []
    start_y = center_y - (num_lines / 2.0) * spacing
    for i in range(num_lines):
        y = start_y + i * spacing
        # Extend lines beyond bounds to ensure coverage after rotation
        line = LineString([(center_x - diagonal, y), (center_x + diagonal, y)])
        lines.append(line)

    # Rotate lines if needed
    if not math.isclose(angle_degrees % 180, 0): # Avoid rotation for 0/180 deg
        rotated_lines = [rotate(line, angle_degrees, origin=(center_x, center_y)) for line in lines]
    else:
        rotated_lines = lines

    # Clip lines against the (potentially inset) layer area
    infill_paths = []
    for line in rotated_lines:
        try:
            intersection = clip_area.intersection(line)
            # Handle different intersection results
            if intersection.is_empty:
                continue
            elif isinstance(intersection, LineString):
                if intersection.length > 1e-6: # Check length threshold
                    infill_paths.append(intersection)
            elif isinstance(intersection, MultiLineString):
                for part in intersection.geoms: # Use .geoms for MultiLineString
                     if part.length > 1e-6:
                         infill_paths.append(part)
            # Ignore Points or other geometry types
        except Exception as e:
             # This can happen with complex intersections
             # print(f"    Warning: Shapely intersection error during infill: {e}")
             pass # Skip this line on error

    return infill_paths



def normalize_vector(v):
    """Normalize a 3D vector to unit length."""
    v = np.array(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        raise ValueError("Vector magnitude too small")
    return v / norm


def rotation_matrix_between_vectors(v_from, v_to):
    """Create a rotation matrix that rotates v_from to align with v_to."""
    v_from = normalize_vector(v_from)
    v_to = normalize_vector(v_to)
    
    v = np.cross(v_from, v_to)
    c = np.dot(v_from, v_to)
    
    if c > 0.9999:
        return np.eye(3)
    if c < -0.9999:
        orth = np.array([1, 0, 0])
        if abs(v_from[0]) > 0.9:
            orth = np.array([0, 1, 0])
        orth = normalize_vector(np.cross(v_from, orth))
        return -np.eye(3) + 2 * np.outer(orth, orth)
    
    s = np.linalg.norm(v)
    skew = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    return np.eye(3) + skew + np.dot(skew, skew) * (1 - c) / (s * s)


def eulerAnglesFromRotationMatrix(R):
    """Extract Euler angles (A, B, C) in degrees from rotation matrix.
    Returns angles for rotation around X (A), Y (B), Z (C) axes.
    Applies rotation in order Z-Y-X (yaw, pitch, roll).
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    
    return math.degrees(x), math.degrees(y), math.degrees(z)


def slice_mesh_at_orientation(mesh, layer_height, print_direction):
    """
    Slice a mesh at an arbitrary orientation.

    Args:
        mesh: trimesh.Trimesh object
        layer_height: float, layer thickness
        print_direction: 3D vector (list or array) specifying the direction 
                        the printer will print (becomes new Z-axis)

    Returns:
        Tuple (distances, layer_polygons) where:
        - distances: list of distances along print direction
        - layer_polygons: list of lists of shapely Polygon objects for each layer
    """
    print_direction = normalize_vector(print_direction)
    z_axis = np.array([0, 0, 1])
    
    rot_matrix = rotation_matrix_between_vectors(print_direction, z_axis)
    
    original_vertices = mesh.vertices.copy()
    original_faces = mesh.faces.copy()
    
    rotated_vertices = np.dot(original_vertices, rot_matrix.T)
    
    transformed_mesh = trimesh.Trimesh(vertices=rotated_vertices, faces=original_faces)
    
    bounds = np.array(transformed_mesh.bounds).flatten()
    z_min, z_max = float(bounds[2]), float(bounds[5])
    
    start_z = z_min + layer_height / 2.0
    num_layers = math.ceil((z_max - start_z) / layer_height)
    if num_layers <= 0:
        return [], []
    
    distances = np.arange(start_z, z_max, layer_height)
    
    all_layer_polygons = []
    for i, z in enumerate(distances):
        try:
            section = transformed_mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
            if section is None or len(section.entities) == 0:
                all_layer_polygons.append([])
                continue
            
            try:
                proj, _ = section.to_planar()
            except Exception as e:
                print(f"  Layer {i+1} at dist={z:.3f}: Projection error: {e}", file=sys.stderr)
                all_layer_polygons.append([])
                continue
            
            lines = []
            for ent in proj.entities:
                if hasattr(ent, 'points') and len(ent.points) >= 2:
                    coords = proj.vertices[ent.points]
                    if len(coords) >= 2:
                        lines.append(LineString(coords))
            
            if not lines:
                all_layer_polygons.append([])
                continue
            
            merged = unary_union(lines)
            polygons_on_layer = list(polygonize(merged))
            
            valid_polygons = []
            for poly in polygons_on_layer:
                if isinstance(poly, Polygon):
                    cleaned_poly = poly.buffer(0)
                    if cleaned_poly.is_valid and cleaned_poly.area > 1e-6:
                        valid_polygons.append(cleaned_poly)
            
            all_layer_polygons.append(valid_polygons)
            
        except Exception as e:
            print(f"  Layer {i+1} at dist={z:.3f}: Slicing error: {e}", file=sys.stderr)
            all_layer_polygons.append([])
    
    return distances.tolist(), all_layer_polygons, rot_matrix


def generate_gcode_6dof(distances, layer_polygons, rot_matrix,
                       feedrate=1500,
                       extrusion_per_mm=0.05,
                       travel_feed=3000,
                       infill_spacing=2.0,
                       infill_angle=45.0):
    """
    Generate 6DOF G-code with ABC orientation commands.

    Args:
        distances: list of distances along print direction (slicing positions)
        layer_polygons: list of lists of shapely Polygons for each layer
        rot_matrix: 3x3 rotation matrix used for slicing (transforms print direction to Z)
        feedrate: printing feedrate in mm/min
        extrusion_per_mm: extrusion amount per mm of XY travel
        travel_feed: travel moves feedrate in mm/min
        infill_spacing: spacing between infill lines (0 to disable)
        infill_angle: angle of infill lines in degrees

    Returns:
        String containing G-code with ABC orientation commands
    """
    e_pos = 0.0
    gcode_lines = []
    
    a, b, c = eulerAnglesFromRotationMatrix(rot_matrix)
    
    gcode_lines += [
        "; 6DOF Slicer G-code",
        "; Print direction transformed to Z-axis",
        f"; Orientation: A={a:.2f} B={b:.2f} C={c:.2f}",
        "G90 ; Absolute positioning",
        "G21 ; Millimeters",
        f"G1 F{travel_feed} ; Set travel feed",
        f"G1 A{a:.2f} B{b:.2f} C{c:.2f} ; Set orientation",
    ]
    
    for layer_idx, (dist, polygons) in enumerate(zip(distances, layer_polygons)):
        if not polygons:
            continue
        
        gcode_lines.append(f"; Layer {layer_idx + 1} at distance {dist:.3f}")
        gcode_lines.append(f"G1 Z{dist:.3f} F{travel_feed}")
        
        gcode_lines.append("; Perimeters")
        for poly in polygons:
            exterior_coords_2d = np.array(poly.exterior.coords)
            if exterior_coords_2d.shape[0] >= 2:
                x0, y0 = exterior_coords_2d[0]
                gcode_lines.append(f"G1 X{x0:.3f} Y{y0:.3f} F{travel_feed} A{a:.2f} B{b:.2f} C{c:.2f}")
                current_x, current_y = x0, y0
                for i in range(1, exterior_coords_2d.shape[0]):
                    x2, y2 = exterior_coords_2d[i]
                    if not (math.isclose(current_x, x2) and math.isclose(current_y, y2)):
                        d = math.hypot(x2 - current_x, y2 - current_y)
                        if d > 1e-6:
                            e_pos += d * extrusion_per_mm
                            gcode_lines.append(f"G1 X{x2:.3f} Y{y2:.3f} Z{dist:.3f} A{a:.2f} B{b:.2f} C{c:.2f} E{e_pos:.5f} F{feedrate}")
                        current_x, current_y = x2, y2
            
            for interior in poly.interiors:
                interior_coords_2d = np.array(interior.coords)
                if interior_coords_2d.shape[0] >= 2:
                    x0, y0 = interior_coords_2d[0]
                    gcode_lines.append(f"G1 X{x0:.3f} Y{y0:.3f} F{travel_feed} A{a:.2f} B{b:.2f} C{c:.2f}")
                    current_x, current_y = x0, y0
                    for i in range(1, interior_coords_2d.shape[0]):
                        x2, y2 = interior_coords_2d[i]
                        if not (math.isclose(current_x, x2) and math.isclose(current_y, y2)):
                            d = math.hypot(x2 - current_x, y2 - current_y)
                            if d > 1e-6:
                                e_pos += d * extrusion_per_mm
                                gcode_lines.append(f"G1 X{x2:.3f} Y{y2:.3f} Z{dist:.3f} A{a:.2f} B{b:.2f} C{c:.2f} E{e_pos:.5f} F{feedrate}")
                            current_x, current_y = x2, y2
        
        if infill_spacing > 0:
            gcode_lines.append("; Infill")
            layer_bounds = get_bounds(polygons)
            infill_lines = generate_infill(polygons, infill_spacing, infill_angle, layer_bounds)
            
            for line_seg in infill_lines:
                infill_coords_2d = np.array(line_seg.coords)
                if infill_coords_2d.shape[0] >= 2:
                    x0, y0 = infill_coords_2d[0]
                    gcode_lines.append(f"G1 X{x0:.3f} Y{y0:.3f} F{travel_feed} A{a:.2f} B{b:.2f} C{c:.2f}")
                    current_x, current_y = x0, y0
                    for i in range(1, infill_coords_2d.shape[0]):
                        x2, y2 = infill_coords_2d[i]
                        if not (math.isclose(current_x, x2) and math.isclose(current_y, y2)):
                            d = math.hypot(x2 - current_x, y2 - current_y)
                            if d > 1e-6:
                                e_pos += d * extrusion_per_mm
                                gcode_lines.append(f"G1 X{x2:.3f} Y{y2:.3f} Z{dist:.3f} A{a:.2f} B{b:.2f} C{c:.2f} E{e_pos:.5f} F{feedrate}")
                            current_x, current_y = x2, y2
    
    gcode_lines += [
        "G1 Z5 F5000 ; Lift",
        "M84 ; Disable steppers",
        "; End of 6DOF G-code",
    ]
    
    return "\n".join(gcode_lines)

def generate_gcode_with_infill(z_levels, layer_polygons,
                               feedrate=1500,
                               extrusion_per_mm=0.05,
                               travel_feed=3000,
                               infill_spacing=2.0,
                               infill_angle=45.0):
    """
    Turn sliced polygons into G-code including perimeters and line infill.
    Also returns the list of 3D paths generated.
    """
    e_pos = 0.0
    gcode_lines = [] # Renamed from 'lines' to avoid confusion
    # --- NEW: List to store actual paths for visualization ---
    visualization_paths = []
    # --- End New ---

    # G-code header (add to gcode_lines)
    gcode_lines += [
        "; Slicer output with Perimeters and Line Infill",
        # ... (rest of header) ...
        "G1 Z5 F5000 ; lift Z 5mm",
    ]

    # --- G-code Generation Loop ---
    for z, polygons in zip(z_levels, layer_polygons):
        if not polygons:
            continue
        gcode_lines.append(f"; Layer at Z={z:.3f}")
        gcode_lines.append(f"G1 Z{z:.3f} F{travel_feed}") # Move to layer height

        # --- 1. Print Perimeters (Exterior and Interior Holes) ---
        gcode_lines.append("; Perimeters")
        for poly in polygons:
            # Exterior perimeter
            exterior_coords_2d = np.array(poly.exterior.coords)
            if exterior_coords_2d.shape[0] >= 2:
                # --- Store path for visualization ---
                z_col = np.full((exterior_coords_2d.shape[0], 1), z)
                visualization_paths.append(np.hstack((exterior_coords_2d, z_col)))
                # --- End Store path ---
                x0, y0 = exterior_coords_2d[0]
                gcode_lines.append(f"G1 X{x0:.3f} Y{y0:.3f} F{travel_feed}") # Travel to start
                current_x, current_y = x0, y0
                for i in range(1, exterior_coords_2d.shape[0]):
                    x2, y2 = exterior_coords_2d[i]
                    if not (math.isclose(current_x, x2) and math.isclose(current_y, y2)):
                        d = math.hypot(x2 - current_x, y2 - current_y)
                        if d > 1e-6:
                            e_pos += d * extrusion_per_mm
                            gcode_lines.append(f"G1 X{x2:.3f} Y{y2:.3f} E{e_pos:.5f} F{feedrate}")
                        current_x, current_y = x2, y2

            # Interior holes (if any)
            for interior in poly.interiors:
                interior_coords_2d = np.array(interior.coords)
                if interior_coords_2d.shape[0] >= 2:
                    # --- Store path for visualization ---
                    z_col = np.full((interior_coords_2d.shape[0], 1), z)
                    visualization_paths.append(np.hstack((interior_coords_2d, z_col)))
                    # --- End Store path ---
                    x0, y0 = interior_coords_2d[0]
                    gcode_lines.append(f"G1 X{x0:.3f} Y{y0:.3f} F{travel_feed}") # Travel to hole start
                    current_x, current_y = x0, y0
                    for i in range(1, interior_coords_2d.shape[0]):
                        x2, y2 = interior_coords_2d[i]
                        if not (math.isclose(current_x, x2) and math.isclose(current_y, y2)):
                            d = math.hypot(x2 - current_x, y2 - current_y)
                            if d > 1e-6:
                                e_pos += d * extrusion_per_mm
                                gcode_lines.append(f"G1 X{x2:.3f} Y{y2:.3f} E{e_pos:.5f} F{feedrate}")
                            current_x, current_y = x2, y2

        # --- 2. Generate and Print Infill ---
        if infill_spacing > 0:
            gcode_lines.append("; Infill")
            layer_bounds = get_bounds(polygons)
            infill_lines = generate_infill(polygons, infill_spacing, infill_angle, layer_bounds)

            for line_seg in infill_lines: # line_seg is a Shapely LineString
                infill_coords_2d = np.array(line_seg.coords)
                if infill_coords_2d.shape[0] >= 2:
                    # --- Store path for visualization ---
                    z_col = np.full((infill_coords_2d.shape[0], 1), z)
                    visualization_paths.append(np.hstack((infill_coords_2d, z_col)))
                    # --- End Store path ---
                    x0, y0 = infill_coords_2d[0]
                    gcode_lines.append(f"G1 X{x0:.3f} Y{y0:.3f} F{travel_feed}") # Travel to infill line start
                    current_x, current_y = x0, y0
                    # Extrude along the segments of the infill line
                    for i in range(1, infill_coords_2d.shape[0]):
                         x2, y2 = infill_coords_2d[i]
                         if not (math.isclose(current_x, x2) and math.isclose(current_y, y2)):
                             d = math.hypot(x2 - current_x, y2 - current_y)
                             if d > 1e-6:
                                 e_pos += d * extrusion_per_mm
                                 gcode_lines.append(f"G1 X{x2:.3f} Y{y2:.3f} E{e_pos:.5f} F{feedrate}")
                             current_x, current_y = x2, y2

    # Footer (add to gcode_lines)
    gcode_lines += [
        "G1 Z{:.3f} F5000 ; Raise Z at the end".format(z_levels[-1] + 5 if z_levels else 5),
        # ... (rest of footer) ...
        "; End of G-code",
    ]
    # --- Return both G-code and the collected pahs ---
    return "\n".join(gcode_lines), visualization_paths
    # --- End Return ---


def generate_gcode_6dof(z_levels, layer_polygons, direction,
                        feedrate=1500,
                        extrusion_per_mm=0.05,
                        travel_feed=3000):
    """
    Generate G-code with ABC orientation commands for 6DOF printing.
    
    Args:
      z_levels: List of Z heights for each layer.
      layer_polygons: List of polygon lists for each layer.
      direction: List [x, y, z] representing the print direction normal vector.
      feedrate: Print feedrate in mm/min.
      extrusion_per_mm: Extrusion amount per mm of travel.
      travel_feed: Travel feedrate in mm/min.
      
    Returns:
      gcode_string: G-code with ABC orientation commands.
      visualization_paths: List of numpy arrays representing toolpaths.
    """
    from math import acos, asin, atan2
    
    dir_vec = np.array(direction, dtype=float)
    norm = np.linalg.norm(dir_vec)
    if norm < 1e-6:
        dir_vec = np.array([0, 0, 1])
    else:
        dir_vec = dir_vec / norm
    
    pitch = asin(-dir_vec[1]) if abs(dir_vec[2]) < 0.999 else 0.0
    roll = acos(dir_vec[2]) if abs(dir_vec[2]) < 0.999 else 0.0
    yaw = atan2(dir_vec[0], dir_vec[2]) if abs(dir_vec[2]) < 0.999 else 0.0
    
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)
    
    e_pos = 0.0
    gcode_lines = []
    visualization_paths = []
    
    gcode_lines += [
        "; 6DOF G-code with orientation",
        f"; Print direction: {direction[0]:.4f}, {direction[1]:.4f}, {direction[2]:.4f}",
        "G90 ; Absolute positioning",
        "M82 ; Absolute extrusion",
        f"G1 F{travel_feed} ; Set travel feed",
    ]
    
    for z, polygons in zip(z_levels, layer_polygons):
        if not polygons:
            continue
        gcode_lines.append(f"; Layer at Z={z:.3f}")
        gcode_lines.append(f"G1 Z{z:.3f} F{travel_feed}")
        gcode_lines.append(f"G1 A{pitch_deg:.2f} B{roll_deg:.2f} C{yaw_deg:.2f} F{travel_feed}")
        
        gcode_lines.append("; Perimeters")
        for poly in polygons:
            exterior_coords_2d = np.array(poly.exterior.coords)
            if exterior_coords_2d.shape[0] >= 2:
                z_col = np.full((exterior_coords_2d.shape[0], 1), z)
                visualization_paths.append(np.hstack((exterior_coords_2d, z_col)))
                
                x0, y0 = exterior_coords_2d[0]
                gcode_lines.append(f"G1 X{x0:.3f} Y{y0:.3f} F{travel_feed}")
                current_x, current_y = x0, y0
                for i in range(1, exterior_coords_2d.shape[0]):
                    x2, y2 = exterior_coords_2d[i]
                    if not (math.isclose(current_x, x2) and math.isclose(current_y, y2)):
                        d = math.hypot(x2 - current_x, y2 - current_y)
                        if d > 1e-6:
                            e_pos += d * extrusion_per_mm
                            gcode_lines.append(f"G1 X{x2:.3f} Y{y2:.3f} A{pitch_deg:.2f} B{roll_deg:.2f} C{yaw_deg:.2f} E{e_pos:.5f} F{feedrate}")
                        current_x, current_y = x2, y2
            
            for interior in poly.interiors:
                interior_coords_2d = np.array(interior.coords)
                if interior_coords_2d.shape[0] >= 2:
                    z_col = np.full((interior_coords_2d.shape[0], 1), z)
                    visualization_paths.append(np.hstack((interior_coords_2d, z_col)))
                    
                    x0, y0 = interior_coords_2d[0]
                    gcode_lines.append(f"G1 X{x0:.3f} Y{y0:.3f} F{travel_feed}")
                    current_x, current_y = x0, y0
                    for i in range(1, interior_coords_2d.shape[0]):
                        x2, y2 = interior_coords_2d[i]
                        if not (math.isclose(current_x, x2) and math.isclose(current_y, y2)):
                            d = math.hypot(x2 - current_x, y2 - current_y)
                            if d > 1e-6:
                                e_pos += d * extrusion_per_mm
                                gcode_lines.append(f"G1 X{x2:.3f} Y{y2:.3f} A{pitch_deg:.2f} B{roll_deg:.2f} C{yaw_deg:.2f} E{e_pos:.5f} F{feedrate}")
                            current_x, current_y = x2, y2
    
    gcode_lines += [
        f"G1 Z{(z_levels[-1] + 5) if z_levels else 5} F5000",
        "M84 ; Disable steppers",
        "; End of 6DOF G-code",
    ]
    
    return "\n".join(gcode_lines), visualization_paths

# ... (keep slice_mesh_to_polygons, generate_infill, helpers, and __main__ the same) ...



# --- Main Execution (for command-line use) ---
if __name__ == "__main__":
    import argparse # Need argparse if running standalone

    p = argparse.ArgumentParser(
        description="Slice an STL into perimeter and infill G-code.")
    p.add_argument("input", help="Input STL file")
    p.add_argument("output", help="Output G-code file")
    p.add_argument("--layer-height", "-l", type=float, default=0.2,
                   help="Layer height (mm)")
    p.add_argument("--extrusion", "-e", type=float, default=0.05,
                   help="Extrusion amount (E units) per mm of XY travel")
    p.add_argument("--feed", "-f", type=int, default=1500,
                   help="Printing feedrate (mm/min)")
    p.add_argument("--travel-feed", type=int, default=3000,
                   help="Travel moves feedrate (mm/min)")
    p.add_argument("--infill-spacing", type=float, default=2.0,
                   help="Spacing between infill lines (mm). 0 to disable.")
    p.add_argument("--infill-angle", type=float, default=45.0,
                   help="Angle of infill lines (degrees from X-axis)")
    # Removed --plot as visualization is now in the GUI
    args = p.parse_args()

    try:
        # Use the new slicing function
        z_levels, layer_polygons = slice_mesh_to_polygons(args.input, args.layer_height)

        if not z_levels or not layer_polygons or all(not lp for lp in layer_polygons):
             print("Warning: No slice data generated. Check mesh integrity and layer height.", file=sys.stderr)
             sys.exit(0)

        print("Generating G-code with infill...")
        # Use the new G-code generation function
        gcode = generate_gcode_with_infill(z_levels, layer_polygons,
                                           feedrate=args.feed,
                                           extrusion_per_mm=args.extrusion,
                                           travel_feed=args.travel_feed,
                                           infill_spacing=args.infill_spacing,
                                           infill_angle=args.infill_angle)
        try:
            with open(args.output, "w") as fp:
                fp.write(gcode)
            print(f"Wrote {len(z_levels)} layers with infill to {args.output}")
        except IOError as e:
            print(f"Error writing G-code file '{args.output}': {e}", file=sys.stderr)
            sys.exit(1)

    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during slicing process: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        sys.exit(1)
