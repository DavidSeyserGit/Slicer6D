# deformer.py
import numpy as np
import pyvista as pv
import vtk
from math import sqrt

def normalize(v):
    n = np.linalg.norm(v)
    return v/(n if n>1e-12 else 1)

def determine_print_bed(mesh):
    """Return (normal, point) of the print‑bed as the XY plane at mesh.min_z."""
    b = mesh.bounds
    min_z = b[4]
    return np.array([0,0,1]), np.array([0,0,min_z])

def detect_overhangs(mesh, angle_threshold=45, print_bed=None):
    """
    Return face indices whose normals point downward within angle_threshold
    of straight down, and sit strictly above the print bed.
    """
    if print_bed is None:
        print_bed = determine_print_bed(mesh)
    bed_z = print_bed[1][2]

    normals = mesh.face_normals
    # angle between face normal and +Z
    # we want normals pointing *below* horizontal: normal_z < 0
    # and with |angle| > angle_threshold:
    cos_t = np.cos(np.radians(angle_threshold))
    # normal·(-Z) = -normal_z  > cos_t  => normal_z < -cos_t
    candidates = np.where(normals[:,2] < -cos_t)[0]

    centers = mesh.cell_centers().points
    tol = 1e-3
    overhangs = [f for f in candidates if centers[f][2] > bed_z + tol]
    return np.array(overhangs, dtype=int)

def deform_mesh(mesh,
                angle_threshold=45,
                print_bed=None,
                support_distance=5.0,
                use_ray=False,
                max_iterations=50):
    """
    Iteratively tilt overhang faces back to vertical by moving vertices
    *until* no overhang faces remain.  No ‘strength’ parameter—each pass
    applies a fixed small tilt.
    """
    if print_bed is None:
        from deformer import determine_print_bed
        print_bed = determine_print_bed(mesh)
    bed_z = print_bed[1][2]
    tol   = 1e-6

    working = mesh.copy()
    all_modified = set()

    # Precompute face→vertex mapping once
    faces = working.faces
    face_map = {}
    idx = fi = 0
    while idx < len(faces):
        n = faces[idx]
        vs = faces[idx+1:idx+1+n].tolist()
        face_map[fi] = vs
        idx += n+1
        fi  += 1

    # We choose a fixed per-iteration tilt amount: 2% of model height
    zmin, zmax = working.points[:,2].min(), working.points[:,2].max()
    step = (zmax - zmin) * 0.02

    # Main loop
    for iteration in range(max_iterations):
        # 1) find all overhang faces by angle
        from deformer import detect_overhangs
        overhangs = detect_overhangs(working, angle_threshold, print_bed)

        # 2) optional ray‐filter
        if use_ray and overhangs.size > 0:
            centers = working.cell_centers().points
            keep = []
            for f in overhangs:
                c = centers[f]
                pts, _ = working.ray_trace(c, [0,0,-1], first_point=True)
                if pts.shape[0] == 0 or np.linalg.norm(pts[0]-c) > support_distance:
                    keep.append(f)
            overhangs = np.array(keep, dtype=int)

        # if nothing to fix, stop
        if overhangs.size == 0:
            break

        # 3) tilt each offending vertex once
        normals = working.face_normals
        pts = working.points.copy()
        verts_to_move = set(v for f in overhangs for v in face_map[f])

        for v in verts_to_move:
            # skip base vertices
            if pts[v,2] <= bed_z + tol:
                continue

            # average normals of overhang faces sharing v
            n_sum = np.zeros(3)
            cnt   = 0
            for f in overhangs:
                if v in face_map[f]:
                    n_sum += normals[f]
                    cnt  += 1
            if cnt == 0:
                continue

            avg_n = normalize(n_sum / cnt)
            # extract horizontal component and tilt against it
            horiz = avg_n.copy()
            horiz[2] = 0
            nh = np.linalg.norm(horiz)
            if nh > tol:
                horiz /= nh
                pts[v,0] -= step * horiz[0]
                pts[v,1] -= step * horiz[1]
            else:
                # if nearly horizontal, fallback to small downward push
                pts[v,2] -= step*0.5

            all_modified.add(v)

        working.points = pts
        working.clean(inplace=True)

    # final manifold cleanup
    surf = working.extract_surface().triangulate().clean()
    filler = vtk.vtkFillHolesFilter()
    filler.SetInputData(surf)
    filler.SetHoleSize(1e6)
    filler.Update()
    filled = pv.wrap(filler.GetOutput())
    filled.clean(inplace=True)

    return filled, all_modified