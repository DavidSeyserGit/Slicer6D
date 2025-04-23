import numpy as np
import pyvista as pv
import vtk
from math import sqrt

def normalize(v):
    """Normalize vector v. Falls der Normwert zu klein ist, wird
    ein Teiler 1 verwendet."""
    n = np.linalg.norm(v)
    return v / (n if n > 1e-12 else 1)


def determine_print_bed(mesh):
    """
    Return (normal, point) of the print‑bed as the XY plane at mesh.min_z.
    """
    b = mesh.bounds
    min_z = b[4]
    return np.array([0, 0, 1]), np.array([0, 0, min_z])


def detect_overhangs(mesh, angle_threshold=45, print_bed=None):
    """
    Return face indices whose normals point downward within angle_threshold
    of straight down and whose face center is strictly above the print bed.
    """
    if print_bed is None:
        print_bed = determine_print_bed(mesh)
    bed_z = print_bed[1][2]

    normals = mesh.face_normals
    cos_t = np.cos(np.radians(angle_threshold))
    # Gesichtsnormale, die nach unten zeigen: normal_z < -cos_t
    candidates = np.where(normals[:, 2] < -cos_t)[0]

    centers = mesh.cell_centers().points
    tol = 1e-3
    overhangs = [f for f in candidates if centers[f][2] > bed_z + tol]
    return np.array(overhangs, dtype=int)


def tetrahedralize_mesh(surface_mesh):
    """
    Erzeugt ein tetraedrisches Volumenmesh aus einem Oberflächenmesh.
    Hier wird pyvista's delaunay_3d() genutzt. Beachte, dass der Parameter
    `alpha` ggf. angepasst werden muss, um alle relevanten Elemente zu erhalten.
    """
    # Erzeuge ein dichtes Punktwolken-Set aus der Oberfläche.
    points = surface_mesh.points
    # Für robuste Tetraedrierung kann ein alpha-wert sinnvoll sein; hier als Startwert
    tet_mesh = surface_mesh.delaunay_3d(alpha=1.0)
    # Optional: Extrahiere das Volumen (beispielsweise mittels thresholding).
    tet = tet_mesh.extract_geometry()
    return tet


def deform_mesh(tet_mesh,
                            angle_threshold=45,
                            print_bed=None,
                            support_distance=5.0,
                            use_ray=False,
                            max_iterations=50):
    """
    Ähnlich wie beim ursprünglichen Deformationsansatz, 
    wird hier ein tetraedrisches Mesh deformiert.

    Wichtig: Wir dehnen dabei die Punkte des Volumenmeshs, orientiert an
    den überhängenden Oberflächenelementen (die extrahiert werden). Danach
    wird ein sauberes Oberflächenmesh extrahiert.
    """
    if print_bed is None:
        print_bed = determine_print_bed(tet_mesh)
    bed_z = print_bed[1][2]
    tol = 1e-6

    # Kopie des tetraedrischen Meshs
    working = tet_mesh.copy()
    all_modified = set()

    # Erstelle die Face-zu-Punkte Zuordnung aus dem extrahierten Oberflächenmesh.
    # Wir extrahieren zunächst die Oberfläche:
    surf = working.extract_surface().triangulate().clean()
    faces = surf.faces
    face_map = {}
    idx = fi = 0
    while idx < len(faces):
        n = faces[idx]
        vs = faces[idx + 1: idx + 1 + n].tolist()
        face_map[fi] = vs
        idx += n + 1
        fi += 1

    # Berechne den Schritt: 2% der Modellhöhe
    pts = working.points.copy()
    zmin, zmax = pts[:, 2].min(), pts[:, 2].max()
    step = (zmax - zmin) * 0.02

    # Haupt-Deformations-Schleife
    for iteration in range(max_iterations):
        # 1) Extrahiere Oberflächensicht und bestimme überhängende Bereiche
        surf = working.extract_surface().triangulate().clean()
        overhangs = detect_overhangs(surf, angle_threshold, print_bed)

        # 2) Optional: Ray‐Filterung zur Prüfung auf Unterstützung
        if use_ray and overhangs.size > 0:
            centers = surf.cell_centers().points
            keep = []
            for f in overhangs:
                c = centers[f]
                pts_ray, _ = surf.ray_trace(c, [0, 0, -1], first_point=True)
                if pts_ray.shape[0] == 0 or (
                    np.linalg.norm(pts_ray[0] - c) > support_distance
                ):
                    keep.append(f)
            overhangs = np.array(keep, dtype=int)

        # Abbruch, wenn keine überhängenden Flächen verbleiben:
        if overhangs.size == 0:
            break

        # 3) Bestimme, welche Punkte bewegt werden sollen
        # Dazu verwenden wir die Face-to-Punkte Zuordnung des aktuellen Oberflächenshapes.
        pts = working.points.copy()
        verts_to_move = set(v for f in overhangs for v in face_map.get(f, []))
        surface_normals = surf.face_normals

        for v in verts_to_move:
            # Basis-Punkte (z. B. am Druckbett) werden nicht verändert.
            if pts[v, 2] <= bed_z + tol:
                continue

            # Mittelwert der Normale aller überhängenden Flächen, die den Punkt enthalten.
            n_sum = np.zeros(3)
            cnt = 0
            for f in overhangs:
                if v in face_map.get(f, []):
                    n_sum += surface_normals[f]
                    cnt += 1
            if cnt == 0:
                continue

            avg_n = normalize(n_sum / cnt)
            # Extrahiere den horizontalen Anteil und bewege gegen die Richtung
            horiz = avg_n.copy()
            horiz[2] = 0
            nh = np.linalg.norm(horiz)
            if nh > tol:
                horiz /= nh
                pts[v, 0] -= step * horiz[0]
                pts[v, 1] -= step * horiz[1]
            else:
                # Zu nahezu horizontaler Orientierung: sanfte Vertikalverschiebung
                pts[v, 2] -= step * 0.5

            all_modified.add(v)

        working.points = pts
        working.clean(inplace=True)

    # Schritt 4: Extrahiere zum Schluss eine saubere Oberfläche
    surf = working.extract_surface().triangulate().clean()
    filler = vtk.vtkFillHolesFilter()
    filler.SetInputData(surf)
    filler.SetHoleSize(1e6)
    filler.Update()
    filled = pv.wrap(filler.GetOutput())
    filled.clean(inplace=True)

    return filled, all_modified


if __name__ == "__main__":
    # Beispiel: Laden eines Oberflächenmeshs aus einer STL-Datei:
    surface_mesh = pv.read("input_model.stl")

    # Schritt 1: Erzeugen eines tetraedrischen Meshes
    tet_mesh = tetrahedralize_mesh(surface_mesh)

    # Schritt 2: Deformation des Volumenmeshs
    deformed_surface, modified_vertices = deform_mesh(tet_mesh,
                                                                  angle_threshold=45,
                                                                  support_distance=5.0,
                                                                  use_ray=False,
                                                                  max_iterations=50)

    # Exportiere das deforme Modell
    deformed_surface.save("deformed_model.stl")
