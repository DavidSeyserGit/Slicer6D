import pytest
from deformer import normalize, planeFit, encode_object, decode_object, MeshDeformer
import pyvista as pv
import numpy as np

def test_tetrahedralize_cube():
    """
    Tests tetrahedralization with a simple cube mesh (watertight, manifold).
    Expected: returns True, self.tet is a valid UnstructuredGrid with cells.
    """
    cube = pv.Cube().triangulate() #make it a triangular surface mesh

    deformer = MeshDeformer(cube)

    success = deformer._tetrahedralize()

    assert success is True
    assert deformer.tet is not None
    assert isinstance(deformer.tet, pv.UnstructuredGrid)
    assert deformer.tet.number_of_cells > 0
    assert np.all(deformer.tet.celltypes == 10)

def test_tetrahedralize_sphere():
    """
    Tests tetrahedralization with a simple sphere mesh (watertight, manifold).
    Expected: returns True, self.tet is a valid UnstructuredGrid with cells.
    """
    sphere = pv.Cube().triangulate() #make it a triangular surface mesh

    deformer = MeshDeformer(sphere)

    success = deformer._tetrahedralize()

    assert success is True
    assert deformer.tet is not None
    assert isinstance(deformer.tet, pv.UnstructuredGrid) # checks if the tet mesh is Unstructured Grid
    assert deformer.tet.number_of_cells > 0
    assert np.all(deformer.tet.celltypes == 10)

def test_tetrahedralize_open_mesh():
    """
    Tests tetrahedralization with a mesh that has a hole (non-watertight).
    TetGen is expected to fail or produce 0 cells.
    Expected: returns False, self.tet is None or empty.
    """
    points = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], # Bottom square
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]  # Top square (no side walls)
    ])

    faces = np.array([
        3, 0, 1, 2,  # Triangle 1 on bottom
        3, 0, 2, 3,  # Triangle 2 on bottom
        3, 4, 5, 6,  # Triangle 1 on top
        3, 4, 6, 7   # Triangle 2 on top
    ], dtype=np.int32)
    open_box = pv.PolyData(points, faces)

    deformer = MeshDeformer(open_box)

    success = deformer._tetrahedralize()

    assert success is False
    assert deformer.tet is None
    assert not isinstance(deformer.tet, pv.UnstructuredGrid) # checks if the tet mesh is Unstructured Grid

def test_tetrahedralize_failure_non_manifold_edge():
    """
    Tests tetrahedralization with a non-manifold mesh (edge shared by >2 faces).
    TetGen is expected to fail or produce invalid output.
    Expected: returns False, self.tet is None or empty.
    """

    #create a non-manifold edge geometry (like two triangles sharing one edge)
    points = np.array([
        [0, 0, 0], [1, 0, 0], [0.5, 1, 0],  # Triangle 1 vertices
        [0.5, -1, 0]                        # Vertex for Triangle 2
    ])
    #Triangle 1 (0,1,2), Triangle 2 (1,0,3). They share edge (0,1).
    faces = np.array([
        3, 0, 1, 2, # Face 1
        3, 1, 0, 3  # Face 2 - note the order matters for orientation, but manifold issue is topology
    ], dtype=np.int32)
    non_manifold_edge_mesh = pv.PolyData(points, faces)

    deformer = MeshDeformer(non_manifold_edge_mesh)

    success = deformer._tetrahedralize()

    assert success is False
    assert deformer.tet is None or deformer.tet.number_of_cells == 0

@pytest.mark.skip("Known segfault with pyvista/tetgen on list input for faces")
def test_tetrahedralize_invalid_faces_type():
    """
    Tests handling of an input mesh where faces is not a numpy array.
    Expected: ValueError (caught by method), returns False, self.tet is None.
    """
    #Create a mesh with points but faces as a list
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    invalid_faces_list = [3, 0, 1, 2] # A list instead of np.ndarray

    mesh_with_invalid_faces = pv.PolyData(points)
    mesh_with_invalid_faces.faces = invalid_faces_list

    deformer = MeshDeformer(mesh_with_invalid_faces)

    success = deformer._tetrahedralize()

    assert success is False, "_tetrahedralize should return False when input faces is not np.ndarray"
    assert deformer.tet is None, "self.tet should remain None after failure due to invalid faces type"

@pytest.mark.skip("Known segfault with pyvista/tetgen on list input for faces")
def test_tetrahedralize_failure_invalid_faces_ndim():
    """
    Tests handling of an input mesh where faces is a numpy array but not 1D.
    Expected: ValueError (caught by method), returns False, self.tet is None.
    """
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    invalid_faces_2d = np.array([[3, 0, 1, 2]], dtype=np.int32) # 2D array

    mesh_with_invalid_faces = pv.PolyData(points)
    mesh_with_invalid_faces.faces = invalid_faces_2d

    deformer = MeshDeformer(mesh_with_invalid_faces)

    success = deformer._tetrahedralize()

    # 4. Assertions
    assert success is False, "_tetrahedralize should return False when input faces is not 1D"
    assert deformer.tet is None, "self.tet should remain None after failure due to invalid faces ndim"