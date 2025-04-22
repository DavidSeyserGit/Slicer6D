import numpy as np
import pyvista as pv
import pytest
from deformer import (
    normalize,
    determine_print_bed,
    detect_overhangs,
    deform_mesh,
)

# Helper function to create a simple mesh (a cube)
def create_cube():
    mesh = pv.Cube()
    return mesh


# Helper function to create a mesh with a known overhang
def create_overhang_mesh():
    # A simple shape with an overhang face pointing downwards
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0.5, 0.5, 1],  # Point in the middle, above the base
    ])
    faces = np.array([
        4, 0, 1, 2, 3,  # Base face
        3, 0, 1, 4,  # Overhang face 1
        3, 1, 2, 4,  # Overhang face 2
        3, 2, 3, 4,  # Overhang face 3
        3, 3, 0, 4,  # Overhang face 4
    ])
    mesh = pv.PolyData(points, faces)
    return mesh


# Test normalize function
def test_normalize():
    v1 = np.array([1, 2, 3])
    n1 = normalize(v1)
    assert np.isclose(np.linalg.norm(n1), 1.0)

    v2 = np.array([0, 0, 0])
    n2 = normalize(v2)
    assert np.allclose(n2, np.array([0, 0, 0]))


# Test determine_print_bed function
def test_determine_print_bed():
    mesh = create_cube()
    normal, point = determine_print_bed(mesh)
    assert np.allclose(normal, np.array([0, 0, 1]))
    assert np.allclose(point, np.array([-0.5, -0.5, -0.5]))

    mesh_shifted = mesh.translate([0, 0, 5])
    normal_shifted, point_shifted = determine_print_bed(mesh_shifted)
    assert np.allclose(normal_shifted, np.array([0, 0, 1]))
    assert np.allclose(point_shifted, np.array([-0.5, -0.5, 4.5]))


# Test detect_overhangs function
def test_detect_overhangs_cube():
    mesh = create_cube()
    overhangs = detect_overhangs(mesh)
    assert len(overhangs) == 0  # A cube should have no overhangs at 45 degrees


def test_detect_overhangs_overhang_mesh():
    mesh = create_overhang_mesh()
    overhangs = detect_overhangs(mesh, angle_threshold=45)
    # Adjust expected overhang faces based on the specific mesh creation
    # The faces with the tip vertex ([0.5, 0.5, 1]) should be overhangs
    # Based on the face definition, these are faces 1, 2, 3, and 4
    expected_overhangs = np.array([1, 2, 3, 4])
    assert np.array_equal(np.sort(overhangs), np.sort(expected_overhangs))


def test_detect_overhangs_with_print_bed():
    mesh = create_overhang_mesh()
    # Set print bed above the overhang
    print_bed = (np.array([0, 0, 1]), np.array([0, 0, 1.1]))
    overhangs = detect_overhangs(mesh, print_bed=print_bed)
    assert len(overhangs) == 0  # Overhang faces are now below the print bed


# Test deform_mesh function
def test_deform_mesh_no_overhangs():
    mesh = create_cube()
    deformed_mesh, modified_verts = deform_mesh(mesh)
    assert np.allclose(
        deformed_mesh.points, mesh.points
    )  # Should not change if no overhangs
    assert len(modified_verts) == 0


def test_deform_mesh_with_overhangs():
    mesh = create_overhang_mesh()
    deformed_mesh, modified_verts = deform_mesh(
        mesh, max_iterations=10
    )  # Limit iterations for test

    # Check that some vertices were modified
    assert len(modified_verts) > 0

    # Check if the mesh has changed (should be different from the original)
    assert not np.allclose(deformed_mesh.points, mesh.points)

    # Basic check on bounding box - the mesh should likely shrink/change shape
    assert not np.allclose(deformed_mesh.bounds, mesh.bounds)

    # Verify the final mesh has no overhangs after deformation (within tolerance)
    final_overhangs = detect_overhangs(
        deformed_mesh
    )  # Use default 45 degrees
    assert len(final_overhangs) == 0 or (
        len(final_overhangs) > 0 and len(final_overhangs) < len(detect_overhangs(mesh))
    )  # Should reduce or eliminate overhangs


def test_deform_mesh_with_ray_filter():
    mesh = create_overhang_mesh()
    # The ray filter should theoretically keep the overhangs in this simple mesh
    deformed_mesh_ray, modified_verts_ray = deform_mesh(
        mesh, use_ray=True, support_distance=0.1, max_iterations=10
    )

    assert len(modified_verts_ray) > 0
    assert not np.allclose(deformed_mesh_ray.points, mesh.points)


def test_deform_mesh_with_custom_angle():
    mesh = create_overhang_mesh()
    # With a higher angle threshold, some initial overhangs might not be detected
    deformed_mesh_angle, modified_verts_angle = deform_mesh(
        mesh, angle_threshold=60, max_iterations=10
    )

    assert len(modified_verts_angle) > 0
    assert not np.allclose(deformed_mesh_angle.points, mesh.points)
