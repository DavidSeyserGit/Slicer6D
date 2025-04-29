import pytest
import numpy as np
import pyvista as pv # Needed for mesh creation
from deformer import normalize, planeFit, encode_object, decode_object

def test_normalize():
    v1 = np.array([1.0, 0.0, 0.0])
    assert np.allclose(normalize(v1), [1.0, 0.0, 0.0]) # Already normalized

    v2 = np.array([0.0, 5.0, 0.0])
    assert np.allclose(normalize(v2), [0.0, 1.0, 0.0])

    v3 = np.array([3.0, 4.0, 0.0])
    assert np.allclose(normalize(v3), [0.6, 0.8, 0.0]) # 5 units long -> 3/5, 4/5

    v4 = np.array([0.0, 0.0, 0.0])
    assert np.allclose(normalize(v4), [0.0, 0.0, 0.0]) # Handle zero vector


def test_planeFit():
    # --- Test Case 1: Simple 2D plane (on XY plane) ---
    # Define 4 points that lie perfectly on the Z=0 plane (the XY plane).
    # The points are (0,0,0), (1,0,0), (0,1,0), and (1,1,0).
    # The .T transposes the array. planeFit expects points in (dimensions, num_points) format.
    points_xy = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]).T

    # Call the planeFit function with these points.
    # It should return the calculated centroid (center_xy) and the plane's normal vector (normal_xy).
    center_xy, normal_xy = planeFit(points_xy)

    # Assertions for Test Case 1:
    # Check if the calculated centroid is approximately [0.5, 0.5, 0.0].
    # This is the average of the input points:
    # np.allclose is used for floating-point comparisons to account for tiny calculation errors.
    assert np.allclose(center_xy, [0.5, 0.5, 0.0])

    # Check if the magnitude (length) of the calculated normal vector is approximately 1.0.
    # planeFit should return a unit normal vector.
    # np.linalg.norm calculates the vector's length.
    # np.isclose is used for comparing single floating-point numbers.
    assert np.isclose(np.linalg.norm(normal_xy), 1.0)

    # Check the components of the normal vector.
    # The normal to the XY plane (Z=0) is along the Z-axis. It can be [0,0,1] or [0,0,-1].
    # np.abs() takes the absolute value of each component, so both [0,0,1] and [0,0,-1]
    # become [0,0,1] after taking the absolute value.
    # This assertion checks that the absolute values of the normal components match [0.0, 0.0, 1.0].
    assert np.allclose(np.abs(normal_xy), [0.0, 0.0, 1.0])

    # --- Test Case 2: Simple 3D plane (e.g., on Z=5 plane) ---
    points_z5 = np.array([[0, 0, 5], [1, 0, 5], [0, 1, 5], [1, 1, 5]]).T

    # Call the planeFit function for these points.
    center_z5, normal_z5 = planeFit(points_z5)

    # Assertions for Test Case 2:
    # Check if the calculated centroid is approximately [0.5, 0.5, 5.0].
    assert np.allclose(center_z5, [0.5, 0.5, 5.0])

    # Check if the normal vector is a unit vector.
    assert np.isclose(np.linalg.norm(normal_z5), 1.0)

    # Check the components of the normal vector.
    assert np.allclose(np.abs(normal_z5), [0.0, 0.0, 1.0])

    # --- Test Case 3: Tilted plane (e.g., Y=X) ---
    # Define 4 points that lie perfectly on a tilted plane.
    # The points are (0,0,0), (1,1,0), (0,0,1), and (1,1,1).
    # These points define the plane x - y = 0.
    points_tilted = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1]]).T

    # Call the planeFit function for these points.
    center_tilted, normal_tilted = planeFit(points_tilted)

    # Assertions for Test Case 3:
    # Check if the calculated centroid is approximately [0.5, 0.5, 0.5].
    assert np.allclose(center_tilted, [0.5, 0.5, 0.5])

    # Check if the normal vector is a unit vector.
    assert np.isclose(np.linalg.norm(normal_tilted), 1.0)

    # Check the components of the normal vector.
    assert np.allclose(np.abs(normal_tilted[[0, 1]]), np.array([1/np.sqrt(2), 1/np.sqrt(2)]))

    # Check that the Z component of the normal vector is approximately 0.0.
    assert np.isclose(normal_tilted[2], 0.0)
