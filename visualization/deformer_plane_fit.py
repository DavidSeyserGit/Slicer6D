# visualize_plane_fit.py
import numpy as np
import pyvista as pv
from numpy.linalg import svd
import logging

# --- Setup Logging (optional, just for potential debug messages from planeFit) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- planeFit Function (copied from deformer.py) ---
# This function needs to be included in this file or imported if you make it a module.
# Copying it here makes the script standalone.
def planeFit(points):
    """Fit an d-dimensional plane to points (d, N)."""
    # Ensure points have correct shape (d, N), assuming 3D points initially as (N, 3)
    # Common user input is (N, dimensions). The function internally works with (dimensions, N).
    original_shape = points.shape
    if len(original_shape) == 2 and original_shape[1] == 3 and original_shape[0] != 3:
         # Assume input was (N, 3), transpose to (3, N)
         points = points.T
         #log.debug("planeFit: Transposed input points from (N, 3) to (3, N)") # Uncomment for more logging
    elif len(original_shape) == 2 and original_shape[0] != 3 and original_shape[1] != 3:
         log.warning(f"planeFit: Unexpected point shape {original_shape}. Proceeding assuming (d, N).")
         points = np.reshape(points, (points.shape[0], -1)) # Ensure shape is (d, N) even if N=1
    elif len(original_shape) == 1 and original_shape[0] == 3:
         # Single point input (3,), reshape to (3, 1)
         points = points.reshape(3, 1)
         #log.debug("planeFit: Reshaped single point from (3,) to (3, 1)") # Uncomment for more logging
    else:
        # Assume input is already (d, N) or something we can't guess, reshape to (d, N) format
        points = np.reshape(points, (points.shape[0], -1))
        #log.debug(f"planeFit: Used reshape(-1) on input shape {original_shape}") # Uncomment for more logging


    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    M = np.dot(x, x.T)
    normal = svd(M)[0][:, -1]
    normal = normal / (np.linalg.norm(normal) if np.linalg.norm(normal) > 1e-12 else 1)

    return ctr, normal # Return the centroid and the unit normal vector


# --- Visualization Script ---

def visualize_plane_fitting():
    """
    Visualizes the planeFit function with different sets of points.
    Uses pyvista to display the points, calculated center, and normal vector.
    """
    # Create a plotter with subplots for multiple visualizations
    # We'll arrange them in a 2x2 grid for example
    plotter = pv.Plotter(shape=(2, 2)) # 2 rows, 2 columns

    # --- Case 1: Points on a simple horizontal plane (XY plane) ---
    plotter.subplot(0, 0) # Select top-left subplot
    plotter.add_title("Case 1: Horizontal Plane (XY)")

    # Define points (N, 3 format - easier to type)
    points1_input = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0.5, 0.5, 0] # Add an extra point to show it still works
    ])

    # Call planeFit (it handles the transpose internally if needed)
    center1, normal1 = planeFit(points1_input)

    # Add points to the plotter
    point_cloud1 = pv.PolyData(points1_input)
    plotter.add_mesh(point_cloud1, color='blue', point_size=15, render_points_as_spheres=True, label="Points")

    # Add center point
    center_mesh1 = pv.PolyData([center1]) # PyVista expects N, 3 even for one point
    plotter.add_mesh(center_mesh1, color='red', point_size=25, render_points_as_spheres=True, label="Center")

    # Add normal vector as an arrow
    arrow_scale = 0.5 # Make arrow size consistent
    normal_arrow1 = pv.Arrow(center1, normal1 * arrow_scale)
    plotter.add_mesh(normal_arrow1, color='green', label="Normal")

    # Add a grid representing the fitted plane (optional, but helpful)
    # Need to create a plane actor based on the center and normal
    plane_actor1 = pv.Plane(center=center1, direction=normal1, i_size=1.5, j_size=1.5)
    plotter.add_mesh(plane_actor1, color='lightblue', opacity=0.5, show_edges=True)

    plotter.add_legend() # Add legend to this subplot
    plotter.view_isometric() # Set a good viewing angle
    plotter.show_bounds() # Show axes and boundaries

    # --- Case 2: Points on a tilted plane (similar to Y=X) ---
    plotter.subplot(0, 1) # Select top-right subplot
    plotter.add_title("Case 2: Tilted Plane (Y=X)")

    points2_input = np.array([
        [0, 0, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
        [0.5, 0.5, 0.5],
        [0.2, 0.2, 0.8] # Add some noise/variation
    ])

    center2, normal2 = planeFit(points2_input)

    point_cloud2 = pv.PolyData(points2_input)
    plotter.add_mesh(point_cloud2, color='blue', point_size=15, render_points_as_spheres=True)

    center_mesh2 = pv.PolyData([center2])
    plotter.add_mesh(center_mesh2, color='red', point_size=25, render_points_as_spheres=True)

    normal_arrow2 = pv.Arrow(center2, normal2 * arrow_scale)
    plotter.add_mesh(normal_arrow2, color='green')

    plane_actor2 = pv.Plane(center=center2, direction=normal2, i_size=1.5, j_size=1.5)
    plotter.add_mesh(plane_actor2, color='lightblue', opacity=0.5, show_edges=True)

    plotter.view_isometric()
    plotter.show_bounds()

    # --- Case 3: Points on a vertical plane (XZ plane) ---
    plotter.subplot(1, 0) # Select bottom-left subplot
    plotter.add_title("Case 3: Vertical Plane (XZ)")

    points3_input = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 1],
        [0.5, 0, 0.5]
    ])

    center3, normal3 = planeFit(points3_input)

    point_cloud3 = pv.PolyData(points3_input)
    plotter.add_mesh(point_cloud3, color='blue', point_size=15, render_points_as_spheres=True)

    center_mesh3 = pv.PolyData([center3])
    plotter.add_mesh(center_mesh3, color='red', point_size=25, render_points_as_spheres=True)

    normal_arrow3 = pv.Arrow(center3, normal3 * arrow_scale)
    plotter.add_mesh(normal_arrow3, color='green')

    plane_actor3 = pv.Plane(center=center3, direction=normal3, i_size=1.5, j_size=1.5)
    plotter.add_mesh(plane_actor3, color='lightblue', opacity=0.5, show_edges=True)

    plotter.view_isometric()
    plotter.show_bounds()

    # --- Case 4: Points with some noise, roughly planar ---
    plotter.subplot(1, 1) # Select bottom-right subplot
    plotter.add_title("Case 4: Noisy Plane")

    points4_input = np.array([
        [0, 0, 0],
        [1, 0, 0.1],
        [0, 1, -0.05],
        [1, 1, 0.05],
        [0.5, 0.5, 0.01],
        [0.2, 0.8, -0.03],
        [0.8, 0.2, 0.08]
    ]) + np.random.randn(7, 3) * 0.05 # Add random noise

    center4, normal4 = planeFit(points4_input)

    point_cloud4 = pv.PolyData(points4_input)
    plotter.add_mesh(point_cloud4, color='blue', point_size=15, render_points_as_spheres=True)

    center_mesh4 = pv.PolyData([center4])
    plotter.add_mesh(center_mesh4, color='red', point_size=25, render_points_as_spheres=True)

    normal_arrow4 = pv.Arrow(center4, normal4 * arrow_scale)
    plotter.add_mesh(normal_arrow4, color='green')

    plane_actor4 = pv.Plane(center=center4, direction=normal4, i_size=1.5, j_size=1.5)
    plotter.add_mesh(plane_actor4, color='lightblue', opacity=0.5, show_edges=True)


    plotter.view_isometric()
    plotter.show_bounds()


    # --- Show the Plotter Window ---
    plotter.show()

# --- Run the visualization ---
if __name__ == "__main__":
    visualize_plane_fitting()
