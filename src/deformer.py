# deformer.py
import numpy as np
import pyvista as pv
import tetgen
import networkx as nx
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix, csr_matrix
import pickle
import base64
import time
import logging


# Implementation like in S4
# https://github.com/jyjblrd/S4_Slicer

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Helper Functions (Remain outside class) ---

def normalize(v):
    """Normalizing vector."""
    n = np.linalg.norm(v)
    return v / (n if n > 1e-12 else 1)

def determine_print_bed(mesh):
    """Return (normal, point) of the print‑bed as the XY plane at mesh.min_z."""
    b = mesh.bounds
    min_z = b[4]
    return np.array([0, 0, 1]), np.array([0, 0, min_z])

def encode_object(obj):
    return base64.b64encode(pickle.dumps(obj)).decode('utf-8')

def decode_object(encoded_str):
    return pickle.loads(base64.b64decode(encoded_str))

def planeFit(points):
    """Fit an d-dimensional plane to points (d, N)."""
    from numpy.linalg import svd
    points = np.reshape(points, (np.shape(points)[0], -1))
    assert points.shape[0] <= points.shape[1], f"Only {points.shape[1]} points in {points.shape[0]} dimensions."
    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    M = np.dot(x, x.T)
    return ctr, svd(M)[0][:, -1]
# --- Constants ---
UP_VECTOR = np.array([0, 0, 1])
# --- MeshDeformer Class ---
class MeshDeformer:
    """Encapsulates the tetrahedral mesh deformation process."""
    def __init__(self, mesh: pv.PolyData, **kwargs):
        self.input_mesh = mesh
        self.params = self._process_params(kwargs)
        self.verbose = kwargs.get('verbose', True) # Control logging level

        # Intermediate state variables
        self.tet = None
        self.undeformed_tet = None
        self.neighbour_dict = None
        self.cell_neighbour_graph = None
        self.bottom_cells = None
        self.optimized_rotation_field_rad = None
        self.new_vertices = None
        self.deformed_surface = None
        self.success = False

        # GIF saving state
        self._plotter = None
        self._gif_frame_count = 0

    def _process_params(self, kwargs):
        """Extracts and converts parameters."""
        params = {
            'max_overhang_rad': np.deg2rad(kwargs.get('max_overhang_deg', 30.0)),
            'neighbour_loss_weight': kwargs.get('neighbour_loss_weight', 20.0),
            'rotation_multiplier': kwargs.get('rotation_multiplier', 2.0),
            'initial_rotation_field_smoothing': kwargs.get('initial_rotation_field_smoothing', 30),
            'set_initial_rotation_to_zero': kwargs.get('set_initial_rotation_to_zero', False),
            'steep_overhang_compensation': kwargs.get('steep_overhang_compensation', True),
            'max_pos_rotation_rad': np.deg2rad(kwargs.get('max_pos_rotation_deg', 360.0)),
            'max_neg_rotation_rad': np.deg2rad(kwargs.get('max_neg_rotation_deg', -360.0)),
            'optimization_iterations': kwargs.get('optimization_iterations', 25),
            'deformation_iterations': kwargs.get('deformation_iterations', 25),
            'part_offset': np.array(kwargs.get('part_offset', [0., 0., 0.])),
            'save_gifs': kwargs.get('save_gifs', False),
            'model_name': kwargs.get('model_name', "deformed_model"),
            'opt_ftol': kwargs.get('opt_ftol', 1e-1), # Rotation opt tolerance
            'opt_xtol': kwargs.get('opt_xtol', 1e-1),
            'opt_gtol': kwargs.get('opt_gtol', 1e-1),
            'def_ftol': kwargs.get('def_ftol', 1e-2), # Deformation opt tolerance
            'def_xtol': kwargs.get('def_xtol', 1e-2),
            'def_gtol': kwargs.get('def_gtol', 1e-2),
        }
        return params

    def _log(self, message, level=logging.INFO):
        if self.verbose:
            log.log(level, message)

    def _initialize_gif(self, filename_suffix):
        """Initializes the plotter for GIF saving."""
        if not self.params['save_gifs']:
            return False
        try:
            # Ensure XVFB is running if needed (might be handled externally)
            # pv.start_xvfb()
            self._plotter = pv.Plotter(off_screen=True)
            gif_path = f"{self.params['model_name']}_{filename_suffix}.gif"
            self._plotter.open_gif(gif_path)
            self._log(f"Saving GIF to {gif_path}")
            self._gif_frame_count = 0
            return True
        except Exception as e:
            self._log(f"Could not initialize plotter for GIF: {e}", logging.WARNING)
            self.params['save_gifs'] = False # Disable if setup fails
            return False

    def _save_gif_frame(self, frame_data, frame_type):
        """Saves a single frame to the open GIF."""
        if not self.params['save_gifs'] or self._plotter is None:
            return

        self._gif_frame_count += 1
        # Optional: Skip frames to speed up saving
        if frame_type == 'deformation' and self._gif_frame_count % 10 != 0:
             return

        self._plotter.clear_actors()
        if frame_type == 'rotation':
            rotation_field_rad = frame_data
            # Create temporary mesh for visualization
            unique_verts_rot = self._calculate_unique_vertices_rotated(self.undeformed_tet, rotation_field_rad)
            num_cells = self.undeformed_tet.number_of_cells
            unique_cells = np.hstack((np.full((num_cells, 1), 4, dtype=int),
                                      np.arange(num_cells * 4).reshape(-1, 4)))
            temp_tet = pv.UnstructuredGrid(unique_cells.flatten(),
                                           np.full(num_cells, pv.CellType.TETRA),
                                           unique_verts_rot.reshape(-1, 3))
            temp_tet.cell_data["rotation (deg)"] = np.rad2deg(rotation_field_rad)
            clim_deg = np.rad2deg([self.params['max_neg_rotation_rad'], self.params['max_pos_rotation_rad']])
            clim_deg = [max(-90, clim_deg[0]), min(90, clim_deg[1])] # Sensible limits
            self._plotter.add_mesh(temp_tet, scalars="rotation (deg)", clim=clim_deg, lighting=False, cmap="coolwarm")

        elif frame_type == 'deformation':
            current_vertices = frame_data
            temp_tet = pv.UnstructuredGrid(self.undeformed_tet.cells, self.undeformed_tet.cell_type, current_vertices)
            self._plotter.add_mesh(temp_tet, color='lightblue', opacity=0.8)

        self._plotter.camera_position = 'iso'
        self._plotter.write_frame()

    def _close_gif(self):
        """Closes the GIF file."""
        if self._plotter:
            try:
                self._plotter.close()
                self._log("GIF saving complete.")
            except Exception as e:
                self._log(f"Error closing GIF plotter: {e}", logging.WARNING)
            self._plotter = None
        
    # --- Core Calculation Steps as Methods ---

    def _tetrahedralize(self):
        self._log("1. Tetrahedralizing...")
        try:
            # Ensure input mesh has faces defined correctly
            if isinstance(self.input_mesh.faces, np.ndarray) and self.input_mesh.faces.ndim == 1:
                 faces_array = self.input_mesh.faces.reshape(-1, 4)[:, 1:]
            else:
                 # Attempt to handle list or other formats if necessary, or raise error
                 raise ValueError("Input mesh faces format not recognized or incompatible.")

            input_tet_gen = tetgen.TetGen(self.input_mesh.points, faces_array)
            # input_tet_gen.make_manifold() # Optional, can fail
            input_tet_gen.tetrahedralize(order=1, mindihedral=10, minratio=1.5) # Add quality options
            self.tet = input_tet_gen.grid
            if self.tet is None or self.tet.number_of_cells == 0:
                raise ValueError("Tetrahedralization resulted in 0 cells or failure.")
            self._log(f"   Created {self.tet.number_of_cells} tetrahedra.")
            return True
        except Exception as e:
            self._log(f"Error during tetrahedralization: {e}", logging.ERROR)
            return False

    def _center_mesh(self):
        self._log("2. Centering tetrahedral mesh...")
        try:
            x_min, x_max, y_min, y_max, z_min, z_max = self.tet.bounds
            center_offset = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min])
            self.tet.points -= center_offset
            self.tet.points += self.params['part_offset']
            self.undeformed_tet = self.tet.copy() # Keep copy *after* centering
            self._log(f"   Mesh centered. Original Z min: {z_min:.2f}")
            return True
        except Exception as e:
            self._log(f"Error centering mesh: {e}", logging.ERROR)
            return False

    def _find_neighbours(self):
        self._log("3. Finding cell neighbours...")
        try:
            self.neighbour_dict = {nt: {i: [] for i in range(self.tet.number_of_cells)} for nt in ["point", "edge", "face"]}
            for ntype in ["point", "edge", "face"]:
                for cell_idx in range(self.tet.number_of_cells):
                    # PyVista uses 'points', 'edges', 'faces'
                    neighbours = self.tet.cell_neighbors(cell_idx, f"{ntype}s")
                    valid_neighbours = [n for n in neighbours if n != -1 and n < self.tet.number_of_cells]
                    self.neighbour_dict[ntype][cell_idx].extend(valid_neighbours)
            self._log("   Neighbours identified.")
            return True
        except Exception as e:
            self._log(f"Error finding neighbours: {e}", logging.ERROR)
            return False

    def _calculate_attributes(self):
        self._log("4. Calculating initial tetrahedral attributes...")
        try:
            # Calculate base attributes (face mapping, initial bottom)
            self.tet, _, self.bottom_cells = self._calculate_tet_attributes_internal(self.tet)
            self._log(f"   Found {len(self.bottom_cells)} initial bottom cells.")

            # Build connectivity graph (using 'point' neighbours for consistency)
            self._log("   Building cell neighbour graph...")
            self.cell_neighbour_graph = nx.Graph()
            cell_centers = self.tet.cell_centers().points
            processed_pairs = set()
            for cell_idx, neighbours in self.neighbour_dict["point"].items():
                for neighbour_idx in neighbours:
                    pair = tuple(sorted((cell_idx, neighbour_idx)))
                    if pair not in processed_pairs:
                        distance = np.linalg.norm(cell_centers[cell_idx] - cell_centers[neighbour_idx])
                        self.cell_neighbour_graph.add_edge(cell_idx, neighbour_idx, weight=distance)
                        processed_pairs.add(pair)
            self._log("   Cell neighbour graph built.")

            # Calculate derived attributes (normals, angles, 'in_air')
            self._log("   Updating full tet attributes (normals, centers, angles)...")
            self.tet = self._update_tet_attributes_internal(self.tet, self.cell_neighbour_graph, self.bottom_cells)

            # Update the undeformed copy too
            self.undeformed_tet, _, _ = self._calculate_tet_attributes_internal(self.undeformed_tet)
            self.undeformed_tet = self._update_tet_attributes_internal(self.undeformed_tet, self.cell_neighbour_graph, self.bottom_cells)
            self._log("   Full attributes calculated.")
            return True

        except Exception as e:
            self._log(f"Error calculating tet attributes: {e}", logging.ERROR)
            return False

    def _optimize_rotations(self):
        self._log("5. Optimizing rotation field...")
        gif_initialized = self._initialize_gif("optimize_rotations")
        start_time = time.time()

        try:
            # --- Calculate Initial Target Rotation ---
            initial_rotation_field = self._calculate_initial_rotation_field()
            initial_rotation_target = np.nan_to_num(initial_rotation_field, nan=0.0)

            # --- Setup Optimization ---
            cell_face_neighbours = []
            processed_pairs = set()
            for cell_idx, neighbours in self.neighbour_dict["face"].items():
                for neighbour_idx in neighbours:
                    pair = tuple(sorted((cell_idx, neighbour_idx)))
                    if pair not in processed_pairs:
                        cell_face_neighbours.append(list(pair))
                        processed_pairs.add(pair)
            cell_face_neighbours = np.array(cell_face_neighbours, dtype=int)
            num_neighbour_pairs = len(cell_face_neighbours)
            num_cells = self.undeformed_tet.number_of_cells
            num_residuals = num_neighbour_pairs + num_cells

            # --- Nested Objective/Jacobian/Sparsity ---
            def objective(current_rotation_field_rad):
                self._save_gif_frame(current_rotation_field_rad, 'rotation')
                if num_neighbour_pairs > 0:
                    diffs = current_rotation_field_rad[cell_face_neighbours[:, 0]] - current_rotation_field_rad[cell_face_neighbours[:, 1]]
                    neighbour_residuals = np.sqrt(self.params['neighbour_loss_weight']) * diffs
                else:
                    neighbour_residuals = np.array([])
                initial_residuals = current_rotation_field_rad - initial_rotation_target
                return np.concatenate((neighbour_residuals, initial_residuals))

            def jacobian(current_rotation_field_rad):
                jac = lil_matrix((num_residuals, num_cells), dtype=np.float32)
                sqrt_weight = np.sqrt(self.params['neighbour_loss_weight'])
                if num_neighbour_pairs > 0:
                    rows = np.arange(num_neighbour_pairs)
                    jac[rows, cell_face_neighbours[:, 0]] = sqrt_weight
                    jac[rows, cell_face_neighbours[:, 1]] = -sqrt_weight
                rows_initial = np.arange(num_neighbour_pairs, num_residuals)
                cols_initial = np.arange(num_cells)
                jac[rows_initial, cols_initial] = 1.0
                return jac.tocsr()

            def sparsity():
                s = lil_matrix((num_residuals, num_cells), dtype=np.int8)
                if num_neighbour_pairs > 0:
                    rows = np.arange(num_neighbour_pairs)
                    s[rows, cell_face_neighbours[:, 0]] = 1
                    s[rows, cell_face_neighbours[:, 1]] = 1
                rows_initial = np.arange(num_neighbour_pairs, num_residuals)
                cols_initial = np.arange(num_cells)
                s[rows_initial, cols_initial] = 1
                return s.tocsr()

            # --- Run Optimization ---
            initial_guess = np.zeros(num_cells)
            result = least_squares(
                objective, initial_guess, jac=jacobian, jac_sparsity=sparsity(),
                max_nfev=self.params['optimization_iterations'], verbose=0, # Use logger instead
                method='trf',
                ftol=self.params['opt_ftol'], xtol=self.params['opt_xtol'], gtol=self.params['opt_gtol']
            )
            self._log(f"   Optimization status: {result.status}, message: {result.message}")
            if not result.success:
                 self._log("   Rotation optimization failed to converge.", logging.WARNING)
                 # Decide whether to proceed with potentially bad rotations or fail

            self.optimized_rotation_field_rad = np.clip(result.x,
                self.params['max_neg_rotation_rad'], self.params['max_pos_rotation_rad'])

            self._log(f"   Optimized rotation field calculated. Range: {np.rad2deg(np.nanmin(self.optimized_rotation_field_rad)):.2f} to {np.rad2deg(np.nanmax(self.optimized_rotation_field_rad)):.2f} deg")
            self.undeformed_tet.cell_data["optimized_rotation_field_rad"] = self.optimized_rotation_field_rad
            self._log(f"   Rotation optimization finished in {time.time() - start_time:.2f} seconds.")
            return True

        except Exception as e:
            self._log(f"Error during rotation field optimization: {e}", logging.ERROR)
            return False
        finally:
            if gif_initialized: self._close_gif()

    def _calculate_deformation(self):
        self._log("6. Calculating mesh deformation...")
        if self.optimized_rotation_field_rad is None:
            self._log("   Skipping deformation: Optimized rotation field not available.", logging.WARNING)
            return False

        gif_initialized = self._initialize_gif("deformation")
        start_time = time.time()

        try:
            # --- Setup ---
            initial_vertices = self.undeformed_tet.points.copy()
            params0 = initial_vertices.flatten()
            num_points = self.undeformed_tet.number_of_points
            num_cells = self.undeformed_tet.number_of_cells
            cells = self.undeformed_tet.field_data["cells"] # Get from undeformed

            # Precompute target shapes
            target_rotation_matrices = self._calculate_rotation_matrices(self.undeformed_tet, self.optimized_rotation_field_rad)
            original_cell_vertices = initial_vertices[cells]
            original_cell_centers = np.mean(original_cell_vertices, axis=1)
            centered_original_vertices = original_cell_vertices - original_cell_centers[:, None, :]
            target_rotated_centered_vertices = np.einsum('cij,cjk->cik', target_rotation_matrices, centered_original_vertices.transpose(0, 2, 1)).transpose(0, 2, 1)

            # --- Nested Objective/Jacobian/Sparsity ---
            def objective(current_params):
                current_vertices = current_params.reshape(num_points, 3)
                self._save_gif_frame(current_vertices, 'deformation')
                current_cell_vertices = current_vertices[cells]
                current_cell_centers = np.mean(current_cell_vertices, axis=1)
                centered_current_vertices = current_cell_vertices - current_cell_centers[:, None, :]
                shape_difference = centered_current_vertices - target_rotated_centered_vertices
                return shape_difference.flatten()

            # --- Optimized Jacobian Calculation ---
            # Precompute indices for faster Jacobian assembly
            rows_jac = np.arange(num_cells * 12) # c*12 + v_local*3 + d_dim
            cells_flat = np.repeat(cells, 3, axis=0) # Repeat each cell 3 times for dimensions
            v_local_indices = np.tile(np.arange(4), num_cells * 3) # 0,1,2,3, 0,1,2,3,...
            d_dim_indices = np.repeat(np.arange(3), num_cells * 4) # 0,0,0,0, 1,1,1,1, 2,2,2,2,...

            # Map (c, v_local, d_dim) to row index
            row_map = np.ravel_multi_index((np.repeat(np.arange(num_cells), 12), # Cell index
                                            np.tile(np.repeat(np.arange(4), 3), num_cells), # v_local
                                            np.tile(np.arange(3), num_cells * 4)), # d_dim
                                           (num_cells, 4, 3))

            # Precompute column indices and values for the sparse matrix
            # Each residual row_idx depends on 4 points (p_prime) in the same dimension (k_dim=d_dim)
            num_entries = num_cells * 4 * 3 * 4 # num_residuals * 4 points influencing each
            jac_rows = np.zeros(num_entries, dtype=np.int32)
            jac_cols = np.zeros(num_entries, dtype=np.int32)
            jac_data = np.zeros(num_entries, dtype=np.float32)
            entry_idx = 0
            for c in range(num_cells):
                cell_global_indices = cells[c]
                for v_local in range(4):
                    for d_dim in range(3):
                        row_idx = c * 12 + v_local * 3 + d_dim
                        for v_prime_local in range(4):
                            p_prime = cell_global_indices[v_prime_local]
                            col_idx = p_prime * 3 + d_dim
                            delta_vv_prime = 1.0 if v_local == v_prime_local else 0.0
                            jac_value = delta_vv_prime - 0.25

                            jac_rows[entry_idx] = row_idx
                            jac_cols[entry_idx] = col_idx
                            jac_data[entry_idx] = jac_value
                            entry_idx += 1

            # Create the final sparse matrix structure once
            num_residuals = num_cells * 12
            num_vars = num_points * 3
            jacobian_structure = csr_matrix((jac_data, (jac_rows, jac_cols)),
                                            shape=(num_residuals, num_vars))

            def jacobian(current_params):
                # The Jacobian structure is constant for this problem
                return jacobian_structure

            def sparsity():
                 # Return the precomputed structure, converted to int8 for sparsity pattern
                 s = jacobian_structure.copy()
                 s.data[:] = 1
                 return s.astype(np.int8)
            # --- End Optimized Jacobian ---

            # --- Run Optimization ---
            result = least_squares(
                objective, params0, jac=jacobian, jac_sparsity=sparsity(),
                max_nfev=self.params['deformation_iterations'], verbose=0, # Use logger
                method='trf',
                ftol=self.params['def_ftol'], xtol=self.params['def_xtol'], gtol=self.params['def_gtol']
            )
            self._log(f"   Optimization status: {result.status}, message: {result.message}")
            if not result.success:
                 self._log("   Deformation optimization failed to converge.", logging.WARNING)
                 # Fallback: use initial vertices? Or fail? Let's store result anyway.

            self.new_vertices = result.x.reshape(num_points, 3)
            self._log(f"   Deformed vertex positions calculated.")
            self._log(f"   Deformation optimization finished in {time.time() - start_time:.2f} seconds.")
            return True

        except Exception as e:
            self._log(f"Error during deformation calculation: {e}", logging.ERROR)
            return False
        finally:
            if gif_initialized: self._close_gif()

    def _create_final_mesh(self):
        self._log("7. Finalizing deformed mesh...")
        if self.new_vertices is None:
            self._log("   Cannot finalize: No deformed vertices available.", logging.ERROR)
            return False
        try:
            # Use undeformed tet structure with new vertices
            deformed_tet = pv.UnstructuredGrid(self.undeformed_tet.cells,
                                               self.undeformed_tet.celltypes,
                                               self.new_vertices)

            # Copy relevant field data
            for key, value in self.undeformed_tet.field_data.items():
                if key not in ['cells', 'cell_vertices', 'faces', 'face_vertices']:
                    try: deformed_tet.field_data[key] = value
                    except Exception: pass # Ignore if copy fails

            # Extract surface and clean (removes unused points/cells)
            self.deformed_surface = deformed_tet.extract_surface().clean()
            self._log("   Final deformed surface created.")
            return True
        except Exception as e:
            self._log(f"Error during final mesh creation: {e}", logging.ERROR)
            return False

    # --- Internal Helper Methods (Moved from global scope or adapted) ---
    def _calculate_tet_attributes_internal(self, tet_mesh):
        """Internal version of calculate_tet_attributes."""
        surface_mesh = tet_mesh.extract_surface()
        cells = tet_mesh.cells.reshape(-1, 5)[:, 1:]
        faces = surface_mesh.faces.reshape(-1, 4)[:, 1:]
        face_vertices = surface_mesh.points

        # Store basic geometry if not already present (e.g., for undeformed_tet)
        if "cells" not in tet_mesh.field_data: tet_mesh.add_field_data(cells, "cells")
        if "faces" not in tet_mesh.field_data: tet_mesh.add_field_data(faces, "faces")
        if "face_vertices" not in tet_mesh.field_data: tet_mesh.add_field_data(face_vertices, "face_vertices")

        cell_to_face = {}
        face_to_cell = {i: [] for i in range(surface_mesh.n_cells)}
        cell_to_face_vertices = {}

        # KDTree for mapping tet vertices to surface vertices
        if len(face_vertices) > 0:
            kdtree = KDTree(face_vertices)
            for cell_v_idx, cell_v in enumerate(tet_mesh.points):
                dist, idx = kdtree.query(cell_v)
                if dist < 1e-5:
                    cell_to_face_vertices[cell_v_idx] = idx
        else:
             self._log("   Warning: Surface mesh has no vertices, cannot map faces.", logging.WARNING)

        # Find surface faces belonging to each tet cell
        for cell_idx, cell in enumerate(cells):
            surface_verts_in_cell = [cell_to_face_vertices[v_idx] for v_idx in cell if v_idx in cell_to_face_vertices]
            if len(surface_verts_in_cell) >= 3:
                candidate_faces = []
                # This part can be slow, consider optimization if needed
                for face_idx, face_verts in enumerate(faces):
                    if all(fv in surface_verts_in_cell for fv in face_verts):
                        candidate_faces.append(face_idx)
                if candidate_faces:
                    cell_to_face[cell_idx] = candidate_faces
                    for face_idx in candidate_faces:
                        if cell_idx not in face_to_cell[face_idx]:
                            face_to_cell[face_idx].append(cell_idx)

        tet_mesh.add_field_data(encode_object(cell_to_face), "cell_to_face")
        # tet_mesh.add_field_data(encode_object(face_to_cell), "face_to_cell") # Less frequently used
        # Calculate has_face attribute
        has_face_arr = np.zeros(tet_mesh.number_of_cells, dtype=int)
        for cell_idx, face_indices in cell_to_face.items():
            if face_indices: has_face_arr[cell_idx] = 1
        tet_mesh.cell_data['has_face'] = has_face_arr

        # --- Initial Bottom Cell Calculation ---
        temp_face_centers = np.full((tet_mesh.number_of_cells, 3), np.nan)
        if surface_mesh.n_cells > 0:
            surface_mesh_cell_centers = surface_mesh.cell_centers().points
            surface_mesh_face_normals = surface_mesh.face_normals
            for cell_idx, face_indices in cell_to_face.items():
                if not face_indices: continue
                face_centers = surface_mesh_cell_centers[face_indices]
                face_normals = surface_mesh_face_normals[face_indices]
                if len(face_normals) > 0:
                    most_down_idx = np.argmin(face_normals[:, 2])
                    temp_face_centers[cell_idx] = face_centers[most_down_idx]

        min_z = np.nanmin(temp_face_centers[:, 2]) if not np.all(np.isnan(temp_face_centers[:, 2])) else 0
        bottom_thresh = min_z + 0.3 # mm tolerance
        bottom_mask = temp_face_centers[:, 2] < bottom_thresh
        bottom_cells_indices = np.where(bottom_mask)[0]
        tet_mesh.cell_data['is_bottom'] = bottom_mask

        return tet_mesh, bottom_mask, bottom_cells_indices

    def _update_tet_attributes_internal(self, tet_mesh, graph, bottom_cells_indices):
        """Internal version of update_tet_attributes."""
        try: cell_to_face = decode_object(tet_mesh.field_data["cell_to_face"])
        except KeyError: return tet_mesh # Cannot proceed

        num_cells = tet_mesh.number_of_cells
        surface_mesh = tet_mesh.extract_surface()

        # Ensure basic geometry is available
        if "cells" not in tet_mesh.field_data: tet_mesh.add_field_data(tet_mesh.cells.reshape(-1, 5)[:, 1:], "cells")
        if "cell_center" not in tet_mesh.cell_data: tet_mesh.cell_data["cell_center"] = tet_mesh.cell_centers().points

        face_normals_out = np.full((num_cells, 3), np.nan)
        face_centers_out = np.full((num_cells, 3), np.nan)

        if surface_mesh.n_cells > 0:
            surf_normals = surface_mesh.face_normals
            surf_centers = surface_mesh.cell_centers().points
            for cell_idx, face_indices in cell_to_face.items():
                if not face_indices: continue
                cell_surf_normals = surf_normals[face_indices]
                if len(cell_surf_normals) > 0:
                    most_down_idx = np.argmin(cell_surf_normals[:, 2])
                    face_normals_out[cell_idx] = cell_surf_normals[most_down_idx]
                    face_centers_out[cell_idx] = surf_centers[face_indices[most_down_idx]]

        # Normalize valid normals
        valid_mask = ~np.isnan(face_normals_out).any(axis=1)
        norms = np.linalg.norm(face_normals_out[valid_mask], axis=1, keepdims=True)
        face_normals_out[valid_mask] /= np.where(norms == 0, 1, norms) # Avoid div by zero
        tet_mesh.cell_data['face_normal'] = face_normals_out
        tet_mesh.cell_data['face_center'] = face_centers_out

        # Recompute bottom mask based on potentially updated geometry
        bottom_thresh = np.nanmin(face_centers_out[:, 2]) + 0.3 if not np.all(np.isnan(face_centers_out[:, 2])) else 0
        bottom_mask = face_centers_out[:, 2] < bottom_thresh
        tet_mesh.cell_data['is_bottom'] = bottom_mask
        current_bottom_cells = np.where(bottom_mask)[0] # Use current geometry

        # Overhang Angle & Direction
        face_normals_calc = face_normals_out.copy()
        face_normals_calc[bottom_mask] = np.nan # Exclude bottom faces
        dot_prod = np.clip(np.einsum('ij,j->i', face_normals_calc, UP_VECTOR), -1.0, 1.0)
        overhang_angle = np.arccos(dot_prod)
        tet_mesh.cell_data['overhang_angle'] = overhang_angle

        overhang_dir = face_normals_calc[:, :2].copy()
        norms_dir = np.linalg.norm(overhang_dir, axis=1, keepdims=True)
        overhang_dir /= np.where(norms_dir == 0, 1, norms_dir)
        tet_mesh.cell_data['overhang_direction'] = overhang_dir

        # 'In Air' Calculation
        tet_mesh.cell_data['in_air'] = np.full(num_cells, False)
        if len(current_bottom_cells) > 0 and graph.number_of_nodes() > 0:
            try:
                _, paths = nx.multi_source_dijkstra(graph, set(current_bottom_cells))
                cell_centers_z = tet_mesh.cell_data['cell_center'][:, 2]
                for cell_idx, path in paths.items():
                    if len(path) > 1:
                        heights_on_path = cell_centers_z[path]
                        if np.any(heights_on_path > cell_centers_z[cell_idx] + 1.0): # IN_AIR_THRESHOLD = 1.0
                            tet_mesh.cell_data['in_air'][cell_idx] = True
            except nx.NetworkXNoPath: pass # Ignore cells with no path
            except Exception as e: self._log(f"Warning: Dijkstra failed during 'in_air' calc: {e}", logging.WARNING)
        else:
             self._log("Warning: No bottom cells or graph for 'in_air' calc.", logging.WARNING)

        return tet_mesh

    def _calculate_path_length_gradient(self):
        """Internal: Calculate path length to base gradient."""
        num_cells = self.undeformed_tet.number_of_cells
        gradient = np.zeros(num_cells)
        distances = np.full(num_cells, np.nan)
        closest_bottom_indices = np.full(num_cells, -1, dtype=int)

        if len(self.bottom_cells) == 0 or self.cell_neighbour_graph.number_of_nodes() == 0:
            self._log("Warning: No bottom cells/graph for path length gradient.", logging.WARNING)
            self.undeformed_tet.cell_data["cell_distance_to_bottom"] = distances
            self.undeformed_tet.cell_data["path_length_to_base_gradient"] = gradient
            return gradient

        try:
            dist_map, path_map = nx.multi_source_dijkstra(self.cell_neighbour_graph, set(self.bottom_cells))
            overhang_angle = self.undeformed_tet.cell_data['overhang_angle']
            overhang_thresh = np.pi/2.0 + self.params['max_overhang_rad']
            is_overhang = ~np.isnan(overhang_angle) & (overhang_angle > overhang_thresh)

            for cell_idx in range(num_cells):
                if is_overhang[cell_idx] and cell_idx not in self.bottom_cells and cell_idx in dist_map:
                    closest_bottom_indices[cell_idx] = path_map[cell_idx][-1]
                    distances[cell_idx] = dist_map[cell_idx]
        except Exception as e:
            self._log(f"Error during Dijkstra path calculation: {e}", logging.WARNING)

        self.undeformed_tet.cell_data["cell_distance_to_bottom"] = distances
        cell_centers = self.undeformed_tet.cell_data["cell_center"]

        # Calculate gradient using plane fitting or fallback
        for cell_idx in range(num_cells):
            if not np.isnan(distances[cell_idx]):
                neighbours = self.neighbour_dict["edge"].get(cell_idx, [])
                local_cells = np.unique(np.hstack((neighbours, cell_idx))) # Ensure unique indices
                local_distances = distances[local_cells]
                valid_mask = ~np.isnan(local_distances)
                local_cells = local_cells[valid_mask]
                local_distances = local_distances[valid_mask]
                current_center_xy = cell_centers[cell_idx, :2]
                norm_cc_xy = np.linalg.norm(current_center_xy)

                if len(local_cells) < 3: # Fallback: Roll towards closest bottom
                    closest_idx = closest_bottom_indices[cell_idx]
                    grad_val = 0
                    if closest_idx != -1:
                        dir_to_bottom = cell_centers[closest_idx, :2] - current_center_xy
                        norm_dir = np.linalg.norm(dir_to_bottom)
                        if norm_dir > 1e-6 and norm_cc_xy > 1e-6:
                            dir_to_bottom /= norm_dir
                            center_dir = current_center_xy / norm_cc_xy
                            dot_prod = np.dot(center_dir, dir_to_bottom)
                            grad_val = np.sign(dot_prod) if not np.isnan(dot_prod) else 0
                    gradient[cell_idx] = grad_val
                else: # Plane fitting
                    points_fit = np.vstack((cell_centers[local_cells, :2].T, local_distances))
                    try:
                        _, normal = planeFit(points_fit)
                        grad_xy = -normal[:2]
                        grad_val = 0
                        if norm_cc_xy > 1e-6:
                            center_dir = current_center_xy / norm_cc_xy
                            grad_val = np.dot(center_dir, grad_xy)
                        gradient[cell_idx] = grad_val if not np.isnan(grad_val) else 0
                    except Exception: gradient[cell_idx] = 0 # Plane fit failed

        # Smoothing
        if self.params['initial_rotation_field_smoothing'] > 0:
            smoothed = gradient.copy()
            mask_initial = (gradient != 0) & (~np.isnan(gradient))
            for _ in range(self.params['initial_rotation_field_smoothing']):
                new_smoothed = smoothed.copy()
                for cell_idx in np.where(mask_initial)[0]:
                    neighbours = self.neighbour_dict["point"].get(cell_idx, [])
                    local_cells = np.array(neighbours, dtype=int)
                    valid_local_mask = mask_initial[local_cells] # Use initial mask
                    local_cells = local_cells[valid_local_mask]
                    if len(local_cells) > 0:
                        new_smoothed[cell_idx] = np.mean(smoothed[local_cells]) # Average previous smooth step
                smoothed = new_smoothed
            gradient = smoothed

        if not self.params['set_initial_rotation_to_zero']:
            gradient[np.isnan(distances)] = np.nan
        else:
            gradient = np.nan_to_num(gradient, nan=0.0)

        self.undeformed_tet.cell_data["path_length_to_base_gradient"] = gradient
        return gradient

    def _calculate_initial_rotation_field(self):
        """Internal: Calculate initial target rotation field."""
        overhang_angle = self.undeformed_tet.cell_data['overhang_angle']
        target_angle = np.pi/2.0 + self.params['max_overhang_rad']
        initial_field = overhang_angle - target_angle
        initial_field[overhang_angle <= target_angle] = np.nan
        initial_field[np.isnan(overhang_angle)] = np.nan

        path_gradient = self._calculate_path_length_gradient()
        gradient_sign = np.sign(path_gradient)
        initial_field *= gradient_sign # Apply direction

        # Steep overhang compensation
        if self.params['steep_overhang_compensation'] and 'in_air' in self.undeformed_tet.cell_data:
            in_air = self.undeformed_tet.cell_data['in_air']
            compensation = 2.0 * (overhang_angle[in_air] - np.pi)
            compensation[compensation < 0] = 0
            # Ensure gradient_sign is broadcastable and handle potential NaNs
            comp_sign = np.nan_to_num(gradient_sign[in_air], nan=1.0) # Default sign if gradient is NaN?
            initial_field[in_air] += compensation * comp_sign

        # Apply multiplier and clip
        valid_mask = ~np.isnan(initial_field)
        initial_field[valid_mask] *= self.params['rotation_multiplier']
        initial_field = np.clip(initial_field,
                                self.params['max_neg_rotation_rad'],
                                self.params['max_pos_rotation_rad'])
        self.undeformed_tet.cell_data["initial_rotation_field"] = initial_field
        return initial_field

    def _calculate_rotation_matrices(self, tet_mesh, rotation_field_rad):
        """Internal: Calculate rotation matrices for a given tet mesh."""
        num_cells = tet_mesh.number_of_cells
        cell_centers_xy = tet_mesh.cell_data["cell_center"][:, :2]
        radial_vecs = cell_centers_xy.copy()
        norms = np.linalg.norm(radial_vecs, axis=1)
        valid = norms > 1e-9
        radial_vecs[valid] /= norms[valid, None]
        radial_vecs[~valid] = [1, 0] # Default for center

        tangential_vecs = np.zeros((num_cells, 3))
        tangential_vecs[:, 0] = -radial_vecs[:, 1]
        tangential_vecs[:, 1] = radial_vecs[:, 0]

        valid_rot = ~np.isnan(rotation_field_rad)
        rot_vecs = np.zeros((num_cells, 3))
        rot_vecs[valid_rot] = rotation_field_rad[valid_rot, None] * tangential_vecs[valid_rot]

        matrices = np.tile(np.eye(3), (num_cells, 1, 1))
        if np.any(valid_rot):
            valid_indices = np.where(valid_rot)[0]
            try:
                matrices[valid_indices] = R.from_rotvec(rot_vecs[valid_indices]).as_matrix()
            except Exception as e:
                self._log(f"Error creating rotation matrices: {e}", logging.WARNING)
        return matrices

    def _calculate_unique_vertices_rotated(self, tet_mesh, rotation_field_rad):
        """Internal: Calculate uniquely rotated vertices (for visualization)."""
        matrices = self._calculate_rotation_matrices(tet_mesh, rotation_field_rad)
        cells = tet_mesh.field_data["cells"]
        vertices = tet_mesh.points
        centers = tet_mesh.cell_data["cell_center"]
        cell_coords = vertices[cells]
        centered = cell_coords - centers[:, None, :]
        rotated = np.einsum('cij,cjk->cik', matrices, centered.transpose(0, 2, 1)).transpose(0, 2, 1)
        unique_verts = rotated + centers[:, None, :]
        return unique_verts

    # --- Public Execution Method ---
    def run(self):
        """Executes the full deformation pipeline."""
        self._log("--- Starting Tetrahedral Deformation ---")
        start_total_time = time.time()
        self.success = False # Reset success flag

        steps = [
            self._tetrahedralize,
            self._center_mesh,
            self._find_neighbours,
            self._calculate_attributes,
            self._optimize_rotations,
            self._calculate_deformation,
            self._create_final_mesh
        ]

        for step_func in steps:
            if not step_func(): # Execute step and check success
                self._log(f"Pipeline stopped due to error in step: {step_func.__name__}", logging.ERROR)
                # Clean up plotter if it was initialized
                if self._plotter: self._close_gif()
                return False # Stop pipeline

        self.success = True
        self._log(f"--- Tetrahedral Deformation Complete ({time.time() - start_total_time:.2f}s) ---")
        return True

    def get_deformed_mesh(self):
        """Returns the final deformed surface mesh."""
        if self.success and self.deformed_surface:
            return self.deformed_surface
        else:
            # Return original cleaned mesh as fallback if deformation failed
            self._log("Deformation failed or mesh not created, returning original cleaned mesh.", logging.WARNING)
            try:
                return self.input_mesh.clean()
            except: # If even cleaning fails
                return self.input_mesh

    def get_vertex_indices(self):
        """Returns the indices of the vertices in the deformed mesh."""
        if self.success and self.deformed_surface:
            return set(range(self.deformed_surface.n_points))
        else:
            return set() # Return empty set on failure

# --- Main Function Wrapper ---
def deform_mesh(mesh: pv.PolyData, **kwargs):
    """
    Deforms the mesh to mitigate overhangs using the MeshDeformer class.

    Args:
        mesh: Input surface mesh (pyvista.PolyData).
        **kwargs: Keyword arguments for MeshDeformer parameters (see class).
            Example: max_overhang_deg=30, optimization_iterations=20, etc.

    Returns:
        pv.PolyData: The deformed surface mesh (or original on failure).
        set: Indices of vertices in the deformed mesh (empty on failure).
    """
    start_time = time.time()
    deformer = MeshDeformer(mesh, **kwargs)
    success = deformer.run()
    end_time = time.time()
    log.info(f"deform_mesh execution time: {end_time - start_time:.2f} seconds")

    deformed_mesh_result = deformer.get_deformed_mesh()
    vertex_indices = deformer.get_vertex_indices()

    return deformed_mesh_result, vertex_indices

