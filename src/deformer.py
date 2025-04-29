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
import sys
import os
from pathlib import Path

# Add the find_neighbors directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "find_neighbors"))

# Try to import the Cython module
try:
    import deformer_cython
    CYTHON_AVAILABLE = True
    import logging
    logging.info("Cython optimization module loaded successfully!")
except ImportError as e:
    CYTHON_AVAILABLE = False
    import logging
    logging.warning(f"Cython optimization not available: {e}")

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


    #these are parameter for the deformation step and all underlying steps that are part of it
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

    # --- Core Calculation Steps as Methods ---


    #first step we make the input mesh into a volume (we might need to make it watertight)
    def _tetrahedralize(self):
        self._log("1. Tetrahedralizing...")
        try:
            # Ensure input mesh has faces defined correctly
            if isinstance(self.input_mesh.faces, np.ndarray) and self.input_mesh.faces.ndim == 1:
                 #get faces from the input mesh
                 faces_array = self.input_mesh.faces.reshape(-1, 4)[:, 1:]
            else:
                 # Attempt to handle list or other formats if necessary, or raise error
                 raise ValueError("Input mesh faces format not recognized or incompatible.")

            #call tetgen and make the input mesh a volume (thetrahedra)
            input_tet_gen = tetgen.TetGen(self.input_mesh.points, faces_array)
            input_tet_gen.make_manifold() # Optional, can fail
            input_tet_gen.tetrahedralize(order=1, mindihedral=10, minratio=1.5) # Add quality options
            self.tet = input_tet_gen.grid #assign self.tet to the newly created volume
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
            # Extract necessary data
            cells = self.tet.field_data.get("cells", 
                self.tet.cells.reshape(-1, 5)[:, 1:])
            n_cells = self.tet.number_of_cells
            
            # Initialize neighbor dict with the CORRECT STRUCTURE
            neighbor_types = ["point", "edge", "face"]
            self.neighbour_dict = {}
            
            # IMPORTANT: Create as dict of lists, not list of lists
            for ntype in neighbor_types:
                self.neighbour_dict[ntype] = {}
            
            # Try to import Cython dynamically (to handle import errors better)
            cython_module = None
            try:
                # Add multiple possible import paths
                import sys
                from pathlib import Path
                current_dir = Path(__file__).parent
                sys.path.append(str(current_dir))
                sys.path.append(str(current_dir / "find_neighbors"))
                sys.path.append(str(current_dir / "find_neighbors" / "build" / "lib.macosx-15.2-arm64-cpython-310"))
                
                # Try multiple import strategies
                try:
                    import deformer_cython
                    cython_module = deformer_cython
                    self._log("   Successfully imported deformer_cython module")
                except ImportError:
                    try:
                        from find_neighbors import deformer_cython
                        cython_module = deformer_cython
                        self._log("   Successfully imported from find_neighbors package")
                    except ImportError:
                        self._log("   Cython module not found in standard locations")
            except Exception as e:
                self._log(f"   Import error: {e}", logging.WARNING)
            
            # Use Cython if available
            if cython_module is not None:
                self._log("   Using Cython-optimized neighbor calculation")
                try:
                    # Convert to int32
                    cells_int32 = np.array(cells, dtype=np.int32)
                    self._log(f"   Cell data type converted from {cells.dtype} to {cells_int32.dtype}")
                    
                    # Call the function with the correct name
                    if hasattr(cython_module, 'find_neighbors_cython'):
                        neighbor_result = cython_module.find_neighbors_cython(cells_int32, n_cells)
                        
                        # Convert result to correct dictionary format
                        for ntype in neighbor_types:
                            for i in range(n_cells):
                                self.neighbour_dict[ntype][i] = neighbor_result[ntype][i]
                        
                        self._log("   Cython neighbor calculation completed")
                    else:
                        available_funcs = [f for f in dir(cython_module) if not f.startswith('_')]
                        self._log(f"   Function 'find_neighbors_cython' not found in module. Available: {available_funcs}")
                        raise AttributeError("Function not found in module")
                        
                except Exception as e:
                    self._log(f"   Cython neighbor calculation failed: {e}", logging.WARNING)
                    # Fall through to Python implementation
            
            # Original Python implementation (always run if Cython failed or not available)
            if cython_module is None or any(len(self.neighbour_dict[nt]) == 0 for nt in neighbor_types):
                self._log("   Using Python neighbor calculation")
                
                for ntype in neighbor_types:
                    self.neighbour_dict[ntype] = {}
                    for cell_idx in range(n_cells):
                        neighbours = self.tet.cell_neighbors(cell_idx, f"{ntype}s")
                        self.neighbour_dict[ntype][cell_idx] = [n for n in neighbours if n != -1 and n < n_cells]
            
            # Verify neighbor dict has correct structure before returning
            for ntype in neighbor_types:
                if not isinstance(self.neighbour_dict.get(ntype, {}), dict):
                    self._log(f"   WARNING: neighbour_dict[{ntype}] is not a dictionary! Fixing...", logging.WARNING)
                    self.neighbour_dict[ntype] = {i: [] for i in range(n_cells)}
            
            self._log("   Neighbours identified.")
            return True
        except Exception as e:
            self._log(f"Error finding neighbours: {e}", logging.ERROR)
            return False

    def _calculate_attributes(self):
        self._log("4. Calculating initial tetrahedral attributes...")
        try:
            # Extract data needed for calculation
            points = self.tet.points  # Keep as float64
            cells = self.tet.field_data.get("cells", 
                self.tet.cells.reshape(-1, 5)[:, 1:])
            
            # Extract faces from surface mesh
            surface_mesh = self.tet.extract_surface()
            faces = surface_mesh.faces.reshape(-1, 4)[:, 1:]
            
            # Ensure cells and faces are int32
            cells_int32 = np.array(cells, dtype=np.int32)
            faces_int32 = np.array(faces, dtype=np.int32)
            
            # Debug info
            self._log(f"   Points dtype: {points.dtype}, shape: {points.shape}")
            self._log(f"   Cells dtype: {cells_int32.dtype}, shape: {cells_int32.shape}")
            self._log(f"   Faces dtype: {faces_int32.dtype}, shape: {faces_int32.shape}")
            
            # Call Cython function with original points (float64)
            self._log("   Using Cython-optimized attribute calculation")
            attr_result = deformer_cython.calculate_attributes_cython(
                points, cells_int32, faces_int32, self.neighbour_dict)
            
            # Extract results
            self.tet.add_field_data(encode_object(attr_result["cell_to_face"]), "cell_to_face")
            
            # Ensure bottom_cells are integers
            self.bottom_cells = [int(cell) for cell in attr_result["bottom_cells"]]
            self._log(f"   Found {len(self.bottom_cells)} initial bottom cells.")
            
            # Add cell centers to mesh
            self.tet.cell_data["cell_center"] = attr_result["cell_centers"]
            
            # Create sparse matrix from dense
            import scipy.sparse as sp
            adjacency = attr_result["adjacency_matrix"]
            self.cell_adjacency_matrix = sp.csr_matrix(adjacency)
            
            # Create graph from adjacency matrix
            import networkx as nx
            self.cell_neighbour_graph = nx.from_scipy_sparse_array(
                self.cell_adjacency_matrix, edge_attribute='weight')
            
            # Continue with attribute updates for both meshes
            self._log("   Updating attributes for both meshes...")
            self.tet = self._update_tet_attributes_internal(self.tet, self.cell_neighbour_graph, self.bottom_cells)
            self.undeformed_tet, _, _ = self._calculate_tet_attributes_internal(self.undeformed_tet)
            self.undeformed_tet = self._update_tet_attributes_internal(
                self.undeformed_tet, self.cell_neighbour_graph, self.bottom_cells)
                
            self._log("   Full attributes calculated.")
            return True
            
        except Exception as e:
            self._log(f"Error calculating tet attributes: {e}", logging.ERROR)
            import traceback
            self._log(traceback.format_exc(), logging.ERROR)
            return False

    def _optimize_rotations(self):
        self._log("5. Optimizing rotation field...")
        start_time = time.time()

        try:
            # --- Safety check and fix for neighbour_dict structure ---
            for ntype in ["point", "edge", "face"]:
                if not isinstance(self.neighbour_dict.get(ntype), dict):
                    self._log(f"   WARNING: Fixing neighbour_dict[{ntype}] structure (was not dict)...", logging.WARNING)
                    temp_dict = {}
                    # Attempt conversion assuming it might be list-like
                    try:
                        for i, neighbors in enumerate(self.neighbour_dict.get(ntype, [])):
                            temp_dict[i] = neighbors
                    except TypeError: # Handle cases where it's not iterable
                        self._log(f"   ERROR: neighbour_dict[{ntype}] is not iterable, cannot fix automatically.", logging.ERROR)
                        temp_dict = {i: [] for i in range(self.undeformed_tet.number_of_cells)} # Fallback
                    self.neighbour_dict[ntype] = temp_dict
            # --- End safety check ---

            # --- Calculate Initial Target Rotation ---
            initial_rotation_field = self._calculate_initial_rotation_field()
            initial_rotation_target = np.nan_to_num(initial_rotation_field, nan=0.0)

            # --- Setup Optimization: Build cell_face_neighbours carefully ---
            cell_face_neighbours_list = [] # Use a list first
            processed_pairs = set()
            face_dict = self.neighbour_dict.get("face", {}) # Use .get for safety

            self._log(f"   Building cell_face_neighbours from face_dict (size: {len(face_dict)})...")
            conversion_errors = 0
            for cell_idx, neighbours in face_dict.items():
                try:
                    cell_idx_int = int(cell_idx) # Convert key
                except (ValueError, TypeError):
                    self._log(f"   ERROR: Skipping invalid cell index key: {cell_idx} (type: {type(cell_idx)})", logging.ERROR)
                    conversion_errors += 1
                    continue

                if not isinstance(neighbours, (list, tuple, np.ndarray)):
                    self._log(f"   WARNING: Skipping invalid neighbours value for cell {cell_idx_int} (type: {type(neighbours)})", logging.WARNING)
                    continue

                for neighbour_val in neighbours:
                    try:
                        neighbour_idx_int = int(neighbour_val) # Convert value
                    except (ValueError, TypeError):
                        self._log(f"   ERROR: Skipping invalid neighbour index value for cell {cell_idx_int}: {neighbour_val} (type: {type(neighbour_val)})", logging.ERROR)
                        conversion_errors += 1
                        continue

                    # Check bounds immediately after conversion
                    if cell_idx_int < 0 or neighbour_idx_int < 0:
                        self._log(f"   ERROR: Skipping negative index pair: ({cell_idx_int}, {neighbour_idx_int})", logging.ERROR)
                        conversion_errors += 1
                        continue

                    pair = tuple(sorted((cell_idx_int, neighbour_idx_int)))
                    if pair not in processed_pairs:
                        cell_face_neighbours_list.append(list(pair))
                        processed_pairs.add(pair)

            if conversion_errors > 0:
                self._log(f"   {conversion_errors} errors occurred during neighbour index conversion.", logging.WARNING)

            # Convert list to numpy array *only if* it's not empty
            if not cell_face_neighbours_list:
                self._log("   WARNING: No valid cell face neighbour pairs found after conversion. Rotation optimization might be trivial.", logging.WARNING)
                cell_face_neighbours = np.empty((0, 2), dtype=np.int32) # Handle empty case
            else:
                # Perform final check for non-integer types before creating array
                all_ints = all(isinstance(p[0], int) and isinstance(p[1], int) for p in cell_face_neighbours_list)
                if not all_ints:
                    self._log("   ERROR: Non-integer values detected in cell_face_neighbours_list before final conversion!", logging.ERROR)
                    # Find and log problematic entries
                    for i, p in enumerate(cell_face_neighbours_list):
                        if not (isinstance(p[0], int) and isinstance(p[1], int)):
                            self._log(f"      Problematic pair at index {i}: {p} (types: {type(p[0])}, {type(p[1])})")
                    # Decide how to proceed: maybe filter them out or raise error
                    # Filtering approach:
                    cell_face_neighbours_list = [p for p in cell_face_neighbours_list if isinstance(p[0], int) and isinstance(p[1], int)]
                    if not cell_face_neighbours_list:
                        cell_face_neighbours = np.empty((0, 2), dtype=np.int32)
                    else:
                        cell_face_neighbours = np.array(cell_face_neighbours_list, dtype=np.int32)
                else:
                    cell_face_neighbours = np.array(cell_face_neighbours_list, dtype=np.int32)
            # --- End Setup Optimization ---


            num_neighbour_pairs = cell_face_neighbours.shape[0]
            num_cells = self.undeformed_tet.number_of_cells
            num_residuals = num_neighbour_pairs + num_cells

            # --- DETAILED DEBUG LOGGING (Post Conversion) ---
            self._log(f"   num_cells: {num_cells}")
            self._log(f"   num_neighbour_pairs: {num_neighbour_pairs}")
            self._log(f"   num_residuals: {num_residuals}")
            self._log(f"   Final cell_face_neighbours shape: {cell_face_neighbours.shape}, dtype: {cell_face_neighbours.dtype}")

            if num_neighbour_pairs > 0:
                self._log(f"   Sample cell_face_neighbours[0]: {cell_face_neighbours[0]}, type: {type(cell_face_neighbours[0,0])}")
                # Check bounds against num_cells
                max_idx_0 = np.max(cell_face_neighbours[:, 0])
                max_idx_1 = np.max(cell_face_neighbours[:, 1])
                min_idx_0 = np.min(cell_face_neighbours[:, 0])
                min_idx_1 = np.min(cell_face_neighbours[:, 1])
                self._log(f"   Index range in cell_face_neighbours[:, 0]: [{min_idx_0}, {max_idx_0}]")
                self._log(f"   Index range in cell_face_neighbours[:, 1]: [{min_idx_1}, {max_idx_1}]")

                if max_idx_0 >= num_cells or max_idx_1 >= num_cells or min_idx_0 < 0 or min_idx_1 < 0:
                    self._log(f"   CRITICAL ERROR: Index in cell_face_neighbours is out of bounds [0, {num_cells-1}]!", logging.ERROR)
                    # Find and log specific problematic indices
                    bad_indices = cell_face_neighbours[(cell_face_neighbours[:, 0] >= num_cells) |
                                                    (cell_face_neighbours[:, 1] >= num_cells) |
                                                    (cell_face_neighbours[:, 0] < 0) |
                                                    (cell_face_neighbours[:, 1] < 0)]
                    self._log(f"   Problematic pairs: {bad_indices}")
                    # Stop execution as this will definitely fail
                    raise IndexError(f"Invalid indices found in cell_face_neighbours. Max index allowed: {num_cells-1}")
            else:
                self._log("   No neighbour pairs to check bounds for.")

            self._log(f"   initial_rotation_target shape: {initial_rotation_target.shape}, dtype: {initial_rotation_target.dtype}")
            if len(initial_rotation_target) != num_cells:
                self._log(f"   ERROR: initial_rotation_target length ({len(initial_rotation_target)}) does not match num_cells ({num_cells})!", logging.ERROR)
                # Adjust or raise error
                initial_rotation_target = np.resize(initial_rotation_target, num_cells) # Example fix: resize
                self._log(f"   Resized initial_rotation_target to shape: {initial_rotation_target.shape}")
            # --- END DETAILED DEBUG LOGGING ---


            # --- Nested Objective/Jacobian/Sparsity ---
            def objective(current_rotation_field_rad):
                # Ensure input is usable
                if len(current_rotation_field_rad) != num_cells:
                    self._log(f"   ERROR in objective: Input length {len(current_rotation_field_rad)} != num_cells {num_cells}", logging.ERROR)
                    # Return something sensible or raise error
                    return np.zeros(num_residuals) # Example fallback

                if num_neighbour_pairs > 0:
                    # Use the already validated integer array directly
                    idx0 = cell_face_neighbours[:, 0]
                    idx1 = cell_face_neighbours[:, 1]
                    # No need for astype(np.int32) if cell_face_neighbours is already int32
                    diffs = current_rotation_field_rad[idx0] - current_rotation_field_rad[idx1]
                    neighbour_residuals = np.sqrt(self.params['neighbour_loss_weight']) * diffs
                else:
                    neighbour_residuals = np.array([], dtype=np.float64)

                initial_residuals = current_rotation_field_rad - initial_rotation_target
                # Check shapes before concatenation
                if len(neighbour_residuals) != num_neighbour_pairs or len(initial_residuals) != num_cells:
                    self._log(f"   ERROR in objective concatenation: Shapes mismatch! neighbour_res={len(neighbour_residuals)} (expected {num_neighbour_pairs}), initial_res={len(initial_residuals)} (expected {num_cells})", logging.ERROR)
                    return np.zeros(num_residuals) # Fallback

                return np.concatenate((neighbour_residuals, initial_residuals))

            def jacobian(current_rotation_field_rad):
                # Ensure input is usable
                if len(current_rotation_field_rad) != num_cells:
                    self._log(f"   ERROR in jacobian: Input length {len(current_rotation_field_rad)} != num_cells {num_cells}", logging.ERROR)
                    return csr_matrix((num_residuals, num_cells), dtype=np.float64) # Return empty sparse

                jac = lil_matrix((num_residuals, num_cells), dtype=np.float64)
                sqrt_weight = np.sqrt(self.params['neighbour_loss_weight'])
                if num_neighbour_pairs > 0:
                    rows = np.arange(num_neighbour_pairs, dtype=np.int32)
                    # Use the already validated integer array directly
                    cols0 = cell_face_neighbours[:, 0]
                    cols1 = cell_face_neighbours[:, 1]
                    jac[rows, cols0] = sqrt_weight
                    jac[rows, cols1] = -sqrt_weight

                rows_initial = np.arange(num_neighbour_pairs, num_residuals, dtype=np.int32)
                cols_initial = np.arange(num_cells, dtype=np.int32)
                jac[rows_initial, cols_initial] = 1.0
                return jac.tocsr()

            def sparsity():
                s = lil_matrix((num_residuals, num_cells), dtype=np.int8)
                if num_neighbour_pairs > 0:
                    rows = np.arange(num_neighbour_pairs, dtype=np.int32)
                    cols0 = cell_face_neighbours[:, 0]
                    cols1 = cell_face_neighbours[:, 1]
                    s[rows, cols0] = 1
                    s[rows, cols1] = 1
                rows_initial = np.arange(num_neighbour_pairs, num_residuals, dtype=np.int32)
                cols_initial = np.arange(num_cells, dtype=np.int32)
                s[rows_initial, cols_initial] = 1
                return s.tocsr()
            # --- End Nested Functions ---

            # --- Run Optimization ---
            initial_guess = np.zeros(num_cells, dtype=np.float64)
            self._log(f"   Running least_squares with initial_guess shape: {initial_guess.shape}, dtype: {initial_guess.dtype}")

            # Final check before calling least_squares
            if num_cells <= 0:
                self._log("   ERROR: num_cells is zero or negative. Cannot run optimization.", logging.ERROR)
                return False # Cannot proceed

            # Check jacobian/sparsity shapes one last time
            try:
                jac_sparse = jacobian(initial_guess)
                sparsity_pattern = sparsity()
                expected_shape = (num_residuals, num_cells)
                if jac_sparse.shape != expected_shape or sparsity_pattern.shape != expected_shape:
                    self._log(f"   ERROR: Final Jacobian/Sparsity shape mismatch! Jac={jac_sparse.shape}, Sparsity={sparsity_pattern.shape}, Expected={expected_shape}", logging.ERROR)
                    return False # Cannot proceed
            except Exception as e:
                self._log(f"   ERROR checking final Jacobian/Sparsity: {e}", logging.ERROR)
                return False

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

        # Catch specific errors that might indicate indexing problems
        except IndexError as e:
            self._log(f"Caught IndexError during rotation optimization: {e}", logging.ERROR)
            import traceback
            self._log(traceback.format_exc(), logging.ERROR)
            # Log relevant variables at the time of error
            #self._log(f"   State at error: num_cells={num_cells}, num_neighbour_pairs={num_neighbour_pairs}")
            #self._log(f"   cell_face_neighbours shape={cell_face_neighbours.shape if 'cell_face_neighbours' in locals() else 'Not defined'}")
            return False
        except TypeError as e:
            self._log(f"Caught TypeError during rotation optimization (often related to indexing): {e}", logging.ERROR)
            import traceback
            self._log(traceback.format_exc(), logging.ERROR)
            return False
        except Exception as e:
            self._log(f"Unexpected error during rotation field optimization: {e}", logging.ERROR)
            import traceback
            self._log(traceback.format_exc(), logging.ERROR) # Log full traceback
            return False


    def _calculate_deformation(self):
        self._log("6. Calculating mesh deformation...")
        if self.optimized_rotation_field_rad is None:
            self._log("   Skipping deformation: Optimized rotation field not available.", logging.WARNING)
            return False

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
        gradient = np.zeros(num_cells, dtype=np.float64) # Use float64 for gradient
        distances = np.full(num_cells, np.nan, dtype=np.float64) # Use float64 for distances
        closest_bottom_indices = np.full(num_cells, -1, dtype=np.int32) # Use int32

        if len(self.bottom_cells) == 0 or self.cell_neighbour_graph.number_of_nodes() == 0:
            self._log("Warning: No bottom cells/graph for path length gradient.", logging.WARNING)
            self.undeformed_tet.cell_data["cell_distance_to_bottom"] = distances
            self.undeformed_tet.cell_data["path_length_to_base_gradient"] = gradient
            return gradient

        try:
            # Ensure bottom_cells are integers for Dijkstra
            valid_bottom_cells = {int(bc) for bc in self.bottom_cells if 0 <= int(bc) < num_cells}
            if not valid_bottom_cells:
                self._log("Warning: No valid bottom cells after filtering for path length gradient.", logging.WARNING)
                # Assign NaN/zeros and return early
                self.undeformed_tet.cell_data["cell_distance_to_bottom"] = distances
                self.undeformed_tet.cell_data["path_length_to_base_gradient"] = gradient
                return gradient

            dist_map, path_map = nx.multi_source_dijkstra(self.cell_neighbour_graph, valid_bottom_cells)

            # Ensure overhang_angle exists and handle potential NaNs
            if 'overhang_angle' not in self.undeformed_tet.cell_data:
                self._log("Warning: 'overhang_angle' not found in cell_data for path gradient calc.", logging.WARNING)
                # Create a dummy array or handle appropriately
                overhang_angle = np.full(num_cells, np.nan)
            else:
                overhang_angle = self.undeformed_tet.cell_data['overhang_angle']

            overhang_thresh = np.pi/2.0 + self.params['max_overhang_rad']
            is_overhang = ~np.isnan(overhang_angle) & (overhang_angle > overhang_thresh)

            for cell_idx in range(num_cells):
                # Ensure cell_idx is valid for dist_map access
                if cell_idx in dist_map:
                    distances[cell_idx] = dist_map[cell_idx] # Assign distance if reachable
                    if is_overhang[cell_idx] and cell_idx not in valid_bottom_cells:
                        # Ensure path exists and is not empty before accessing last element
                        if cell_idx in path_map and path_map[cell_idx]:
                            closest_bottom_indices[cell_idx] = int(path_map[cell_idx][-1]) # Ensure integer
                        else:
                            # Handle case where cell is overhang but has no path (should be rare)
                            self._log(f"Warning: Overhang cell {cell_idx} has no path in Dijkstra result.", logging.WARNING)

        except Exception as e:
            self._log(f"Error during Dijkstra path calculation: {e}", logging.WARNING)
            # Continue, but distances/closest_bottom might be incomplete

        self.undeformed_tet.cell_data["cell_distance_to_bottom"] = distances

        # Ensure cell_center exists
        if 'cell_center' not in self.undeformed_tet.cell_data:
            self._log("Error: 'cell_center' not found in cell_data for path gradient calc.", logging.ERROR)
            self.undeformed_tet.cell_data["path_length_to_base_gradient"] = gradient # Return zero gradient
            return gradient
        cell_centers = self.undeformed_tet.cell_data["cell_center"]

        # --- Calculate gradient using plane fitting or fallback ---
        edge_neighbours_dict = self.neighbour_dict.get("edge", {}) # Use .get for safety

        for cell_idx in range(num_cells):
            if not np.isnan(distances[cell_idx]): # Only process cells with a valid distance
                # --- Validate Neighbours ---
                raw_neighbours = edge_neighbours_dict.get(cell_idx, [])
                valid_neighbours = []
                if isinstance(raw_neighbours, (list, tuple, np.ndarray)):
                    for n in raw_neighbours:
                        try:
                            n_int = int(n)
                            # Check bounds immediately
                            if 0 <= n_int < num_cells:
                                valid_neighbours.append(n_int)
                            else:
                                self._log(f"   Skipping out-of-bounds edge neighbour {n_int} for cell {cell_idx}", logging.WARNING)
                        except (ValueError, TypeError):
                            self._log(f"   Skipping invalid edge neighbour value {n} (type: {type(n)}) for cell {cell_idx}", logging.WARNING)
                else:
                    self._log(f"   Invalid neighbour structure for cell {cell_idx}: type {type(raw_neighbours)}", logging.WARNING)
                # --- End Validate Neighbours ---

                # Combine valid neighbours and self, ensure unique, and cast to int32
                combined_indices = np.array(valid_neighbours + [cell_idx], dtype=np.int32)
                local_cells = np.unique(combined_indices)

                # Check if local_cells is now valid before indexing
                if local_cells.size == 0:
                    self._log(f"   Warning: No valid local cells (neighbours + self) for cell {cell_idx} after filtering.", logging.WARNING)
                    gradient[cell_idx] = 0 # Assign default gradient
                    continue # Skip to next cell_idx

                # --- THIS IS THE CRITICAL LINE ---
                try:
                    local_distances = distances[local_cells] # Use the validated integer array
                except IndexError as ie:
                    self._log(f"   CRITICAL INDEX ERROR at distances[local_cells] for cell_idx={cell_idx}", logging.ERROR)
                    self._log(f"      distances shape: {distances.shape}, dtype: {distances.dtype}")
                    self._log(f"      local_cells: {local_cells}, dtype: {local_cells.dtype}")
                    self._log(f"      Min/Max local_cells: {np.min(local_cells)}, {np.max(local_cells)}")
                    # Assign default gradient and continue to avoid crashing
                    gradient[cell_idx] = 0
                    continue # Skip to next cell_idx
                except Exception as e:
                    self._log(f"   Unexpected error indexing distances for cell_idx={cell_idx}: {e}", logging.ERROR)
                    gradient[cell_idx] = 0
                    continue # Skip to next cell_idx
                # --- END CRITICAL LINE ---

                valid_mask = ~np.isnan(local_distances)
                local_cells_filtered = local_cells[valid_mask] # Filter based on valid distances
                local_distances_filtered = local_distances[valid_mask]

                # Ensure cell_idx is valid for cell_centers access
                if not (0 <= cell_idx < cell_centers.shape[0]):
                    self._log(f"   Error: cell_idx {cell_idx} out of bounds for cell_centers (shape {cell_centers.shape})", logging.ERROR)
                    gradient[cell_idx] = 0
                    continue

                current_center_xy = cell_centers[cell_idx, :2]
                norm_cc_xy = np.linalg.norm(current_center_xy)

                if len(local_cells_filtered) < 3: # Fallback: Roll towards closest bottom
                    closest_idx = closest_bottom_indices[cell_idx]
                    grad_val = 0.0 # Default to float
                    # Ensure closest_idx is valid before using it
                    if 0 <= closest_idx < cell_centers.shape[0]:
                        dir_to_bottom = cell_centers[closest_idx, :2] - current_center_xy
                        norm_dir = np.linalg.norm(dir_to_bottom)
                        if norm_dir > 1e-6 and norm_cc_xy > 1e-6:
                            dir_to_bottom /= norm_dir
                            center_dir = current_center_xy / norm_cc_xy
                            dot_prod = np.dot(center_dir, dir_to_bottom)
                            grad_val = np.sign(dot_prod) if not np.isnan(dot_prod) else 0.0
                    gradient[cell_idx] = grad_val
                else: # Plane fitting
                    # Ensure local_cells_filtered are valid indices for cell_centers
                    if np.any(local_cells_filtered < 0) or np.any(local_cells_filtered >= cell_centers.shape[0]):
                        self._log(f"   Error: Invalid indices in local_cells_filtered for plane fitting (cell {cell_idx})", logging.ERROR)
                        gradient[cell_idx] = 0.0
                        continue

                    points_fit = np.vstack((cell_centers[local_cells_filtered, :2].T, local_distances_filtered))
                    try:
                        _, normal = planeFit(points_fit)
                        grad_xy = -normal[:2]
                        grad_val = 0.0
                        if norm_cc_xy > 1e-6:
                            center_dir = current_center_xy / norm_cc_xy
                            grad_val = np.dot(center_dir, grad_xy)
                        gradient[cell_idx] = grad_val if not np.isnan(grad_val) else 0.0
                    except Exception as pf_e:
                        self._log(f"   Warning: Plane fit failed for cell {cell_idx}: {pf_e}", logging.WARNING)
                        gradient[cell_idx] = 0.0 # Plane fit failed

        # Smoothing
        if self.params['initial_rotation_field_smoothing'] > 0:
            smoothed = gradient.copy()
            mask_initial = (gradient != 0) & (~np.isnan(gradient)) # Use original gradient for mask
            point_neighbours_dict = self.neighbour_dict.get("point", {}) # Use point neighbours for smoothing

            for _ in range(self.params['initial_rotation_field_smoothing']):
                new_smoothed = smoothed.copy()
                # Iterate only over cells that initially had a non-zero gradient
                for cell_idx in np.where(mask_initial)[0]:
                    # --- Validate Point Neighbours ---
                    raw_point_neighbours = point_neighbours_dict.get(cell_idx, [])
                    valid_point_neighbours = []
                    if isinstance(raw_point_neighbours, (list, tuple, np.ndarray)):
                        for n in raw_point_neighbours:
                            try:
                                n_int = int(n)
                                if 0 <= n_int < num_cells: # Check bounds
                                    valid_point_neighbours.append(n_int)
                            except (ValueError, TypeError): pass # Ignore invalid neighbours silently during smoothing
                    # --- End Validate Point Neighbours ---

                    if valid_point_neighbours:
                        # Use the *previous* smoothed values of valid neighbours
                        neighbour_values = smoothed[valid_point_neighbours]
                        # Filter out NaNs from neighbour values before averaging
                        valid_neighbour_values = neighbour_values[~np.isnan(neighbour_values)]
                        if valid_neighbour_values.size > 0:
                            new_smoothed[cell_idx] = np.mean(valid_neighbour_values)
                        # else: keep original smoothed value if all neighbours were NaN

                smoothed = new_smoothed
            gradient = smoothed # Assign smoothed result back

        # Final assignment based on parameter
        if not self.params['set_initial_rotation_to_zero']:
            # Keep NaNs where distance was NaN, otherwise use calculated/smoothed gradient
            gradient[np.isnan(distances)] = np.nan
        else:
            # Convert all remaining NaNs (e.g., from smoothing isolated cells) to zero
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

    # --- Public Execution Method ---
    def run(self):
        """Executes the full deformation pipeline."""
        self._log("--- Starting Tetrahedral Deformation ---")
        start_total_time = time.time()
        self.success = False # Reset success flag

        steps = [
            self._tetrahedralize, # start by making a volume out of surface model
            self._center_mesh, # force mesh to be at the bed
            self._find_neighbours, # find the neighbors of each cell
            self._calculate_attributes, # normals, graph ...
            self._optimize_rotations, # 
            self._calculate_deformation,
            self._create_final_mesh
        ]

        for step_func in steps:
            if not step_func(): # Execute step and check success
                self._log(f"Pipeline stopped due to error in step: {step_func.__name__}", logging.ERROR)
                # Clean up plotter if it was initialized
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
# need because i dont wanna change the main function
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

