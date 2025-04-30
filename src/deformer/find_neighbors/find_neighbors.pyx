import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set

# Define typed numpy arrays
ctypedef np.int32_t INT32_t
ctypedef np.float64_t FLOAT64_t

def find_neighbors_cython(np.ndarray[INT32_t, ndim=2] cells, int n_cells):
    """
    Fast neighbor computation using Cython.
    Returns point, edge, and face neighbors for each cell.
    """
    # Pre-allocate results
    cdef vector[vector[int]] point_neighbors = vector[vector[int]](n_cells)
    cdef vector[vector[int]] edge_neighbors = vector[vector[int]](n_cells)
    cdef vector[vector[int]] face_neighbors = vector[vector[int]](n_cells)
    
    # Create maps for fast lookups
    cdef unordered_map[int, vector[int]] point_to_cells
    
    # Build lookup maps
    cdef int i, j, p1
    
    # Build point-to-cell map (which cells contain each point)
    for i in range(n_cells):
        for j in range(4):
            p1 = cells[i, j]
            point_to_cells[p1].push_back(i)
    
    # Find neighbors by point sharing
    cdef unordered_set[int] point_neighbors_set
    for i in range(n_cells):
        point_neighbors_set.clear()
        
        # For each point in cell, get all cells containing that point
        for j in range(4):
            p1 = cells[i, j]
            for neighbor_idx in point_to_cells[p1]:
                if neighbor_idx != i:  # Don't include self
                    point_neighbors_set.insert(neighbor_idx)
        
        # Convert set to vector
        for j in point_neighbors_set:
            point_neighbors[i].push_back(j)
    
    # Convert to Python format
    result = {
        "point": {i: list(point_neighbors[i]) for i in range(n_cells)},
        "edge": {i: list(edge_neighbors[i]) for i in range(n_cells)},
        "face": {i: list(face_neighbors[i]) for i in range(n_cells)}
    }
    
    return result


def calculate_attributes_cython(np.ndarray[FLOAT64_t, ndim=2] points,
                              np.ndarray[INT32_t, ndim=2] cells,
                              np.ndarray[INT32_t, ndim=2] faces,
                              dict neighbor_dict):
    """
    Fast calculation of tetrahedral attributes using Cython.
    Returns cell-to-face mapping, bottom cells, and adjacency information.
    """
    cdef int n_cells = cells.shape[0]
    cdef int n_faces = faces.shape[0]
    cdef int n_points = points.shape[0]
    cdef int i, j, k, v1, v2, v3, face_idx, idx_val, neighbor_idx
    cdef double dx, dy, dz, dist_val, nx, ny, nz, norm, min_z = float('inf')
    
    # Calculate cell centers - use float64 to match points dtype
    cdef np.ndarray[FLOAT64_t, ndim=2] cell_centers = np.zeros((n_cells, 3), dtype=np.float64)
    for i in range(n_cells):
        for j in range(3):  # x, y, z
            cell_centers[i, j] = (points[cells[i, 0], j] + 
                               points[cells[i, 1], j] + 
                               points[cells[i, 2], j] + 
                               points[cells[i, 3], j]) / 4.0
    
    # Calculate face centers and normals - also use float64
    cdef np.ndarray[FLOAT64_t, ndim=2] face_centers = np.zeros((n_faces, 3), dtype=np.float64)
    cdef np.ndarray[FLOAT64_t, ndim=2] face_normals = np.zeros((n_faces, 3), dtype=np.float64)
    
    for i in range(n_faces):
        # Get vertices of the face
        v1, v2, v3 = faces[i, 0], faces[i, 1], faces[i, 2]
        
        # Calculate face center
        for j in range(3):
            face_centers[i, j] = (points[v1, j] + points[v2, j] + points[v3, j]) / 3.0
        
        # Calculate face normal using cross product
        # Vector 1: v2 - v1
        dx = points[v2, 0] - points[v1, 0]
        dy = points[v2, 1] - points[v1, 1]
        dz = points[v2, 2] - points[v1, 2]
        
        # Vector 2: v3 - v1
        dx2 = points[v3, 0] - points[v1, 0]
        dy2 = points[v3, 1] - points[v1, 1]
        dz2 = points[v3, 2] - points[v1, 2]
        
        # Cross product
        nx = dy * dz2 - dz * dy2
        ny = dz * dx2 - dx * dz2
        nz = dx * dy2 - dy * dx2
        
        # Normalize
        norm = sqrt(nx*nx + ny*ny + nz*nz)
        if norm > 1e-10:
            face_normals[i, 0] = nx / norm
            face_normals[i, 1] = ny / norm
            face_normals[i, 2] = nz / norm
        
        # Track minimum z coordinate for bottom cells
        if face_centers[i, 2] < min_z:
            min_z = face_centers[i, 2]
    
    # Find cell-to-face mapping using KDTree
    from scipy.spatial import KDTree
    face_centers_tree = KDTree(face_centers)
    
    # For each cell, find the closest faces
    cell_to_face = {}
    bottom_cells = []
    bottom_threshold = min_z + 0.3  # mm tolerance
    
    for i in range(n_cells):
        # Get the vertices of the cell
        cell_vertices = []
        for j in range(4):
            cell_vertices.append(points[cells[i, j]])
        
        # Query KDTree for each vertex
        closest_faces = set()
        for vertex in cell_vertices:
            distances, indices = face_centers_tree.query(vertex, k=3)
            
            # Handle both scalar and array return types
            if np.isscalar(distances):
                if distances < 1e-5:
                    closest_faces.add(int(indices))
            else:
                for k in range(len(indices)):
                    if distances[k] < 1e-5:
                        closest_faces.add(int(indices[k]))
        
        # Store cell-to-face mapping
        if closest_faces:
            cell_to_face[i] = [int(face) for face in closest_faces]
            
            # Check if this is a bottom cell
            for face_idx in closest_faces:
                if face_centers[int(face_idx), 2] < bottom_threshold:
                    bottom_cells.append(int(i))
                    break
    
    # Build adjacency matrix for the neighborhood graph
    cdef np.ndarray[FLOAT64_t, ndim=2] adjacency_matrix = np.zeros((n_cells, n_cells), dtype=np.float64)
    
    # Process point neighbors to build adjacency matrix
    point_neighbors = neighbor_dict.get("point", {})
    for i in range(n_cells):
        if i in point_neighbors:
            neighbors = point_neighbors[i]
            for neighbor_idx in neighbors:
                neighbor_idx = int(neighbor_idx)
                if neighbor_idx > i:  # Only process higher indices to avoid duplicates
                    # Calculate Euclidean distance between cell centers
                    dx = cell_centers[i, 0] - cell_centers[neighbor_idx, 0]
                    dy = cell_centers[i, 1] - cell_centers[neighbor_idx, 1]
                    dz = cell_centers[i, 2] - cell_centers[neighbor_idx, 2]
                    dist_val = sqrt(dx*dx + dy*dy + dz*dz)
                    
                    # Set both directions for symmetric matrix
                    adjacency_matrix[i, neighbor_idx] = dist_val
                    adjacency_matrix[neighbor_idx, i] = dist_val
    
    return {
        "cell_to_face": cell_to_face,
        "bottom_cells": bottom_cells,
        "cell_centers": cell_centers,
        "adjacency_matrix": adjacency_matrix
    }
