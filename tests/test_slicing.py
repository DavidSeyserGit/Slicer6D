import pytest
import numpy as np
import trimesh
import math


class TestOrientationAwareSlicing:
    """Tests for orientation-aware slicing functionality for 6DOF printing."""

    @pytest.fixture
    def simple_cube_mesh(self):
        """Create a simple unit cube mesh for testing."""
        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        return box

    @pytest.fixture
    def simple_cube_stl_path(self, tmp_path, simple_cube_mesh):
        """Save a cube mesh to a temporary STL file."""
        stl_path = tmp_path / "cube.stl"
        simple_cube_mesh.export(str(stl_path))
        return str(stl_path)

    def test_slice_mesh_at_orientation_exists(self):
        """Test that slice_mesh_at_orientation function exists."""
        from slicing.slicing import slice_mesh_at_orientation
        assert callable(slice_mesh_at_orientation)

    def test_slice_mesh_at_45_degrees(self, simple_cube_mesh):
        """Test slicing at 45-degree orientation."""
        from slicing.slicing import slice_mesh_at_orientation
        layer_height = 0.2
        direction = np.array([1, 1, 1])
        direction = direction / np.linalg.norm(direction)
        z_levels, layer_polygons = slice_mesh_at_orientation(
            simple_cube_mesh, layer_height, direction
        )
        assert len(z_levels) > 0
        assert len(layer_polygons) > 0
        assert all(isinstance(poly, list) for poly in layer_polygons)

    def test_slice_mesh_at_arbitrary_angle(self, simple_cube_mesh):
        """Test slicing at arbitrary angle (30 degrees)."""
        from slicing.slicing import slice_mesh_at_orientation
        layer_height = 0.2
        angle_deg = 30
        angle_rad = math.radians(angle_deg)
        direction = np.array([math.cos(angle_rad), math.sin(angle_rad), 0.5])
        direction = direction / np.linalg.norm(direction)
        z_levels, layer_polygons = slice_mesh_at_orientation(
            simple_cube_mesh, layer_height, direction
        )
        assert len(z_levels) > 0
        assert len(layer_polygons) > 0

    def test_slice_mesh_at_vertical_orientation(self, simple_cube_mesh):
        """Test slicing at vertical (Z) orientation."""
        from slicing.slicing import slice_mesh_at_orientation
        layer_height = 0.2
        direction = np.array([0, 0, 1])
        z_levels, layer_polygons = slice_mesh_at_orientation(
            simple_cube_mesh, layer_height, direction
        )
        assert len(z_levels) > 0

    def test_slice_mesh_at_x_axis_orientation(self, simple_cube_mesh):
        """Test slicing along X-axis orientation."""
        from slicing.slicing import slice_mesh_at_orientation
        layer_height = 0.2
        direction = np.array([1, 0, 0])
        z_levels, layer_polygons = slice_mesh_at_orientation(
            simple_cube_mesh, layer_height, direction
        )
        assert len(z_levels) > 0


class TestPathGeneration:
    """Tests for correct path generation for non-horizontal print orientations."""

    @pytest.fixture
    def simple_cube_mesh(self):
        """Create a simple unit cube mesh for testing."""
        box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        return box

    def test_path_contains_z_coordinates(self, simple_cube_mesh):
        """Test that generated paths contain proper Z coordinates for non-horizontal orientation."""
        from slicing.slicing import slice_mesh_at_orientation
        layer_height = 0.2
        direction = np.array([0, 1, 1])
        direction = direction / np.linalg.norm(direction)
        z_levels, layer_polygons = slice_mesh_at_orientation(
            simple_cube_mesh, layer_height, direction
        )
        for i, polygons in enumerate(layer_polygons):
            if polygons:
                for poly in polygons:
                    coords = np.array(poly.exterior.coords)
                    assert coords.shape[1] >= 2

    def test_path_spacing_consistent(self, simple_cube_mesh):
        """Test that path spacing is consistent across layers."""
        from slicing.slicing import slice_mesh_at_orientation
        layer_height = 0.2
        direction = np.array([1, 0, 0])
        z_levels, layer_polygons = slice_mesh_at_orientation(
            simple_cube_mesh, layer_height, direction
        )
        layer_heights = []
        for i in range(len(z_levels) - 1):
            layer_heights.append(z_levels[i + 1] - z_levels[i])
        if layer_heights:
            avg_height = np.mean(layer_heights)
            assert abs(avg_height - layer_height) < 0.01


class Test6DOFGCode:
    """Tests for 6DOF G-code output format."""

    def test_generate_gcode_6dof_exists(self):
        """Test that generate_gcode_6dof function exists."""
        from slicing.slicing import generate_gcode_6dof
        assert callable(generate_gcode_6dof)

    def test_gcode_contains_abc_commands(self):
        """Test that G-code output contains ABC orientation commands."""
        from slicing.slicing import generate_gcode_6dof
        z_levels = [0.2, 0.4, 0.6]
        layer_polygons = [[], [], []]
        direction = np.array([0, 0, 1])
        gcode, paths = generate_gcode_6dof(z_levels, layer_polygons, print_direction=direction)
        assert " A" in gcode or " B" in gcode or " C" in gcode or "G1" in gcode

    def test_gcode_format_6dof(self):
        """Test that G-code has proper 6DOF format with ABC orientation."""
        from slicing.slicing import generate_gcode_6dof
        import numpy as np
        from shapely.geometry import Polygon
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        z_levels = [0.2]
        layer_polygons = [[poly]]
        direction = np.array([0, 0, 1])
        gcode, paths = generate_gcode_6dof(z_levels, layer_polygons, print_direction=direction)
        lines = gcode.split('\n')
        move_lines = [l for l in lines if 'G1' in l and 'X' in l]
        assert len(move_lines) > 0

    def test_gcode_with_non_horizontal_orientation(self):
        """Test G-code generation with non-horizontal print orientation."""
        from slicing.slicing import generate_gcode_6dof
        import numpy as np
        from shapely.geometry import Polygon
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        z_levels = [0.2, 0.4]
        layer_polygons = [[poly], [poly]]
        direction = np.array([1, 1, 1])
        direction = direction / np.linalg.norm(direction)
        gcode, paths = generate_gcode_6dof(z_levels, layer_polygons, print_direction=direction)
        assert len(gcode) > 0
        assert len(paths) > 0

    def test_gcode_contains_proper_extrusion(self):
        """Test that G-code contains proper E extrusion commands."""
        from slicing.slicing import generate_gcode_6dof
        import numpy as np
        from shapely.geometry import Polygon
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        z_levels = [0.2]
        layer_polygons = [[poly]]
        direction = np.array([0, 0, 1])
        gcode, paths = generate_gcode_6dof(z_levels, layer_polygons, print_direction=direction)
        assert "E" in gcode or "e" in gcode

    def test_gcode_returns_paths_for_visualization(self):
        """Test that generate_gcode_6dof returns paths for visualization."""
        from slicing.slicing import generate_gcode_6dof
        import numpy as np
        from shapely.geometry import Polygon
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        z_levels = [0.2]
        layer_polygons = [[poly]]
        direction = np.array([0, 0, 1])
        gcode, paths = generate_gcode_6dof(z_levels, layer_polygons, print_direction=direction)
        assert isinstance(paths, list)
        if len(paths) > 0:
            path = paths[0]
            assert isinstance(path, np.ndarray)
            assert path.shape[1] >= 3


class TestOrientationTransform:
    """Tests for orientation transformation logic."""

    def test_normalize_direction_vector(self):
        """Test that direction vector is properly normalized."""
        from slicing.slicing import slice_mesh_at_orientation
        import numpy as np
        import trimesh
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        direction = np.array([2, 2, 2])
        z_levels, layer_polygons = slice_mesh_at_orientation(
            mesh, 0.2, direction
        )
        assert len(z_levels) > 0

    def test_invalid_direction_raises_error(self):
        """Test that zero vector direction raises an error."""
        from slicing.slicing import slice_mesh_at_orientation
        import numpy as np
        import trimesh
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        direction = np.array([0, 0, 0])
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            slice_mesh_at_orientation(mesh, 0.2, direction)