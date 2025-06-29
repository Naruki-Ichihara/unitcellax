"""Comprehensive tests for the UnitCell abstract base class.

This module tests all functionality of the UnitCell class including:
- Abstract base class behavior
- Geometric queries and boundary identification
- Coordinate transformations and mapping
- Macro displacement calculations
- Edge cases and error handling

The tests verify both correctness and robustness of the implementation.
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as onp
from unittest.mock import Mock, patch
from typing import Tuple

from unitcellax.unitcell import UnitCell
from unitcellax.fem.mesh import box_mesh, Mesh


class CubeUnitCell(UnitCell):
    """Concrete implementation of UnitCell for testing purposes."""
    
    def mesh_build(self, nx: int = 5, ny: int = 5, nz: int = 5, **kwargs) -> Mesh:
        """Build a structured cube mesh."""
        return box_mesh(nx, ny, nz, 1.0, 1.0, 1.0)


class QuadUnitCell(UnitCell):
    """2D rectangular unit cell for testing 2D functionality."""
    
    def mesh_build(self, nx: int = 4, ny: int = 4, **kwargs) -> Mesh:
        """Build a 2D structured mesh using QUAD4 elements."""
        # Create a simple 2D grid
        x = jnp.linspace(0, 1, nx + 1)
        y = jnp.linspace(0, 1, ny + 1)
        
        points = []
        for j in range(ny + 1):
            for i in range(nx + 1):
                points.append([x[i], y[j]])
        
        cells = []
        for j in range(ny):
            for i in range(nx):
                # QUAD4 connectivity
                p0 = j * (nx + 1) + i
                p1 = j * (nx + 1) + i + 1
                p2 = (j + 1) * (nx + 1) + i + 1
                p3 = (j + 1) * (nx + 1) + i
                cells.append([p0, p1, p2, p3])
        
        mesh = Mesh(
            points=jnp.array(points),
            cells=jnp.array(cells),
            ele_type="QUAD4"
        )
        
        return mesh


class TestUnitCellAbstract:
    """Test abstract base class behavior."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that UnitCell cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class UnitCell"):
            UnitCell()
    
    def test_must_implement_mesh_build(self):
        """Test that subclasses must implement mesh_build."""
        class IncompleteUnitCell(UnitCell):
            pass
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteUnitCell()
    
    def test_concrete_implementation_works(self):
        """Test that concrete implementations can be instantiated."""
        unit_cell = CubeUnitCell()
        assert unit_cell is not None
        assert hasattr(unit_cell, 'mesh')
        assert hasattr(unit_cell, 'points')
        assert hasattr(unit_cell, 'cells')


class TestUnitCellInitialization:
    """Test UnitCell initialization and setup."""
    
    def test_basic_initialization(self):
        """Test basic initialization with default parameters."""
        unit_cell = CubeUnitCell()
        
        assert unit_cell.atol == 1e-6
        assert unit_cell.mesh is not None
        assert unit_cell.points.shape[1] == 3  # 3D coordinates
        assert unit_cell.cells.shape[0] > 0   # Has elements
        assert unit_cell.ele_type == "HEX8"
        
        # Check bounding box
        assert hasattr(unit_cell, 'lb')
        assert hasattr(unit_cell, 'ub')
        assert jnp.allclose(unit_cell.lb, jnp.array([0.0, 0.0, 0.0]))
        assert jnp.allclose(unit_cell.ub, jnp.array([1.0, 1.0, 1.0]))
    
    def test_custom_tolerance(self):
        """Test initialization with custom tolerance."""
        custom_atol = 1e-8
        unit_cell = CubeUnitCell(atol=custom_atol)
        assert unit_cell.atol == custom_atol
    
    def test_mesh_parameters_passed(self):
        """Test that parameters are passed to mesh_build."""
        unit_cell = CubeUnitCell(nx=3, ny=3, nz=3)
        # For 3x3x3 mesh: (3+1)^3 = 64 nodes
        assert unit_cell.points.shape[0] == 64
    
    def test_2d_unit_cell(self):
        """Test 2D unit cell initialization."""
        unit_cell = QuadUnitCell()
        
        assert unit_cell.points.shape[1] == 2  # 2D coordinates
        assert unit_cell.ele_type == "QUAD4"
        assert jnp.allclose(unit_cell.lb, jnp.array([0.0, 0.0]))
        assert jnp.allclose(unit_cell.ub, jnp.array([1.0, 1.0]))


class TestUnitCellProperties:
    """Test UnitCell computed properties."""
    
    @pytest.fixture
    def cube_unit_cell(self):
        """Fixture providing a standard cube unit cell."""
        return CubeUnitCell(nx=2, ny=2, nz=2)
    
    @pytest.fixture
    def quad_unit_cell(self):
        """Fixture providing a standard 2D unit cell."""
        return QuadUnitCell(nx=2, ny=2)
    
    def test_cell_centers_3d(self, cube_unit_cell):
        """Test cell centers calculation for 3D mesh."""
        centers = cube_unit_cell.cell_centers
        
        # 2x2x2 mesh has 8 elements
        assert centers.shape == (8, 3)
        
        # Check that centers are within the unit cube
        assert jnp.all(centers >= 0.0)
        assert jnp.all(centers <= 1.0)
        
        # Check that centers are reasonable (should be around 0.25, 0.5, 0.75 in each direction)
        expected_coords = [0.25, 0.75]
        for center in centers:
            for coord in center:
                assert jnp.any(jnp.isclose(coord, jnp.array(expected_coords), atol=1e-3))
    
    def test_cell_centers_2d(self, quad_unit_cell):
        """Test cell centers calculation for 2D mesh."""
        centers = quad_unit_cell.cell_centers
        
        # 2x2 mesh has 4 elements
        assert centers.shape == (4, 2)
        assert jnp.all(centers >= 0.0)
        assert jnp.all(centers <= 1.0)
    
    def test_bounds_3d(self, cube_unit_cell):
        """Test bounds property for 3D unit cell."""
        lb, ub = cube_unit_cell.bounds
        
        assert jnp.allclose(lb, jnp.array([0.0, 0.0, 0.0]))
        assert jnp.allclose(ub, jnp.array([1.0, 1.0, 1.0]))
    
    def test_bounds_2d(self, quad_unit_cell):
        """Test bounds property for 2D unit cell."""
        lb, ub = quad_unit_cell.bounds
        
        assert jnp.allclose(lb, jnp.array([0.0, 0.0]))
        assert jnp.allclose(ub, jnp.array([1.0, 1.0]))
    
    def test_corners_3d(self, cube_unit_cell):
        """Test corners calculation for 3D unit cell."""
        corners = cube_unit_cell.corners
        
        # 3D cube has 8 corners
        assert corners.shape == (8, 3)
        
        # Expected corners of unit cube
        expected_corners = jnp.array([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
        ], dtype=float)
        
        # Check that all expected corners are present
        for expected_corner in expected_corners:
            # Find matches by computing distances to all corners
            distances = jnp.linalg.norm(corners - expected_corner, axis=1)
            found = jnp.any(distances < 1e-6)
            assert found, f"Expected corner {expected_corner} not found"
    
    def test_corners_2d(self, quad_unit_cell):
        """Test corners calculation for 2D unit cell."""
        corners = quad_unit_cell.corners
        
        # 2D rectangle has 4 corners
        assert corners.shape == (4, 2)
        
        expected_corners = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        
        for expected_corner in expected_corners:
            distances = jnp.linalg.norm(corners - expected_corner, axis=1)
            found = jnp.any(distances < 1e-6)
            assert found, f"Expected corner {expected_corner} not found"


class TestGeometricQueries:
    """Test geometric point classification methods."""
    
    @pytest.fixture
    def cube_unit_cell(self):
        return CubeUnitCell(nx=2, ny=2, nz=2)
    
    def test_is_corner_3d(self, cube_unit_cell):
        """Test corner detection for 3D points."""
        # Test actual corners
        assert cube_unit_cell.is_corner(jnp.array([0.0, 0.0, 0.0]))
        assert cube_unit_cell.is_corner(jnp.array([1.0, 1.0, 1.0]))
        assert cube_unit_cell.is_corner(jnp.array([0.0, 1.0, 0.0]))
        
        # Test non-corners
        assert not cube_unit_cell.is_corner(jnp.array([0.5, 0.5, 0.5]))  # Center
        assert not cube_unit_cell.is_corner(jnp.array([0.0, 0.0, 0.5]))  # Edge
        assert not cube_unit_cell.is_corner(jnp.array([0.0, 0.5, 0.5]))  # Face
    
    def test_is_edge_3d(self, cube_unit_cell):
        """Test edge detection for 3D points."""
        # Test actual edges (2 coordinates at boundaries, 1 free)
        assert cube_unit_cell.is_edge(jnp.array([0.0, 0.0, 0.5]))  # Bottom edge
        assert cube_unit_cell.is_edge(jnp.array([0.0, 1.0, 0.5]))  # Top edge
        assert cube_unit_cell.is_edge(jnp.array([0.5, 0.0, 0.0]))  # Side edge
        
        # Test non-edges
        assert not cube_unit_cell.is_edge(jnp.array([0.0, 0.0, 0.0]))  # Corner
        assert not cube_unit_cell.is_edge(jnp.array([0.5, 0.5, 0.5]))  # Interior
        assert not cube_unit_cell.is_edge(jnp.array([0.0, 0.5, 0.5]))  # Face interior
    
    def test_is_face_3d(self, cube_unit_cell):
        """Test face detection for 3D points."""
        # Test actual face points (1 coordinate at boundary, 2 free)
        assert cube_unit_cell.is_face(jnp.array([0.0, 0.5, 0.5]))  # Left face center
        assert cube_unit_cell.is_face(jnp.array([1.0, 0.5, 0.5]))  # Right face center
        assert cube_unit_cell.is_face(jnp.array([0.5, 0.0, 0.5]))  # Front face center
        
        # Test non-faces
        assert not cube_unit_cell.is_face(jnp.array([0.0, 0.0, 0.0]))  # Corner
        assert not cube_unit_cell.is_face(jnp.array([0.0, 0.0, 0.5]))  # Edge
        assert not cube_unit_cell.is_face(jnp.array([0.5, 0.5, 0.5]))  # Interior
    
    def test_tolerance_effects(self):
        """Test how different tolerances affect geometric queries."""
        strict_cell = CubeUnitCell(atol=1e-8)
        loose_cell = CubeUnitCell(atol=1e-2)
        
        # Point very close to corner but not exact
        near_corner = jnp.array([1e-4, 0.0, 0.0])
        
        assert not strict_cell.is_corner(near_corner)
        assert loose_cell.is_corner(near_corner)


class TestBoundaryMasks:
    """Test boundary identification masks."""
    
    @pytest.fixture
    def cube_unit_cell(self):
        return CubeUnitCell(nx=2, ny=2, nz=2)
    
    def test_corner_mask(self, cube_unit_cell):
        """Test corner mask identifies correct nodes."""
        corner_mask = cube_unit_cell.corner_mask
        
        # Should identify exactly 8 corners for a cube
        assert jnp.sum(corner_mask) == 8
        
        # Verify actual corner coordinates
        corner_points = cube_unit_cell.points[corner_mask]
        assert corner_points.shape == (8, 3)
        
        # All corner points should have coordinates at 0 or 1
        for point in corner_points:
            for coord in point:
                assert jnp.isclose(coord, 0.0) or jnp.isclose(coord, 1.0)
    
    def test_edge_mask(self, cube_unit_cell):
        """Test edge mask identifies correct nodes."""
        edge_mask = cube_unit_cell.edge_mask
        
        # Should have some edge nodes (depends on mesh density)
        assert jnp.sum(edge_mask) > 0
        
        # Edge points should not be corners
        corner_mask = cube_unit_cell.corner_mask
        assert jnp.sum(jnp.logical_and(edge_mask, corner_mask)) == 0
    
    def test_face_mask(self, cube_unit_cell):
        """Test face mask identifies correct nodes."""
        face_mask = cube_unit_cell.face_mask
        
        # Should have some face nodes
        assert jnp.sum(face_mask) > 0
        
        # Face points should not be corners or edges
        corner_mask = cube_unit_cell.corner_mask
        edge_mask = cube_unit_cell.edge_mask
        
        assert jnp.sum(jnp.logical_and(face_mask, corner_mask)) == 0
        assert jnp.sum(jnp.logical_and(face_mask, edge_mask)) == 0
    
    def test_mask_completeness(self, cube_unit_cell):
        """Test that masks partition the boundary nodes correctly."""
        corner_mask = cube_unit_cell.corner_mask
        edge_mask = cube_unit_cell.edge_mask
        face_mask = cube_unit_cell.face_mask
        
        # No overlaps
        assert jnp.sum(jnp.logical_and(corner_mask, edge_mask)) == 0
        assert jnp.sum(jnp.logical_and(corner_mask, face_mask)) == 0
        assert jnp.sum(jnp.logical_and(edge_mask, face_mask)) == 0


class TestBoundaryFunctions:
    """Test boundary identification functions."""
    
    @pytest.fixture
    def cube_unit_cell(self):
        return CubeUnitCell(nx=3, ny=3, nz=3)
    
    def test_face_function_basic(self, cube_unit_cell):
        """Test basic face function creation and usage."""
        # Left face (x = 0)
        left_face = cube_unit_cell.face_function(axis=0, value=0.0)
        
        # Test function works
        assert left_face(jnp.array([0.0, 0.5, 0.5])) == True
        assert left_face(jnp.array([0.5, 0.5, 0.5])) == False
        assert left_face(jnp.array([0.0, 0.0, 0.0])) == True  # Corner on face
    
    def test_face_function_excluding_corners(self, cube_unit_cell):
        """Test face function with corner exclusion."""
        left_face = cube_unit_cell.face_function(axis=0, value=0.0, excluding_corner=True)
        
        # Should exclude corners
        assert left_face(jnp.array([0.0, 0.0, 0.0])) == False  # Corner
        assert left_face(jnp.array([0.0, 0.5, 0.5])) == True   # Face interior
    
    def test_face_function_excluding_edges(self, cube_unit_cell):
        """Test face function with edge exclusion."""
        left_face = cube_unit_cell.face_function(axis=0, value=0.0, excluding_edge=True)
        
        # Should exclude edges and corners
        assert left_face(jnp.array([0.0, 0.0, 0.0])) == False  # Corner
        assert left_face(jnp.array([0.0, 0.0, 0.5])) == False  # Edge
        assert left_face(jnp.array([0.0, 0.5, 0.5])) == True   # Face interior
    
    def test_edge_function_basic(self, cube_unit_cell):
        """Test basic edge function creation and usage."""
        # Bottom-left edge (x=0, y=0)
        bottom_left_edge = cube_unit_cell.edge_function([0, 1], [0.0, 0.0])
        
        assert bottom_left_edge(jnp.array([0.0, 0.0, 0.5])) == True   # On edge
        assert bottom_left_edge(jnp.array([0.0, 0.0, 0.0])) == True   # Corner on edge
        assert bottom_left_edge(jnp.array([0.0, 0.5, 0.0])) == False  # Different edge
        assert bottom_left_edge(jnp.array([0.5, 0.0, 0.0])) == False  # Different edge
    
    def test_edge_function_excluding_corners(self, cube_unit_cell):
        """Test edge function with corner exclusion."""
        bottom_left_edge = cube_unit_cell.edge_function([0, 1], [0.0, 0.0], excluding_corner=True)
        
        assert bottom_left_edge(jnp.array([0.0, 0.0, 0.0])) == False  # Corner
        assert bottom_left_edge(jnp.array([0.0, 0.0, 0.5])) == True   # Edge interior
    
    def test_corner_function(self, cube_unit_cell):
        """Test corner function creation and usage."""
        origin_corner = cube_unit_cell.corner_function([0.0, 0.0, 0.0])
        
        assert origin_corner(jnp.array([0.0, 0.0, 0.0])) == True
        assert origin_corner(jnp.array([0.0, 0.0, 0.1])) == False
        assert origin_corner(jnp.array([1.0, 1.0, 1.0])) == False
    
    def test_face_function_vectorized(self, cube_unit_cell):
        """Test face function works with vectorized operations."""
        left_face = cube_unit_cell.face_function(axis=0, value=0.0)
        
        # Apply to all points
        face_mask = jax.vmap(left_face)(cube_unit_cell.points)
        
        # Should identify some points on the left face
        assert jnp.sum(face_mask) > 0
        
        # All identified points should have x ≈ 0
        face_points = cube_unit_cell.points[face_mask]
        assert jnp.allclose(face_points[:, 0], 0.0, atol=cube_unit_cell.atol)


class TestMapping:
    """Test coordinate mapping functionality."""
    
    @pytest.fixture
    def cube_unit_cell(self):
        return CubeUnitCell(nx=2, ny=2, nz=2)
    
    def test_basic_mapping(self, cube_unit_cell):
        """Test basic mapping between opposite faces."""
        left_face = cube_unit_cell.face_function(axis=0, value=0.0)
        right_face = cube_unit_cell.face_function(axis=0, value=1.0)
        
        # Create mapping
        mapper = cube_unit_cell.mapping(left_face, right_face)
        
        # Test mapping
        left_point = jnp.array([0.0, 0.5, 0.5])
        right_point = mapper(left_point)
        
        # Should map to corresponding point on right face
        expected_right = jnp.array([1.0, 0.5, 0.5])
        assert jnp.allclose(right_point, expected_right, atol=1e-3)
    
    def test_mapping_preserves_other_coordinates(self, cube_unit_cell):
        """Test that mapping preserves coordinates in non-mapped directions."""
        left_face = cube_unit_cell.face_function(axis=0, value=0.0)
        right_face = cube_unit_cell.face_function(axis=0, value=1.0)
        mapper = cube_unit_cell.mapping(left_face, right_face)
        
        # Test multiple points
        test_points = [
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array([0.0, 1.0, 0.0]),
            jnp.array([0.0, 0.5, 1.0]),
            jnp.array([0.0, 0.33, 0.67])
        ]
        
        for left_point in test_points:
            right_point = mapper(left_point)
            
            # x-coordinate should change from 0 to 1
            assert jnp.isclose(right_point[0], 1.0, atol=1e-3)
            
            # y and z coordinates should be preserved
            assert jnp.isclose(right_point[1], left_point[1], atol=1e-3)
            assert jnp.isclose(right_point[2], left_point[2], atol=1e-3)
    
    def test_mapping_mismatched_boundaries_error(self, cube_unit_cell):
        """Test that mapping fails when boundaries have different point counts."""
        # Face vs edge (different number of points)
        left_face = cube_unit_cell.face_function(axis=0, value=0.0)
        bottom_edge = cube_unit_cell.edge_function([0, 1], [0.0, 0.0])
        
        with pytest.raises(ValueError, match="same number of points"):
            cube_unit_cell.mapping(left_face, bottom_edge)


class TestMacroDisplacement:
    """Test macroscopic displacement calculations."""
    
    @pytest.fixture
    def cube_unit_cell(self):
        return CubeUnitCell(nx=2, ny=2, nz=2)
    
    @pytest.fixture
    def quad_unit_cell(self):
        return QuadUnitCell(nx=2, ny=2)
    
    def test_basic_macro_displacement_3d(self, cube_unit_cell):
        """Test basic macro displacement for 3D unit cell."""
        # Simple tension in x-direction
        strain = jnp.array([
            [0.01, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        
        displacements = cube_unit_cell.apply_macro_displacement(strain)
        
        # Should have displacements for all nodes in all directions
        num_nodes = cube_unit_cell.points.shape[0]
        expected_length = num_nodes * 3
        assert len(displacements) == expected_length
        
        # Check some specific displacements
        # Point at (1,0,0) should have displacement 0.01 in x
        target_point = jnp.array([1.0, 0.0, 0.0])
        distances = jnp.linalg.norm(cube_unit_cell.points - target_point, axis=1)
        point_idx = jnp.where(distances < 1e-6)[0]
        if len(point_idx) > 0:
            idx = point_idx[0]
            x_displacement = displacements[idx * 3]  # x-component
            assert jnp.isclose(x_displacement, 0.01, atol=1e-6)
    
    def test_macro_displacement_2d(self, quad_unit_cell):
        """Test macro displacement for 2D unit cell."""
        # Simple shear strain
        strain = jnp.array([
            [0.0, 0.02],
            [0.02, 0.0]
        ])
        
        displacements = quad_unit_cell.apply_macro_displacement(strain)
        
        # Should have displacements for all nodes in both directions
        num_nodes = quad_unit_cell.points.shape[0]
        expected_length = num_nodes * 2
        assert len(displacements) == expected_length
    
    def test_macro_displacement_custom_origin(self, cube_unit_cell):
        """Test macro displacement with custom origin."""
        strain = jnp.array([
            [0.01, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        
        # Use center as origin
        origin = jnp.array([0.5, 0.5, 0.5])
        displacements = cube_unit_cell.apply_macro_displacement(strain, origin=origin)
        
        # Point at origin should have zero displacement
        distances = jnp.linalg.norm(cube_unit_cell.points - origin, axis=1)
        origin_idx = jnp.where(distances < 1e-6)[0]
        if len(origin_idx) > 0:
            idx = origin_idx[0]
            x_displacement = displacements[idx * 3]
            assert jnp.isclose(x_displacement, 0.0, atol=1e-6)
    
    def test_macro_displacement_dimension_mismatch(self, cube_unit_cell):
        """Test error handling for dimension mismatch."""
        # 2D strain tensor for 3D unit cell
        wrong_strain = jnp.array([
            [0.01, 0.0],
            [0.0, 0.0]
        ])
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            cube_unit_cell.apply_macro_displacement(wrong_strain)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_mesh_build_failure_handling(self):
        """Test handling of mesh build failures."""
        class FailingUnitCell(UnitCell):
            def mesh_build(self, **kwargs):
                raise RuntimeError("Mesh generation failed")
        
        with pytest.raises(RuntimeError, match="Mesh generation failed"):
            FailingUnitCell()
    
    def test_empty_mesh_handling(self):
        """Test handling of empty or invalid meshes."""
        class EmptyMeshUnitCell(UnitCell):
            def mesh_build(self, **kwargs):
                mesh = Mock()
                mesh.points = jnp.array([]).reshape(0, 3)
                mesh.cells = jnp.array([]).reshape(0, 8)
                mesh.ele_type = "HEX8"
                return mesh
        
        # This should not crash but might produce unusual bounds
        unit_cell = EmptyMeshUnitCell()
        assert unit_cell.points.shape[0] == 0
    
    def test_invalid_axis_in_face_function(self):
        """Test error handling for invalid axis in face function."""
        unit_cell = CubeUnitCell()
        
        # Axis out of range for 3D mesh
        with pytest.raises((IndexError, ValueError)):
            face_func = unit_cell.face_function(axis=5, value=0.0)
            # Try to use the function to trigger the error
            face_func(jnp.array([0.0, 0.0, 0.0]))
    
    def test_invalid_corner_coordinates(self):
        """Test error handling for invalid corner coordinates."""
        unit_cell = CubeUnitCell()
        
        # Wrong number of coordinates
        with pytest.raises((IndexError, ValueError)):
            corner_func = unit_cell.corner_function([0.0, 0.0])  # 2D for 3D mesh
            corner_func(jnp.array([0.0, 0.0, 0.0]))


class TestDocstringExamples:
    """Test that examples in docstrings work correctly."""
    
    def test_class_docstring_example(self):
        """Test the example from the class docstring."""
        class CubeUnitCell(UnitCell):
            def mesh_build(self, **kwargs):
                return box_mesh(10, 10, 10, 1.0, 1.0, 1.0)
        
        unit_cell = CubeUnitCell()
        corners = unit_cell.corners
        assert len(corners) == 8  # 3D cube has 8 corners
    
    def test_face_function_docstring_example(self):
        """Test the example from face_function docstring."""
        unit_cell = CubeUnitCell()
        
        # Create function for left face (x=0), excluding corners
        left_face = unit_cell.face_function(axis=0, value=0.0, excluding_corner=True)
        test_point = jnp.array([0.0, 0.5, 0.5])
        on_face = left_face(test_point)
        
        # Should be on face but we excluded corners, and this point is not a corner
        assert on_face == True
    
    def test_mapping_docstring_example(self):
        """Test the example from mapping docstring."""
        unit_cell = CubeUnitCell()
        
        # Map left face to right face for periodic BC
        left_face = unit_cell.face_function(0, 0.0)   # x = 0
        right_face = unit_cell.face_function(0, 1.0)  # x = 1
        mapper = unit_cell.mapping(left_face, right_face)
        
        # Map a point from left to right
        left_point = jnp.array([0.0, 0.5, 0.5])
        right_point = mapper(left_point)
        
        # Should be [1.0, 0.5, 0.5]
        expected = jnp.array([1.0, 0.5, 0.5])
        assert jnp.allclose(right_point, expected, atol=1e-3)


@pytest.mark.integration
class TestUnitCellIntegration:
    """Integration tests combining multiple UnitCell features."""
    
    def test_periodic_boundary_setup(self):
        """Test setting up periodic boundary conditions."""
        unit_cell = CubeUnitCell(nx=3, ny=3, nz=3)
        
        # Define opposite face pairs for 3D periodicity
        face_pairs = [
            (unit_cell.face_function(0, 0.0), unit_cell.face_function(0, 1.0)),  # x-faces
            (unit_cell.face_function(1, 0.0), unit_cell.face_function(1, 1.0)),  # y-faces
            (unit_cell.face_function(2, 0.0), unit_cell.face_function(2, 1.0)),  # z-faces
        ]
        
        mappers = []
        for master_face, slave_face in face_pairs:
            mapper = unit_cell.mapping(master_face, slave_face)
            mappers.append(mapper)
        
        # Test that each mapper works
        assert len(mappers) == 3
        
        # Test a specific mapping
        x_mapper = mappers[0]
        test_point = jnp.array([0.0, 0.33, 0.67])
        mapped_point = x_mapper(test_point)
        expected = jnp.array([1.0, 0.33, 0.67])
        assert jnp.allclose(mapped_point, expected, atol=1e-3)
    
    def test_multiscale_analysis_setup(self):
        """Test setup for multiscale analysis."""
        unit_cell = CubeUnitCell(nx=4, ny=4, nz=4)
        
        # Apply macroscopic strain
        macro_strain = jnp.array([
            [0.001, 0.0005, 0.0],
            [0.0005, 0.0, 0.0],
            [0.0, 0.0, -0.0002]  # Poisson effect
        ])
        
        displacements = unit_cell.apply_macro_displacement(macro_strain)
        
        # Verify displacement field makes sense
        assert len(displacements) == unit_cell.points.shape[0] * 3
        
        # Check corner displacements
        corners = unit_cell.corners
        for i, corner in enumerate(corners):
            corner_disp = unit_cell.apply_macro_displacement(macro_strain, origin=jnp.zeros(3))
            
            # Find this corner in the mesh
            distances = jnp.linalg.norm(unit_cell.points - corner, axis=1)
            corner_idx = jnp.where(distances < 1e-6)[0]
            if len(corner_idx) > 0:
                idx = corner_idx[0]
                mesh_disp = displacements[idx*3:(idx+1)*3]
                expected_disp = (corner @ macro_strain.T).flatten()
                assert jnp.allclose(mesh_disp, expected_disp, atol=1e-6)
    
    def test_boundary_node_classification_completeness(self):
        """Test that boundary node classification is complete and consistent."""
        unit_cell = CubeUnitCell(nx=2, ny=2, nz=2)
        
        # Get all boundary masks
        corner_mask = unit_cell.corner_mask
        edge_mask = unit_cell.edge_mask
        face_mask = unit_cell.face_mask
        
        # Create boundary mask manually
        boundary_mask = jnp.zeros(unit_cell.points.shape[0], dtype=bool)
        
        for i, point in enumerate(unit_cell.points):
            # Point is on boundary if any coordinate is at min or max
            on_boundary = jnp.any(
                jnp.logical_or(
                    jnp.isclose(point, unit_cell.lb, atol=unit_cell.atol),
                    jnp.isclose(point, unit_cell.ub, atol=unit_cell.atol)
                )
            )
            boundary_mask = boundary_mask.at[i].set(on_boundary)
        
        # All boundary points should be classified as corner, edge, or face
        classified_mask = corner_mask | edge_mask | face_mask
        
        # Check that classification covers all boundary points
        unclassified_boundary = boundary_mask & ~classified_mask
        assert jnp.sum(unclassified_boundary) == 0, "Some boundary points are unclassified"


if __name__ == "__main__":
    """Run tests directly for development."""
    pytest.main([__file__, "-v"])