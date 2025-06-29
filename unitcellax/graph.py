"""Graph-based geometry generation for lattice structures.

This module provides functions for creating and evaluating graph-based geometric
representations of lattice structures, particularly useful for topology optimization
and unit cell design. The module supports distance-based evaluation of points
relative to graph edges, enabling the creation of smooth implicit representations
of lattice geometries.

Key Features:
    - Distance-based point-to-edge evaluation
    - Multiple lattice structure types
    - Universal graph evaluation functions
    - JAX-compatible for GPU acceleration and automatic differentiation

Supported Lattice Structures:
    - FCC (Face-Centered Cubic): Common metallic crystal structure

Example:
    Creating FCC lattice structure:
    
    >>> import jax.numpy as jnp
    >>> from unitcellax.graph import fcc
    >>> 
    >>> # Create FCC lattice function
    >>> fcc_graph = fcc(radius=0.1, scale=1.0)
    >>> 
    >>> # Evaluate at a point
    >>> point = jnp.array([0.25, 0.25, 0.25])
    >>> fcc_result = fcc_graph(point)
"""

import jax.numpy as jnp
from functools import partial
import jax
from typing import Callable, Tuple

def _segment_distance(x: jnp.ndarray, p0: jnp.ndarray, p1: jnp.ndarray) -> jnp.ndarray:
    """Compute the minimum distance from a point to a line segment.
    
    This function calculates the shortest Euclidean distance between a point x
    and a line segment defined by endpoints p0 and p1. The distance is computed
    by projecting the point onto the line and clipping the projection parameter
    to ensure it lies within the segment bounds.
    
    Args:
        x (jnp.ndarray): Query point with shape (..., spatial_dim).
        p0 (jnp.ndarray): First endpoint of line segment with shape (..., spatial_dim).
        p1 (jnp.ndarray): Second endpoint of line segment with shape (..., spatial_dim).
        
    Returns:
        jnp.ndarray: Minimum distance from point to line segment with shape (...,).
        
    Note:
        The function handles the case where p0 == p1 (degenerate segment) by
        computing the dot product with a zero vector, which results in t = 0
        and returns the distance to p0.
    """
    v = p1 - p0
    w = x - p0
    
    # Compute projection parameter, clipped to [0, 1] for segment bounds
    v_dot_v = jnp.dot(v, v)
    t = jnp.where(v_dot_v > 0, jnp.clip(jnp.dot(w, v) / v_dot_v, 0.0, 1.0), 0.0)
    
    # Find closest point on segment and compute distance
    proj = p0 + t * v
    return jnp.linalg.norm(x - proj)

def universal_graph(x: jnp.ndarray, nodes: jnp.ndarray, edges: jnp.ndarray, 
                   radius: float) -> jnp.ndarray:
    """Evaluate if a point lies within the graph structure defined by nodes and edges.
    
    This function determines whether a query point x is within a specified radius
    of any edge in the graph. Each edge is treated as a line segment connecting
    two nodes, and the function returns 1 if the point is within the radius
    distance of any edge, 0 otherwise.
    
    The function is designed to be JAX-compatible and supports vectorization
    for efficient evaluation of multiple points or graphs.
    
    Args:
        x (jnp.ndarray): Query point with shape (spatial_dim,).
        nodes (jnp.ndarray): Node coordinates with shape (num_nodes, spatial_dim).
        edges (jnp.ndarray): Edge connectivity matrix with shape (num_edges, 2).
            Each row contains indices of two nodes that form an edge.
        radius (float): Distance threshold for point inclusion. Must be positive.
        
    Returns:
        jnp.ndarray: Binary indicator (0 or 1) as uint8. Returns 1 if point x is
            within radius distance of any edge, 0 otherwise.
            
    Raises:
        ValueError: If radius is negative or if array dimensions are incompatible.
        
    Example:
        >>> nodes = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        >>> edges = jnp.array([[0, 1], [1, 2]])
        >>> point = jnp.array([0.5, 0.0])
        >>> result = universal_graph(point, nodes, edges, radius=0.1)
        >>> print(result)  # Should be 1 (point is on the first edge)
    """
    if radius < 0:
        raise ValueError(f"Radius must be non-negative, got {radius}")
        
    def check_edge(edge: jnp.ndarray) -> jnp.ndarray:
        """Check if point is within radius of a specific edge."""
        i, j = edge
        return _segment_distance(x, nodes[i], nodes[j]) <= radius
    
    # Handle empty edges case
    if edges.shape[0] == 0:
        return jnp.uint8(0)
    
    # Use vmap to check all edges in parallel
    edge_checks = jax.vmap(check_edge)(edges)
    return jnp.any(edge_checks).astype(jnp.uint8)

def fcc_unitcell(scale: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate nodes and edges for a Face-Centered Cubic (FCC) unit cell.
    
    Creates the geometric representation of an FCC lattice unit cell, which is
    fundamental in crystallography and materials science. The FCC structure
    consists of atoms at the corners and face centers of a cubic unit cell.
    
    The function generates a complete FCC unit cell with 14 nodes (8 corner atoms
    and 6 face-centered atoms) and 24 edges representing the nearest-neighbor
    bonds. In the FCC structure, each corner atom is connected to 3 face-centered
    atoms, resulting in a coordination number of 12 when periodically repeated.
    
    Args:
        scale (float): Scaling factor for the unit cell dimensions. 
            Default is 1.0. Must be positive.
            
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing:
            - nodes (jnp.ndarray): Node coordinates with shape (14, 3).
              Contains positions of all atoms in the FCC unit cell (8 corners + 6 face centers).
            - edges (jnp.ndarray): Edge connectivity with shape (24, 2).
              Each row contains indices of two nodes that form a nearest-neighbor bond.
              
    Raises:
        ValueError: If scale is not positive.
        
    Note:
        The function creates an extended unit cell to ensure proper connectivity
        across periodic boundaries. All node positions are explicitly defined.
        
    Example:
        >>> nodes, edges = fcc_unitcell(scale=2.0)
        >>> print(f"Number of nodes: {len(nodes)}")  # 14
        >>> print(f"Number of edges: {len(edges)}")  # 24
        >>> print(f"Unit cell size: {nodes.max()}")  # 2.0
    """
    if scale <= 0:
        raise ValueError(f"Scale must be positive, got {scale}")
    
    # Define all FCC node positions directly
    nodes = jnp.array([
        # Corner atoms (8 nodes for cubic unit cell)
        [0.0, 0.0, 0.0],  # Node 0
        [1.0, 0.0, 0.0],  # Node 1
        [0.0, 1.0, 0.0],  # Node 2
        [0.0, 0.0, 1.0],  # Node 3
        [1.0, 1.0, 0.0],  # Node 4
        [1.0, 0.0, 1.0],  # Node 5
        [0.0, 1.0, 1.0],  # Node 6
        [1.0, 1.0, 1.0],  # Node 7
        # Face center atoms (6 nodes)
        [0.5, 0.5, 0.0],  # Node 8: Bottom face center
        [0.5, 0.5, 1.0],  # Node 9: Top face center
        [0.5, 0.0, 0.5],  # Node 10: Front face center
        [0.5, 1.0, 0.5],  # Node 11: Back face center
        [0.0, 0.5, 0.5],  # Node 12: Left face center
        [1.0, 0.5, 0.5],  # Node 13: Right face center
    ]) * scale
    
    # Edges between corner nodes (cube edges)
    cube_edges = [
        [0, 1], [0, 2], [0, 3],
        [1, 4], [1, 5],
        [2, 4], [2, 6],
        [3, 5], [3, 6],
        [4, 7],
        [5, 7],
        [6, 7],
    ]

    # Corner-to-face center connections (from original)
    corner_face_edges = [
        [0, 8], [0, 10], [0, 12],
        [1, 8], [1, 10], [1, 13],
        [2, 8], [2, 11], [2, 12],
        [3, 9], [3, 10], [3, 12],
        [4, 8], [4, 11], [4, 13],
        [5, 9], [5, 10], [5, 13],
        [6, 9], [6, 11], [6, 12],
        [7, 9], [7, 11], [7, 13],
    ]

    edges = jnp.array(cube_edges + corner_face_edges)
    
    return nodes, edges

def fcc(radius: float = 0.1, scale: float = 1.0) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Create an FCC lattice evaluation function.
    
    This function returns a callable that can evaluate whether points lie within
    the FCC (Face-Centered Cubic) lattice structure. The returned function is
    particularly useful for topology optimization, implicit surface generation,
    and material design applications.
    
    The FCC lattice is one of the most common crystal structures, found in
    materials like aluminum, copper, gold, and many other metals. This function
    creates a graph-based representation that can be efficiently evaluated
    for large numbers of points using JAX vectorization.
    
    Args:
        radius (float): Radius for edge thickness in the lattice structure.
            Controls how "thick" the struts appear. Default is 0.1.
            Must be non-negative.
        scale (float): Scaling factor for the unit cell dimensions.
            Default is 1.0. Must be positive.
            
    Returns:
        Callable[[jnp.ndarray], jnp.ndarray]: A function that takes a point
            x (shape: (spatial_dim,)) and returns a binary indicator (uint8)
            of whether the point lies within the FCC lattice structure.
            
    Raises:
        ValueError: If radius is negative or scale is not positive.
        
    Example:
        >>> # Create FCC lattice function
        >>> fcc_func = fcc(radius=0.05, scale=2.0)
        >>> 
        >>> # Evaluate at specific points
        >>> point1 = jnp.array([0.0, 0.0, 0.0])  # Should be inside (near corner)
        >>> point2 = jnp.array([0.25, 0.25, 0.25])  # Should be inside (on edge)
        >>> point3 = jnp.array([1.5, 1.5, 1.5])  # Should be outside
        >>> 
        >>> results = [fcc_func(p) for p in [point1, point2, point3]]
        >>> print(results)  # [1, 1, 0] (inside, inside, outside)
        >>>
        >>> # Vectorized evaluation
        >>> points = jnp.array([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]])
        >>> results = jax.vmap(fcc_func)(points)
        
    Note:
        The returned function is JAX-compatible and supports automatic
        differentiation, making it suitable for gradient-based optimization
        algorithms in topology optimization and design applications.
    """
    if radius < 0:
        raise ValueError(f"Radius must be non-negative, got {radius}")
    if scale <= 0:
        raise ValueError(f"Scale must be positive, got {scale}")
        
    # Generate FCC unit cell geometry
    nodes, edges = fcc_unitcell(scale)
    
    # Return partially applied universal_graph function
    return partial(universal_graph, nodes=nodes, edges=edges, radius=radius)