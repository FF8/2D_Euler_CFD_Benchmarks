# tests/test_boundary_conditions.py
"""
Unit tests for the boundary condition functions in boundary_conditions.py.
These tests verify that ghost cells are populated correctly for each
of the major problem setups.
"""
import sys
import os
import pytest
import numpy as np

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from boundary_conditions import apply_boundary_conditions
from utils import primitive_to_conserved_general

# Define a simple physical state to fill the interior of the domain for testing
P_INTERIOR = np.array([1.0, 2.0, 3.0, 4.0]) # rho, u, v, p

@pytest.fixture
def setup_padded_grid():
    """
    Creates a simple padded grid with a uniform interior state.
    The ghost cells are initialized with a different, recognizable value (-1).
    """
    ny, nx = 5, 6  # A small physical domain
    n_ghost = 2    # Use 2 ghost cells to test compatibility with both schemes
    
    # Padded dimensions
    ny_pad, nx_pad = ny + 2 * n_ghost, nx + 2 * n_ghost
    
    # Create a padded array initialized with dummy values
    P_padded = np.full((4, ny_pad, nx_pad), -1.0)
    
    # Fill the physical interior domain with the known state
    P_padded[:, n_ghost:-n_ghost, n_ghost:-n_ghost] = P_INTERIOR[:, np.newaxis, np.newaxis]
    
    # Basic bc_params dictionary to be updated by each test
    params = {
        'n_ghost': n_ghost,
        'nx': nx,
        'ny': ny
    }
    
    return P_padded, params

def test_sod_boundaries(setup_padded_grid):
    """
    Tests the transmissive and reflective boundaries for the '2d_sod' case.
    """
    P_padded, params = setup_padded_grid
    params['problem_type'] = '2d_sod'
    
    # Apply the boundary conditions
    P_result = apply_boundary_conditions(P_padded.copy(), params, t=0)
    
    n_ghost = params['n_ghost']
    
    # --- Test Left/Right Transmissive (Outflow) BCs ---
    first_phys_col = P_result[:, n_ghost:-n_ghost, n_ghost]
    last_phys_col = P_result[:, n_ghost:-n_ghost, -n_ghost-1]
    for i in range(n_ghost):
        # Left ghost cells should match the first physical column
        assert np.array_equal(P_result[:, n_ghost:-n_ghost, i], first_phys_col)
        # Right ghost cells should match the last physical column
        assert np.array_equal(P_result[:, n_ghost:-n_ghost, -1-i], last_phys_col)

    # --- Test Top/Bottom Reflective BCs ---
    for i in range(n_ghost):
        # Bottom boundary
        phys_row_b = P_result[:, n_ghost+i, n_ghost:-n_ghost]
        ghost_row_b = P_result[:, n_ghost-1-i, n_ghost:-n_ghost]
        assert np.allclose(ghost_row_b[2], -phys_row_b[2]) # v = -v
        assert np.allclose(ghost_row_b[[0,1,3]], phys_row_b[[0,1,3]]) # rho, u, p are equal

        # Top boundary
        phys_row_t = P_result[:, -n_ghost-1-i, n_ghost:-n_ghost]
        ghost_row_t = P_result[:, -n_ghost+i, n_ghost:-n_ghost]
        assert np.allclose(ghost_row_t[2], -phys_row_t[2]) # v = -v
        assert np.allclose(ghost_row_t[[0,1,3]], phys_row_t[[0,1,3]]) # rho, u, p are equal


def test_riemann_boundaries(setup_padded_grid):
    """
    Tests the purely transmissive (outflow) boundaries for the '2d_riemann' case.
    """
    P_padded, params = setup_padded_grid
    params['problem_type'] = '2d_riemann'

    # Apply boundary conditions
    P_result = apply_boundary_conditions(P_padded.copy(), params, t=0)
    n_ghost = params['n_ghost']

    # --- Test all boundaries are transmissive ---
    first_phys_col = P_result[:, n_ghost:-n_ghost, n_ghost]
    last_phys_col = P_result[:, n_ghost:-n_ghost, -n_ghost-1]
    first_phys_row = P_result[:, n_ghost, :]
    last_phys_row = P_result[:, -n_ghost-1, :]
    
    for i in range(n_ghost):
        assert np.array_equal(P_result[:, n_ghost:-n_ghost, i], first_phys_col)
        assert np.array_equal(P_result[:, n_ghost:-n_ghost, -1-i], last_phys_col)
        assert np.array_equal(P_result[:, i, :], first_phys_row)
        assert np.array_equal(P_result[:, -1-i, :], last_phys_row)


def test_dmr_boundaries(setup_padded_grid):
    """
    Tests the complex mixed boundaries for the 'dmr' (Double Mach Reflection) case.
    """
    P_padded, params = setup_padded_grid
    params['problem_type'] = 'dmr'
    
    # Add DMR specific params required by the function
    params['x_max'] = 1.0; params['y_max'] = 1.0
    params['v_shock_horizontal'] = 0.0 # Simplify by testing at t=0
    
    P_post = np.array([8.0, 7.1, -4.1, 116.5])
    params['U_post_state'] = primitive_to_conserved_general(P_post)
    params['U_pre_state']  = primitive_to_conserved_general(np.array([1.4, 0.0, 0.0, 1.0]))

    # Apply BCs
    P_result = apply_boundary_conditions(P_padded.copy(), params, t=0)
    n_ghost = params['n_ghost']

    # Test Left Boundary (should be fixed post-shock inflow)
    for i in range(n_ghost):
        assert np.allclose(P_result[:, :, i], P_post[:, np.newaxis])
        
    # Test Right Boundary (should be simple extrapolation)
    last_phys_col = P_result[:, n_ghost:-n_ghost, -n_ghost-1]
    for i in range(n_ghost):
        assert np.array_equal(P_result[:, n_ghost:-n_ghost, -1-i], last_phys_col)
        
    # Test Bottom Boundary (check reflective part after the wedge)
    wedge_start_idx = int((1.0 / 6.0) / (params['x_max']/params['nx']))
    if wedge_start_idx < params['nx']: # Only test if the reflective part exists
        phys_row = P_result[:, n_ghost, n_ghost+wedge_start_idx:]
        ghost_row = P_result[:, n_ghost-1, n_ghost+wedge_start_idx:]
        assert np.allclose(ghost_row[2], -phys_row[2]) # v = -v
        assert np.allclose(ghost_row[[0,1,3]], phys_row[[0,1,3]])