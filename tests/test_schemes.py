# tests/test_schemes.py
"""
Unit tests for the numerical schemes in schemes.py, focusing on calculate_rhs.
"""
import sys
import os
import pytest
import numpy as np

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from schemes import calculate_rhs
from utils import primitive_to_conserved_general

@pytest.fixture
def uniform_flow_setup():
    """
    Creates a simple grid and a uniform flow state to test the RHS calculation.
    """
    # Grid parameters
    nx, ny = 10, 8
    dx, dy = 0.1, 0.1
    
    # Uniform primitive state [rho, u, v, p]
    P_uniform = np.array([1.2, 25.0, -10.0, 101325.0])
    
    # Create the conserved state array U for the entire grid
    U = np.zeros((4, ny, nx))
    U[:] = primitive_to_conserved_general(P_uniform)[:, np.newaxis, np.newaxis]
    
    # Basic parameters dictionary
    bc_params = {
        'problem_type': '2d_riemann', # Using transmissive boundaries for simplicity
        'riemann_solver': 'hllc',
        'hllc_wave_speed': 'roe',
        'nx': nx, 'ny': ny
    }
    
    return U, dx, dy, bc_params

@pytest.mark.parametrize("scheme", ["piecewise_constant", "piecewise_linear"])
def test_calculate_rhs_on_uniform_flow(uniform_flow_setup, scheme):
    """
    Tests that calculate_rhs returns a zero (or near-zero) RHS for a uniform flow.
    This is a fundamental consistency check. The derivative of a constant flow
    field should be zero.
    """
    U, dx, dy, bc_params = uniform_flow_setup
    bc_params['reconstruction_scheme'] = scheme
    
    # Current time (t=0) is arbitrary for this test
    t = 0.0
    
    # Calculate the Right-Hand Side
    rhs = calculate_rhs(U, dx, dy, t, bc_params)
    
    # The RHS should be an array of zeros. We use np.allclose to account
    # for potential floating-point inaccuracies.
    assert np.allclose(rhs, 0.0), f"RHS for uniform flow is non-zero with {scheme} scheme"