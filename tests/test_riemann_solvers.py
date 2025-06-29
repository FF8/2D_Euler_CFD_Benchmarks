"""
Unit tests for the Riemann solver functions in riemann_solvers.py.
These tests verify the flux conservation property and check for basic
functionality using the Sod shock tube problem.

To run these tests, go to your project's root directory and run pytest
"""

import sys
import os
import pytest
import numpy as np

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from riemann_solvers import hll_flux, hllc_flux
from utils import primitive_to_conserved_general, get_flux_general
from config import GAMMA

@pytest.fixture
def sod_shock_tube_states():
    """
    Provides the left and right states for the 1D Sod shock tube problem.
    The states are returned as conserved variable vectors.
    """
    # Primitive Left State: [rho, u, v, p]
    P_L = np.array([1.0, 0.0, 0.0, 1.0])
    U_L = primitive_to_conserved_general(P_L)

    # Primitive Right State: [rho, u, v, p]
    P_R = np.array([0.125, 0.0, 0.0, 0.1])
    U_R = primitive_to_conserved_general(P_R)
    
    return U_L, U_R

@pytest.fixture
def uniform_flow_state():
    """
    Provides a single state for a uniform supersonic flow. This is used
    to test the flux conservation property of the solvers.
    """
    # Primitive state: [rho, u, v, p]
    P = np.array([1.4, 3.0, 0.5, 1.0])
    U = primitive_to_conserved_general(P)
    return U

def test_flux_conservation(uniform_flow_state):
    """
    Tests the fundamental property that for a uniform flow (U_L = U_R), 
    the numerical flux must equal the analytical flux F(U).
    """
    U = uniform_flow_state
    
    # Expected analytical flux in the x-direction
    expected_F, _ = get_flux_general(U)
    
    # Test HLL solver
    hll_f = hll_flux(U, U, direction='x')
    assert np.allclose(hll_f, expected_F), "HLL solver failed x-direction flux conservation"
    
    # Test HLLC solver (with both wave speed estimates)
    hllc_f_roe = hllc_flux(U, U, direction='x', wave_speed_method='roe')
    assert np.allclose(hllc_f_roe, expected_F), "HLLC (Roe) solver failed x-direction flux conservation"
    
    hllc_f_davis = hllc_flux(U, U, direction='x', wave_speed_method='davis')
    assert np.allclose(hllc_f_davis, expected_F), "HLLC (Davis) solver failed x-direction flux conservation"

def test_y_direction_flux_conservation(uniform_flow_state):
    """
    Tests the flux conservation property in the y-direction. For a uniform flow,
    the numerical flux G_hat must equal the analytical flux G(U).
    """
    U = uniform_flow_state
    
    # Expected analytical flux in the y-direction
    _, expected_G = get_flux_general(U)
    
    # Test HLL solver
    hll_g = hll_flux(U, U, direction='y')
    assert np.allclose(hll_g, expected_G), "HLL solver failed y-direction flux conservation"

    # Test HLLC solver
    hllc_g = hllc_flux(U, U, direction='y')
    assert np.allclose(hllc_g, expected_G), "HLLC solver failed y-direction flux conservation"

def test_solvers_with_sod_problem(sod_shock_tube_states):
    """
    Runs the solvers with the Sod shock tube initial conditions to check for
    basic functionality (correct output shape, no NaNs).
    """
    U_L, U_R = sod_shock_tube_states
    
    # Add a dimension to simulate a single interface
    U_L_col = U_L[:, np.newaxis]
    U_R_col = U_R[:, np.newaxis]
    
    # Test HLL
    flux_hll = hll_flux(U_L_col, U_R_col)
    assert flux_hll.shape == (4, 1), f"HLL flux has incorrect shape: {flux_hll.shape}"
    assert not np.isnan(flux_hll).any(), "HLL flux produced NaNs"

    # Test HLLC
    flux_hllc = hllc_flux(U_L_col, U_R_col)
    assert flux_hllc.shape == (4, 1), f"HLLC flux has incorrect shape: {flux_hllc.shape}"
    assert not np.isnan(flux_hllc).any(), "HLLC flux produced NaNs"

