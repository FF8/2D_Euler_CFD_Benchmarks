"""
Unit tests for the utility functions in utils.py.
These tests use pytest to verify the correctness of the variable conversion
and flux calculation functions.

Go to project directory in the terminal and run pytest
"""

import sys
import os
import pytest
import numpy as np

# Add the project root directory to the Python path to allow imports from other modules
# like 'utils' and 'config'. This is necessary because this test file is in a subdirectory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from utils import cons_to_prim_general, primitive_to_conserved_general, get_flux_general
from config import GAMMA

@pytest.fixture
def fluid_state():
    """
    Provides a sample known fluid state for testing.
    This fixture returns both the primitive and corresponding conserved variables.
    """
    # Define a simple, known primitive state: [rho, u, v, p]
    primitive_vars = np.array([1.0, 2.0, 3.0, 1.5]) # [rho, u, v, p]
    
    # Manually calculate the corresponding conserved variables
    rho, u, v, p = primitive_vars
    E = p / (GAMMA - 1) + 0.5 * rho * (u**2 + v**2)
    conserved_vars = np.array([rho, rho * u, rho * v, E])
    
    return primitive_vars, conserved_vars

def test_conversion_consistency(fluid_state):
    """
    Tests that converting from primitive to conserved and back again
    yields the original primitive variables.
    """
    p_vars, c_vars = fluid_state
    
    # Test primitive -> conserved -> primitive
    calculated_c_vars = primitive_to_conserved_general(p_vars)
    reverted_p_vars = np.array(cons_to_prim_general(calculated_c_vars))
    
    # Assert that the reverted variables are close to the original ones
    assert np.allclose(reverted_p_vars, p_vars), "P -> C -> P conversion failed"

    # Test conserved -> primitive -> conserved
    calculated_p_vars = np.array(cons_to_prim_general(c_vars))
    reverted_c_vars = primitive_to_conserved_general(calculated_p_vars)
    
    assert np.allclose(reverted_c_vars, c_vars), "C -> P -> C conversion failed"

def test_get_flux_general(fluid_state):
    """
    Tests the get_flux_general function against manually calculated flux vectors.
    """
    _, c_vars = fluid_state
    rho, u, v, p = cons_to_prim_general(c_vars)
    
    # Manually calculate the expected fluxes
    E = c_vars[3]
    expected_F = np.array([
        rho * u,
        rho * u**2 + p,
        rho * u * v,
        (E + p) * u
    ])
    
    expected_G = np.array([
        rho * v,
        rho * u * v,
        rho * v**2 + p,
        (E + p) * v
    ])
    
    # Calculate fluxes using the function
    calculated_F, calculated_G = get_flux_general(c_vars)
    
    # Assert that the calculated fluxes match the expected ones
    assert np.allclose(calculated_F, expected_F), "Flux F calculation is incorrect"
    assert np.allclose(calculated_G, expected_G), "Flux G calculation is incorrect"

def test_cons_to_prim_vacuum():
    """
    Tests the cons_to_prim_general function in a vacuum (zero density) case
    to ensure it doesn't crash due to division by zero.
    """
    U_vacuum = np.array([0.0, 0.0, 0.0, 0.0])
    rho, u, v, p = cons_to_prim_general(U_vacuum)
    
    assert rho == 0.0
    assert u == 0.0
    assert v == 0.0
    assert p >= 0.0 # Pressure should be non-negative

