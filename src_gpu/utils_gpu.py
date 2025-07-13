# src/utils_gpu.py
"""
GPU-accelerated utility functions for converting between primitive and conserved
variables, and for calculating physical flux vectors.
"""
import cupy as np
from config import GAMMA

def cons_to_prim_general(U):
    """
    Converts a conserved state vector/array U to primitive variables [rho, u, v, p].
    Works on any array shape with state vectors along the first axis.
    """
    rho = U[0]
    # Add a small epsilon to avoid division by zero for vacuum states
    epsilon = 1e-12
    u = U[1] / (rho + epsilon)
    v = U[2] / (rho + epsilon)
    # Ensure pressure is non-negative
    p = np.maximum((GAMMA - 1) * (U[3] - 0.5 * rho * (u**2 + v**2)), 1e-9)
    return rho, u, v, p

def primitive_to_conserved_general(P):
    """
    Converts a primitive state vector/array P = [rho, u, v, p] to conserved variables.
    """
    rho, u, v, p = P[0], P[1], P[2], P[3]
    E = p / (GAMMA - 1) + 0.5 * rho * (u**2 + v**2)
    
    # Create an empty array with the same shape and type as the input, then fill it.
    # This is a more robust way to build the array on the GPU.
    U = np.empty_like(P)
    U[0] = rho
    U[1] = rho * u
    U[2] = rho * v
    U[3] = E
    return U

def get_flux_general(U):
    """
    Calculates the flux vectors F (for x-direction) and G (for y-direction)
    from the conserved state vector U.
    """
    rho, u, v, p = cons_to_prim_general(U)
    
    F = np.zeros_like(U)
    G = np.zeros_like(U)
    
    # x-direction flux (F)
    F[0] = rho * u
    F[1] = rho * u**2 + p
    F[2] = rho * u * v
    F[3] = (U[3] + p) * u
    
    # y-direction flux (G)
    G[0] = rho * v
    G[1] = rho * u * v
    G[2] = rho * v**2 + p
    G[3] = (U[3] + p) * v
    
    return F, G