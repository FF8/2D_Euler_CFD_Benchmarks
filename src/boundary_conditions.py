"""
This module handles the application of boundary conditions to the padded
computational grid. Each problem type has specific boundary requirements.
"""
import numpy as np
from config import GAMMA
from utils import cons_to_prim_general

def apply_boundary_conditions(P_padded, bc_params, t):
    """
    Applies boundary conditions to the ghost cells of the padded primitive variable array.
    
    Args:
        P_padded (np.array): The array of primitive variables including ghost cells.
        bc_params (dict): A dictionary of parameters for the specific problem.
        t (float): The current simulation time.
    
    Returns:
        np.array: The padded primitive variable array with updated ghost cells.
    """
    problem_type = bc_params['problem_type']
    N_GHOST = bc_params.get('n_ghost', 1)
    
    if problem_type == 'dmr':
        # --- Double Mach Reflection Boundaries ---
        nx, x_max, y_max = bc_params['nx'], bc_params['x_max'], bc_params['y_max']
        U_post_state, U_pre_state = bc_params['U_post_state'], bc_params['U_pre_state']
        v_shock_horizontal = bc_params['v_shock_horizontal']
        dx = x_max / nx
        
        P_post_state = np.array(cons_to_prim_general(U_post_state[:, np.newaxis]))
        P_pre_state = np.array(cons_to_prim_general(U_pre_state[:, np.newaxis]))
        
        # Left boundary (post-shock inflow)
        for i in range(N_GHOST): P_padded[:, :, i] = P_post_state
        
        # Right boundary (extrapolation)
        for i in range(N_GHOST): P_padded[:, :, -1 - i] = P_padded[:, :, -1 - N_GHOST]
        
        # Top boundary (time-dependent inflow)
        shock_angle = 60 * np.pi / 180
        x_centers = np.linspace(dx / 2, x_max - dx / 2, nx)
        x_shock_top = 1 / 6 + y_max / np.tan(shock_angle) + v_shock_horizontal * t
        top_mask = x_centers < x_shock_top
        P_ghost_top = np.where(top_mask[np.newaxis, :], P_post_state, P_pre_state)
        for i in range(N_GHOST): P_padded[:, -1 - i, N_GHOST:-N_GHOST] = P_ghost_top
        
        # Bottom boundary (reflective wall after wedge)
        wedge_start_idx = int((1.0 / 6.0) / dx)
        for i in range(N_GHOST):
            ghost_j, phys_j = N_GHOST - 1 - i, N_GHOST + i
            # Pre-wedge inflow
            P_padded[:, ghost_j, N_GHOST:N_GHOST + wedge_start_idx] = P_post_state
            # Reflective wall (FIXED: Slice now extends to the end of the padded array)
            P_padded[:, ghost_j, N_GHOST + wedge_start_idx:] = P_padded[:, phys_j, N_GHOST + wedge_start_idx:]
            P_padded[2, ghost_j, N_GHOST + wedge_start_idx:] *= -1.0 # Invert v

    elif problem_type == '2d_riemann':
        # --- 2D Riemann Boundaries (Transmissive/Outflow) ---
        for i in range(N_GHOST):
            P_padded[:, :, i] = P_padded[:, :, N_GHOST]
            P_padded[:, :, -1 - i] = P_padded[:, :, -1 - N_GHOST]
            P_padded[:, i, :] = P_padded[:, N_GHOST, :]
            P_padded[:, -1 - i, :] = P_padded[:, -1 - N_GHOST, :]

    elif problem_type == '2d_sod':
        # --- 2D Sod Tube Boundaries ---
        for i in range(N_GHOST):
            # Left/Right (Transmissive/Outflow)
            P_padded[:, :, i] = P_padded[:, :, N_GHOST]
            P_padded[:, :, -1-i] = P_padded[:, :, -1-N_GHOST]
            
            # Top/Bottom (Reflective)
            P_padded[:, -1-i, :] = P_padded[:, -1-N_GHOST, :] # Top
            P_padded[2, -1-i, :] *= -1.0 # Invert v
            P_padded[:, i, :] = P_padded[:, N_GHOST, :]     # Bottom
            P_padded[2, i, :] *= -1.0 # Invert v
            
    return P_padded
