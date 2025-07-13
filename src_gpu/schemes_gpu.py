# schemes.py
"""
This module contains the implementation of the numerical schemes, including
spatial reconstruction and the calculation of the RHS of the semi-discretized
Euler equations.
"""
import cupy as np
from riemann_solvers_gpu  import hllc_flux, hll_flux
from utils_gpu  import primitive_to_conserved_general, cons_to_prim_general
from boundary_conditions_gpu  import apply_boundary_conditions

def minmod_limiter(a, b):
    """Minmod slope limiter."""
    return np.where(a * b <= 0, 0.0, np.where(np.abs(a) < np.abs(b), a, b))

def calculate_rhs(U, dx, dy, t, bc_params):
    """Calculates the right-hand side of the Euler equations for time integration."""
    
    reconstruction_scheme = bc_params['reconstruction_scheme']
    riemann_solver = bc_params['riemann_solver']
    hllc_wave_speed = bc_params['hllc_wave_speed']

    N_GHOST = 2 if reconstruction_scheme == 'piecewise_linear' else 1
    bc_params['n_ghost'] = N_GHOST
    
    # Pad the conserved variable array
    U_padded = np.pad(U, ((0,0), (N_GHOST,N_GHOST), (N_GHOST,N_GHOST)), 'edge')
    
    # Convert to primitive and apply boundary conditions
    P_padded = np.array(cons_to_prim_general(U_padded))
    P_padded = apply_boundary_conditions(P_padded, bc_params, t)

    # --- Spatial Reconstruction ---
    if reconstruction_scheme == 'piecewise_constant':
        U_padded = primitive_to_conserved_general(P_padded)
        U_L_x = U_padded[:, N_GHOST:-N_GHOST, N_GHOST-1:-1]
        U_R_x = U_padded[:, N_GHOST:-N_GHOST, N_GHOST:]
        U_B_y = U_padded[:, N_GHOST-1:-1, N_GHOST:-N_GHOST]
        U_T_y = U_padded[:, N_GHOST:, N_GHOST:-N_GHOST]
        
    elif reconstruction_scheme == 'piecewise_linear':
        # --- MUSCL-Hancock Reconstruction ---
        # 1. Calculate slopes at cell centers (including one layer of ghost cells)
        P_center = P_padded[:, 1:-1, 1:-1]
        delta_L_x = P_center - P_padded[:, 1:-1, :-2]
        delta_R_x = P_padded[:, 1:-1, 2:] - P_center
        slopes_x = minmod_limiter(delta_L_x, delta_R_x)
        
        delta_B_y = P_center - P_padded[:, :-2, 1:-1]
        delta_T_y = P_padded[:, 2:, 1:-1] - P_center
        slopes_y = minmod_limiter(delta_B_y, delta_T_y)

        # --- X-Direction States ---
        # Reconstruct on the physical domain + one layer of ghost cells in the y-dir
        P_center_x = P_padded[:, 1:-1, 1:-1]
        slopes_x_sliced = slopes_x
        
        P_at_right_faces = P_center_x + 0.5 * slopes_x_sliced
        P_at_left_faces  = P_center_x - 0.5 * slopes_x_sliced

        # Assemble states for vertical interfaces
        P_L_x = P_at_right_faces[:, 1:-1, :-1] 
        P_R_x = P_at_left_faces[:, 1:-1, 1:]  
        
        # --- Y-Direction States ---
        # Reconstruct on the physical domain + one layer of ghost cells in the x-dir
        P_center_y = P_padded[:, 1:-1, 1:-1]
        slopes_y_sliced = slopes_y
        
        P_at_top_faces = P_center_y + 0.5 * slopes_y_sliced
        P_at_bottom_faces = P_center_y - 0.5 * slopes_y_sliced
        
        # Assemble states for horizontal interfaces
        P_B_y = P_at_top_faces[:, :-1, 1:-1]
        P_T_y = P_at_bottom_faces[:, 1:, 1:-1]
        
        # --- Convert to conserved ---
        U_L_x = primitive_to_conserved_general(P_L_x)
        U_R_x = primitive_to_conserved_general(P_R_x)
        U_B_y = primitive_to_conserved_general(P_B_y)
        U_T_y = primitive_to_conserved_general(P_T_y)

    # --- Numerical Flux Calculation ---
    if riemann_solver == 'hllc':
        F_hat = hllc_flux(U_L_x, U_R_x, 'x', wave_speed_method=hllc_wave_speed)
        G_hat = hllc_flux(U_B_y, U_T_y, 'y', wave_speed_method=hllc_wave_speed)
    elif riemann_solver == 'hll':
        F_hat = hll_flux(U_L_x, U_R_x, 'x')
        G_hat = hll_flux(U_B_y, U_T_y, 'y')
    else:
        raise ValueError(f"Unknown Riemann solver: {riemann_solver}")
        
    # --- RHS Calculation using divergence of fluxes ---
    div_F = (F_hat[:, :, 1:] - F_hat[:, :, :-1]) / dx 
    div_G = (G_hat[:, 1:, :] - G_hat[:, :-1, :]) / dy
            
    RHS = -(div_F + div_G)
            
    return RHS
