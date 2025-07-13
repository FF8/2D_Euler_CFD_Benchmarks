# src/setups_gpu.py
"""
Contains functions to set up the initial conditions for each benchmark problem.
Each function returns the initial conserved state array and a dictionary
of simulation parameters.
"""
import cupy as np
from config import GAMMA
from utils_gpu import primitive_to_conserved_general

def setup_dmr():
    """Setup for the Double Mach Reflection problem."""
    nx, ny, x_max, y_max, t_final = 240*4, 240, 4.0, 1.0, 0.2
    dx, dy = x_max / nx, y_max / ny
    x = np.linspace(dx / 2, x_max - dx / 2, nx)
    y = np.linspace(dy / 2, y_max - dy / 2, ny)
    X, Y = np.meshgrid(x, y)
    
    # --- FIX: Explicitly handle scalar creation ---
    # Calculate u and v on the GPU, then retrieve the single float value using .item()
    # This ensures the list passed to np.array contains only Python native types.
    u_post_val = (8.25 * np.cos(np.pi/6)).item()
    v_post_val = (-8.25 * np.sin(np.pi/6)).item()
    
    # Create the primitive state vector on the GPU from a clean list of floats
    P_post = np.array([8.0, u_post_val, v_post_val, 116.5])
    U_post = primitive_to_conserved_general(P_post)
    
    rho_pre, p_pre = 1.4, 1.0
    P_pre = np.array([rho_pre, 0.0, 0.0, p_pre])
    U_pre = primitive_to_conserved_general(P_pre)
    
    angle = 60 * np.pi / 180
    x_shock = 1/6 + Y / np.tan(angle)
    mask = X < x_shock
    
    # Use the original scalar values for np.where
    rho = np.where(mask, 8.0, rho_pre)
    u = np.where(mask, u_post_val, 0.0)
    v = np.where(mask, v_post_val, 0.0)
    p = np.where(mask, 116.5, p_pre)
    
    # This call is now safe because it's operating on full CuPy arrays
    U = primitive_to_conserved_general(np.array([rho, u, v, p]))
    v_shock_h = (10 * np.sqrt(GAMMA * p_pre / rho_pre)) / np.sin(angle)
    
    params = {
        'nx': nx, 'ny': ny, 'x_max': x_max, 'y_max': y_max, 't_final': t_final,
        'U_post_state': U_post, 'U_pre_state': U_pre, 'v_shock_horizontal': v_shock_h,
        'X': X, 'Y': Y, 'dx': dx, 'dy': dy
    }
    return U, params

def setup_2d_riemann():
    """Setup for the 2D Riemann problem."""
    nx, ny, x_max, y_max, t_final = 200, 200, 1.0, 1.0, 0.25
    dx, dy = x_max / nx, y_max / ny
    x = np.linspace(dx / 2, x_max - dx / 2, nx)
    y = np.linspace(dy / 2, y_max - dy / 2, ny)
    X, Y = np.meshgrid(x, y)
    
    rho_ne, u_ne, v_ne, p_ne = 0.5313, 0.0, 0.0, 0.4
    rho_nw, u_nw, v_nw, p_nw = 1.0, 0.7276, 0.0, 1.0
    rho_sw, u_sw, v_sw, p_sw = 0.8, 0.0, 0.0, 1.0
    rho_se, u_se, v_se, p_se = 1.0, 0.0, 0.7276, 1.0
    
    rho, u, v, p = np.zeros((ny,nx)), np.zeros((ny,nx)), np.zeros((ny,nx)), np.zeros((ny,nx))
    mask_ne=(X>=0.5)&(Y>=0.5); mask_nw=(X<0.5)&(Y>=0.5)
    mask_se=(X>=0.5)&(Y<0.5); mask_sw=(X<0.5)&(Y<0.5)
    
    rho[mask_ne]=rho_ne; u[mask_ne]=u_ne; v[mask_ne]=v_ne; p[mask_ne]=p_ne
    rho[mask_nw]=rho_nw; u[mask_nw]=u_nw; v[mask_nw]=v_nw; p[mask_nw]=p_nw
    rho[mask_sw]=rho_sw; u[mask_sw]=u_sw; v[mask_sw]=v_sw; p[mask_sw]=p_sw
    rho[mask_se]=rho_se; u[mask_se]=u_se; v[mask_se]=v_se; p[mask_se]=p_se
    
    U = primitive_to_conserved_general(np.array([rho, u, v, p]))
    params = {
        'nx': nx, 'ny': ny, 'x_max': x_max, 'y_max': y_max, 't_final': t_final,
        'X': X, 'Y': Y, 'dx': dx, 'dy': dy
    }
    return U, params

def setup_2d_sod():
    """Setup for the 2D Sod Shock Tube problem."""
    nx, ny, x_max, y_max, t_final = 400, 40, 1.0, 0.1, 0.2
    dx, dy = x_max / nx, y_max / ny
    x = np.linspace(dx / 2, x_max - dx / 2, nx)
    y = np.linspace(dy / 2, y_max - dy / 2, ny)
    X, Y = np.meshgrid(x, y)

    rho_L, p_L = 1.0, 1.0
    rho_R, p_R = 0.125, 0.1
    
    rho = np.where(X < 0.5, rho_L, rho_R)
    p = np.where(X < 0.5, p_L, p_R)
    u = np.zeros_like(X)
    v = np.zeros_like(X)
    
    U = primitive_to_conserved_general(np.array([rho, u, v, p]))
    params = {
        'nx': nx, 'ny': ny, 'x_max': x_max, 'y_max': y_max, 't_final': t_final,
        'X': X, 'Y': Y, 'dx': dx, 'dy': dy
    }
    return U, params
