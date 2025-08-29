"""
Contains functions to set up the initial conditions for each benchmark problem.
Each function returns the initial conserved state array and a dictionary
of simulation parameters.
"""
import numpy as np
from config import GAMMA
from utils import primitive_to_conserved_general

def setup_dmr():
    """Setup for the Double Mach Reflection problem."""
    nx, ny, x_max, y_max, t_final = 480, 120, 4.0, 1.0, 0.2
    dx, dy = x_max / nx, y_max / ny
    x = np.linspace(dx / 2, x_max - dx / 2, nx)
    y = np.linspace(dy / 2, y_max - dy / 2, ny)
    X, Y = np.meshgrid(x, y)
    
    rho_post, u_post, v_post, p_post = 8.0, 8.25 * np.cos(np.pi/6), -8.25 * np.sin(np.pi/6), 116.5
    U_post = primitive_to_conserved_general(np.array([rho_post, u_post, v_post, p_post]))
    
    rho_pre, p_pre = 1.4, 1.0
    U_pre = primitive_to_conserved_general(np.array([rho_pre, 0.0, 0.0, p_pre]))
    
    angle = 60 * np.pi / 180
    x_shock = 1/6 + Y / np.tan(angle)
    mask = X < x_shock
    
    rho = np.where(mask, rho_post, rho_pre)
    u = np.where(mask, u_post, 0.0)
    v = np.where(mask, v_post, 0.0)
    p = np.where(mask, p_post, p_pre)
    
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
    nx, ny, x_max, y_max, t_final = 200, 10, 1.0, 0.1, 0.2
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



def setup_sedov_explosion():
    """Setup for the Sedov Explosion problem."""
    nx, ny, x_max, y_max, t_final = 200, 200, 1.0, 1.0, 0.05
    dx, dy = x_max / nx, y_max / ny
    
    # Center the computational domain at (0,0)
    x = np.linspace(-x_max / 2 + dx / 2, x_max / 2 - dx / 2, nx)
    y = np.linspace(-y_max / 2 + dy / 2, y_max / 2 - dy / 2, ny)
    X, Y = np.meshgrid(x, y)

    # Ambient conditions
    p_ambient = 2.5e-11  # Use a small number for pressure instead of zero
    rho_ambient = 1.0
    
    # Calculate blast pressure based on total energy deposition
    r0 = 1.5 * dx  # Radius of the initial high-pressure region
    total_energy = 1.0
    p_blast = total_energy * (GAMMA - 1) / (np.pi * r0**2)
    
    # Create a circular mask for the blast region
    radius = np.sqrt(X**2 + Y**2)
    mask = radius <= r0
    
    # Initialize primitive variables
    rho = np.ones_like(X) * rho_ambient
    p = np.full_like(X, p_ambient)
    p[mask] = p_blast
    u = np.zeros_like(X)
    v = np.zeros_like(X)
    
    # Convert from primitive to conserved variables
    U = primitive_to_conserved_general(np.array([rho, u, v, p]))
    
    params = {
        'nx': nx, 'ny': ny, 'x_max': x_max, 'y_max': y_max, 't_final': t_final,
        'X': X, 'Y': Y, 'dx': dx, 'dy': dy
    }
    return U, params