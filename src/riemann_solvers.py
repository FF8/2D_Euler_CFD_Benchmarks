"""
This module contains the Riemann solvers used to calculate the numerical flux
at the interfaces between computational cells. These solvers are the core
of the finite volume method used here.
"""
import numpy as np
from config import GAMMA
from utils import cons_to_prim_general, get_flux_general

def hll_flux(U_L, U_R, direction='x'):
    """HLL Riemann solver for the 2D Euler equations."""
    # Rotate states for y-direction to treat it as an x-direction problem
    if direction == 'y':
        U_L_rot = np.copy(U_L); U_L_rot[1] = U_L[2]; U_L_rot[2] = U_L[1]
        U_R_rot = np.copy(U_R); U_R_rot[1] = U_R[2]; U_R_rot[2] = U_R[1]
    else:
        U_L_rot, U_R_rot = U_L, U_R
    
    rho_L, u_L, _, p_L = cons_to_prim_general(U_L_rot)
    rho_R, u_R, _, p_R = cons_to_prim_general(U_R_rot)
    
    F_L, _ = get_flux_general(U_L_rot)
    F_R, _ = get_flux_general(U_R_rot)
    
    # --- ROBUSTNESS FIX ---
    # Ensure arguments to sqrt are non-negative
    c_L = np.sqrt(GAMMA * np.maximum(p_L, 1e-9) / np.maximum(rho_L, 1e-9))
    c_R = np.sqrt(GAMMA * np.maximum(p_R, 1e-9) / np.maximum(rho_R, 1e-9))
    
    S_L = np.minimum(u_L - c_L, u_R - c_R)
    S_R = np.maximum(u_L + c_L, u_R + c_R)
    
    den = S_R - S_L + 1e-9
    F_HLL = (S_R * F_L - S_L * F_R + S_L * S_R * (U_R_rot - U_L_rot)) / den
    
    flux_rot = np.where(S_L[np.newaxis, ...] >= 0, F_L, F_HLL)
    flux_rot = np.where(S_R[np.newaxis, ...] <= 0, F_R, flux_rot)

    # De-rotate the flux if necessary
    if direction == 'y':
        flux_final = np.copy(flux_rot)
        flux_final[1] = flux_rot[2]
        flux_final[2] = flux_rot[1]
        return flux_final
    else:
        return flux_rot

def hllc_flux(U_L, U_R, direction='x', wave_speed_method='roe'):
    """Robust HLLC Riemann solver for the 2D Euler equations."""
    # Rotate states for y-direction to treat it as an x-direction problem
    if direction == 'y':
        U_L_rot = np.copy(U_L); U_L_rot[1] = U_L[2]; U_L_rot[2] = U_L[1]
        U_R_rot = np.copy(U_R); U_R_rot[1] = U_R[2]; U_R_rot[2] = U_R[1]
    else:
        U_L_rot, U_R_rot = U_L, U_R
        
    rho_L, u_L, v_L, p_L = cons_to_prim_general(U_L_rot)
    rho_R, u_R, v_R, p_R = cons_to_prim_general(U_R_rot)
    
    # --- ROBUSTNESS FIX ---
    # Ensure arguments to sqrt are non-negative
    c_L = np.sqrt(GAMMA * np.maximum(p_L, 1e-9) / np.maximum(rho_L, 1e-9))
    c_R = np.sqrt(GAMMA * np.maximum(p_R, 1e-9) / np.maximum(rho_R, 1e-9))

    if wave_speed_method == 'roe':
        E_L=U_L_rot[3]; E_R=U_R_rot[3]; h_L=(E_L+p_L)/rho_L; h_R=(E_R+p_R)/rho_R
        sqrt_rho_L=np.sqrt(np.maximum(rho_L, 1e-9)); sqrt_rho_R=np.sqrt(np.maximum(rho_R, 1e-9))
        u_roe=(sqrt_rho_L*u_L+sqrt_rho_R*u_R)/(sqrt_rho_L+sqrt_rho_R)
        h_roe=(sqrt_rho_L*h_L+sqrt_rho_R*h_R)/(sqrt_rho_L+sqrt_rho_R)
        # v_sq_roe term here would not exist in a purely 1D solver
        # Accounts for the kinetic energy of the flow moving parallel to the cell interface.
        v_sq_roe=(sqrt_rho_L*v_L**2+sqrt_rho_R*v_R**2)/(sqrt_rho_L+sqrt_rho_R)  
        c_roe=np.sqrt(np.maximum((GAMMA-1)*(h_roe-0.5*(u_roe**2+v_sq_roe)),1e-9))
        S_L=np.minimum(u_L-c_L, u_roe-c_roe)
        S_R=np.maximum(u_R+c_R, u_roe+c_roe)
    else: # Davis wave speed estimate
        S_L=np.minimum(u_L-c_L,u_R-c_R)
        S_R=np.maximum(u_L+c_L,u_R+c_R)

    den = rho_L*(S_L-u_L)-rho_R*(S_R-u_R)
    S_M = (p_R-p_L+rho_L*u_L*(S_L-u_L)-rho_R*u_R*(S_R-u_R))/(den+1e-9)
    F_L,_=get_flux_general(U_L_rot)
    F_R,_=get_flux_general(U_R_rot)

    U_star_L=np.zeros_like(U_L_rot)
    U_star_L[0]=rho_L*(S_L-u_L)/(S_L-S_M+1e-9)
    U_star_L[1]=U_star_L[0]*S_M; U_star_L[2]=U_star_L[0]*v_L # We have four terms, this is a 2D problem. 
    U_star_L[3]=U_star_L[0]*(U_L_rot[3]/rho_L+(S_M-u_L)*(S_M+p_L/(rho_L*(S_L-u_L)+1e-9)))
    
    U_star_R=np.zeros_like(U_R_rot)
    U_star_R[0]=rho_R*(S_R-u_R)/(S_R-S_M+1e-9)
    U_star_R[1]=U_star_R[0]*S_M; U_star_R[2]=U_star_R[0]*v_R
    U_star_R[3]=U_star_R[0]*(U_R_rot[3]/rho_R+(S_M-u_R)*(S_M+p_R/(rho_R*(S_R-u_R)+1e-9)))
    
    F_star_L=F_L+S_L[np.newaxis,...]*(U_star_L-U_L_rot)
    F_star_R=F_R+S_R[np.newaxis,...]*(U_star_R-U_R_rot)
    
    cond_L=S_L>=0
    cond_R=S_R<=0
    cond_M=S_M>=0
    
    flux_rot=np.where(cond_L[np.newaxis,...],F_L,F_star_L)
    flux_rot=np.where(np.logical_and(np.logical_not(cond_L), cond_M)[np.newaxis,...],F_star_L,flux_rot)
    flux_rot=np.where(np.logical_and(np.logical_not(cond_L), np.logical_not(cond_M))[np.newaxis,...],F_star_R,flux_rot)
    flux_rot=np.where(cond_R[np.newaxis,...],F_R,flux_rot)

    # De-rotate the flux if necessary
    if direction=='y':
        flux_final=np.copy(flux_rot)
        flux_final[1]=flux_rot[2]
        flux_final[2]=flux_rot[1]
        return flux_final
    else: 
        return flux_rot
