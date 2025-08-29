# main.py
"""
The main entry point for the 2D Euler CFD solver.
This script handles command-line argument parsing, problem setup,
the main time-integration loop, and final plotting.
"""
import numpy as np
import time
import argparse
import os # <<< NEW: Import os for directory handling

# --- Local Module Imports ---
from config import CFL, GAMMA
from setups import setup_dmr, setup_2d_riemann, setup_2d_sod, setup_sedov_explosion
from schemes import calculate_rhs
from utils import cons_to_prim_general
from plotting import plot_results

def run_simulation(args):
    """Main simulation loop for the 2D CFD solver."""
    
    # --- Create output directory if saving frames ---
    # <<< NEW: This whole block is new
    output_dir = "output_frames"
    if args.save_frames:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Frames will be saved in '{output_dir}/'")

    # --- Problem Setup ---
    problem_setups = {
        'dmr': setup_dmr,
        '2d_riemann': setup_2d_riemann,
        '2d_sod': setup_2d_sod,
        'blast': setup_sedov_explosion
        }
    if args.problem not in problem_setups:
        raise ValueError(f"Unknown problem_type: {args.problem}")
        
    U, params = problem_setups[args.problem]()
    
    # Unpack parameters
    t_final = params['t_final']
    dx, dy = params['dx'], params['dy']
    X, Y = params['X'], params['Y']
    ny = params['ny']
    
    bc_params = {
        'problem_type': args.problem,
        'reconstruction_scheme': args.scheme,
        'riemann_solver': args.solver,
        'hllc_wave_speed': args.hllc_ws,
        **params
    }

    print(f"--- Starting Simulation ---")
    print(f"Problem: {args.problem.upper()}, Time Integrator: {args.integrator.upper()}")
    print(f"Scheme: {args.scheme}, Riemann Solver: {args.solver.upper()}")
    if args.solver == 'hllc': print(f"HLLC Wave Speed: {args.hllc_ws.upper()}")
    print(f"Grid: {params['nx']}x{params['ny']}, Target Time: {t_final:.2f}")
    
    # --- Main Time Loop ---
    t = 0.0
    start_time = time.time()
    
    # <<< NEW: Variables for frame saving
    frame_count = 0
    next_frame_time = 0.0
    
    while t < t_final:
        # Calculate time step (dt)
        rho_curr, u_vel, v_vel, p_pres = cons_to_prim_general(U)
        cs = np.sqrt(GAMMA * np.maximum(p_pres, 1e-9) / np.maximum(rho_curr, 1e-9))
        dt = CFL / (np.max(np.abs(u_vel) / dx + cs / dx) + np.max(np.abs(v_vel) / dy + cs / dy))
        if t + dt > t_final: dt = t_final - t

        # --- Time Integration ---
        if args.integrator == "euler":
            RHS = calculate_rhs(U, dx, dy, t, bc_params)
            U = U + dt * RHS
        elif args.integrator == "ssprk2":
            RHS1 = calculate_rhs(U, dx, dy, t, bc_params)
            U_star = U + dt * RHS1
            RHS2 = calculate_rhs(U_star, dx, dy, t + dt, bc_params)
            U = 0.5 * U + 0.5 * (U_star + dt * RHS2)
            
        t += dt
        if np.isnan(U).any():
            print("Error: NaN detected in solution array. Simulation stopped.")
            break
            
        print(f"t = {t:.4f}/{t_final} (dt = {dt:.2e})")

        # <<< NEW: Frame saving logic
        if args.save_frames and t >= next_frame_time:
            rho, _, _, _ = cons_to_prim_general(U)
            ws_str = f", HLLC WS: {args.hllc_ws.upper()}" if args.solver == 'hllc' else ""
            config_str = f"Scheme: {args.scheme}, Solver: {args.solver.upper()}{ws_str}"
            
            save_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
            plot_results(X, Y, rho, t, ny, args.problem, config_str, save_path=save_path)
            
            frame_count += 1
            next_frame_time += args.frame_interval

    end_time = time.time()
    print(f"\n--- Simulation Finished ---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")
    
    # --- Final Plotting (only if not saving frames) ---
    if not args.save_frames:
        rho, _, _, _ = cons_to_prim_general(U)
        ws_str = f", HLLC WS: {args.hllc_ws.upper()}" if args.solver == 'hllc' else ""
        config_str = (f"Scheme: {args.scheme}, Solver: {args.solver.upper()}{ws_str}, "
                      f"Integrator: {args.integrator}")
        plot_results(X, Y, rho, t, ny, args.problem, config_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A 2D Euler equation solver for CFD benchmarks.")
    
    parser.add_argument('--problem', type=str, default='dmr', choices=['dmr', '2d_riemann', '2d_sod', 'sedov_explosion'], help='The benchmark problem to solve.')
    parser.add_argument('--scheme', type=str, default='piecewise_linear', choices=['piecewise_constant', 'piecewise_linear'], help='The spatial reconstruction scheme.')
    parser.add_argument('--integrator', type=str, default='ssprk2', choices=['euler', 'ssprk2'], help='The time integration scheme.')
    parser.add_argument('--solver', type=str, default='hllc', choices=['hll', 'hllc'], help='The Riemann solver to use for numerical flux.')
    parser.add_argument('--hllc_ws', type=str, default='roe', choices=['roe', 'davis'], help='Wave speed estimation method for HLLC solver.')
    
    # <<< NEW: Arguments for frame saving
    parser.add_argument('--save_frames', action='store_true', help='Enable saving frames for a movie.')
    parser.add_argument('--frame_interval', type=float, default=0.005, help='Time interval between saved frames.')

    args = parser.parse_args()
    run_simulation(args)
