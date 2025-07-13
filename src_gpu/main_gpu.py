# src/main_gpu.py
"""
The main entry point for the GPU-accelerated 2D Euler CFD solver.
This script uses CuPy for all array operations and handles data transfers
to the CPU for plotting.
"""
import cupy as np 
import time
import argparse
import os

# --- Local Module Imports (from GPU-specific files) ---
from config import CFL, GAMMA
from setups_gpu import setup_dmr, setup_2d_riemann, setup_2d_sod
from schemes_gpu import calculate_rhs
from utils_gpu import cons_to_prim_general
from plotting import plot_results # Plotting remains on the CPU

def run_simulation(args):
    """Main simulation loop for the 2D CFD solver."""
    
    output_dir = "output_frames_gpu"
    if args.save_frames:
        os.makedirs(output_dir, exist_ok=True)
        print(f"GPU frames will be saved in '{output_dir}/'")

    # --- Problem Setup (will create CuPy arrays) ---
    problem_setups = {
        'dmr': setup_dmr,
        '2d_riemann': setup_2d_riemann,
        '2d_sod': setup_2d_sod
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

    print(f"--- Starting GPU Simulation ---")
    print(f"Problem: {args.problem.upper()}, Time Integrator: {args.integrator.upper()}")
    print(f"Scheme: {args.scheme}, Riemann Solver: {args.solver.upper()}")
    if args.solver == 'hllc': print(f"HLLC Wave Speed: {args.hllc_ws.upper()}")
    print(f"Grid: {params['nx']}x{params['ny']}, Target Time: {t_final:.2f}")
    
    # --- Main Time Loop ---
    t = 0.0
    start_time = time.time()
    
    frame_count = 0
    next_frame_time = 0.0
    
    while t < t_final:
        # Calculate time step (dt)
        rho_curr, u_vel, v_vel, p_pres = cons_to_prim_general(U)
        cs = np.sqrt(GAMMA * np.maximum(p_pres, 1e-9) / np.maximum(rho_curr, 1e-9))
        
        # Perform max on GPU, then get the single scalar value back to the CPU
        max_val = (np.max(np.abs(u_vel) / dx + cs / dx) + np.max(np.abs(v_vel) / dy + cs / dy)).get()
        
        dt = CFL / max_val
        if t + dt > t_final: dt = t_final - t

        # --- Time Integration (all on GPU) ---
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

        # --- Frame saving logic ---
        if args.save_frames and t >= next_frame_time:
            rho, _, _, _ = cons_to_prim_general(U)
            
            # <<< CRITICAL: Move data to CPU for plotting >>>
            X_plot, Y_plot, rho_plot = X.get(), Y.get(), rho.get()

            ws_str = f", HLLC WS: {args.hllc_ws.upper()}" if args.solver == 'hllc' else ""
            config_str = f"Scheme: {args.scheme}, Solver: {args.solver.upper()}{ws_str}"
            
            save_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
            plot_results(X_plot, Y_plot, rho_plot, t, ny, args.problem, config_str, save_path=save_path)
            
            frame_count += 1
            next_frame_time += args.frame_interval

    end_time = time.time()
    print(f"\n--- Simulation Finished ---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")
    
    # --- Final Plotting (only if not saving frames) ---
    if not args.save_frames:
        rho, _, _, _ = cons_to_prim_general(U)
        
        # <<< CRITICAL: Move data to CPU for plotting >>>
        X_plot, Y_plot, rho_plot = X.get(), Y.get(), rho.get()

        ws_str = f", HLLC WS: {args.hllc_ws.upper()}" if args.solver == 'hllc' else ""
        config_str = (f"Scheme: {args.scheme}, Solver: {args.solver.upper()}{ws_str}, "
                      f"Integrator: {args.integrator}")
        plot_results(X_plot, Y_plot, rho_plot, t, ny, args.problem, config_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A 2D Euler equation solver for CFD benchmarks (GPU Version).")
    
    parser.add_argument('--problem', type=str, default='dmr', choices=['dmr', '2d_riemann', '2d_sod'], help='The benchmark problem to solve.')
    parser.add_argument('--scheme', type=str, default='piecewise_linear', choices=['piecewise_constant', 'piecewise_linear'], help='The spatial reconstruction scheme.')
    parser.add_argument('--integrator', type=str, default='ssprk2', choices=['euler', 'ssprk2'], help='The time integration scheme.')
    parser.add_argument('--solver', type=str, default='hllc', choices=['hll', 'hllc'], help='The Riemann solver to use for numerical flux.')
    parser.add_argument('--hllc_ws', type=str, default='roe', choices=['roe', 'davis'], help='Wave speed estimation method for HLLC solver.')
    
    parser.add_argument('--save_frames', action='store_true', help='Enable saving frames for a movie.')
    parser.add_argument('--frame_interval', type=float, default=0.005, help='Time interval between saved frames.')

    args = parser.parse_args()
    run_simulation(args)
