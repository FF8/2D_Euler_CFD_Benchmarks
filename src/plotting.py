"""
This module contains functions for visualizing the results of the CFD simulation.
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_results(X, Y, rho, t, ny, problem_type, config_str, save_path=None):
    """
    Generates and displays or saves plots for the simulation state.
    
    Args:
        X, Y (np.array): Meshgrid arrays for coordinates.
        rho (np.array): The density field.
        t (float): The current simulation time.
        ny (int): Number of points in y-direction.
        problem_type (str): The name of the problem being solved.
        config_str (str): A string describing the solver configuration.
        save_path (str, optional): If provided, saves the figure to this path
                                   instead of showing it. Defaults to None.
    """
    title = f'{problem_type.upper()} @ t={t:.3f}\n({config_str})'
    fig = plt.figure(figsize=(12, 6)) # Create a figure object

    # --- Main 2D Density Plot ---
    if problem_type == 'dmr':
        fig.set_size_inches(16, 4)
        levels = np.linspace(1, 22, 60)
        line_levels = np.linspace(1, 22, 20)
        #plt.contour(X, Y, rho, levels=line_levels, colors='black', linewidths=0.5)
        plt.contourf(X, Y, rho, levels=levels, cmap='jet')

    elif problem_type == '2d_riemann':
        fig.set_size_inches(9, 8)
        # Use nanmin/nanmax to handle potential NaNs from plotting before simulation finishes
        min_rho, max_rho = np.nanmin(rho), np.nanmax(rho)
        levels = np.linspace(min_rho, max_rho, 100)
        line_levels = np.linspace(min_rho, max_rho, 30)
        plt.contour(X, Y, rho, levels=line_levels, colors='white', linewidths=0.5)
        plt.contourf(X, Y, rho, levels=levels, cmap='jet')

    elif problem_type == '2d_sod':
        fig.set_size_inches(16, 4)
        plt.imshow(rho, cmap='jet', extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower', aspect='auto')

    plt.colorbar(label='Density (ρ)')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    if problem_type != '2d_sod':
        plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    # --- Logic to Save or Show ---
    if save_path:
        print(f"Saving frame: {save_path}")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig) # Close the figure to free memory and continue the loop
    else:
        plt.show() # Show the interactive plot window

    # --- 1D Plot for 2D Sod Problem (only shown at the very end) ---
    if problem_type == '2d_sod' and not save_path:
        plt.figure(figsize=(10, 6))
        j_slice = ny // 2
        density_1d = rho[j_slice, :]
        x_coords = X[j_slice, :]
        
        plt.plot(x_coords, density_1d, 'b-o', markersize=3, label=f'Density at y ≈ {Y[j_slice, 0]:.2f}')
        plt.title(f"1D Density Profile from 2D Sod Tube at t={t:.3f}")
        plt.xlabel("x-coordinate")
        plt.ylabel("Density ($\\rho$)")
        plt.legend()
        plt.grid(True)
        plt.show()

