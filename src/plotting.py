"""
This module contains functions for visualizing the results of the CFD simulation.
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_results(X, Y, rho, t_final, ny, problem_type, config_str):
    """
    Generates and displays plots for the final state of the simulation.
    
    Args:
        X, Y (np.array): Meshgrid arrays for coordinates.
        rho (np.array): The final density field.
        t_final (float): The final simulation time.
        ny (int): Number of points in y-direction.
        problem_type (str): The name of the problem being solved.
        config_str (str): A string describing the solver configuration.
    """
    title = f'{problem_type.upper()} @ t={t_final:.2f}\n({config_str})'
    
    if problem_type == 'dmr':
        plt.figure(figsize=(16, 4))
        levels = np.linspace(1, 22, 60)
        line_levels = np.linspace(1, 22, 60)
        plt.contourf(X, Y, rho, levels=levels, cmap='jet')
        #plt.contour(X, Y, rho, levels=line_levels, colors='black', linewidths=0.5)

    elif problem_type == '2d_riemann':
        plt.figure(figsize=(9, 8))
        levels = 100
        line_levels = np.linspace(rho.min(), rho.max(), 30)
        plt.contour(X, Y, rho, levels=line_levels, colors='white', linewidths=0.5)
        plt.contourf(X, Y, rho, levels=levels, cmap='jet')

    elif problem_type == '2d_sod':
        plt.figure(figsize=(16, 4))
        # Use imshow for a clean rectangular plot
        plt.imshow(rho, cmap='jet', extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower')

    plt.colorbar(label='Density (ρ)')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

    # --- ADDED: 1D Plot for 2D Sod Problem ---
    if problem_type == '2d_sod':
        plt.figure(figsize=(10, 6))
        j_slice = ny // 2
        density_1d = rho[j_slice, :]
        x_coords = X[j_slice, :]
        
        plt.plot(x_coords, density_1d, 'b-o', markersize=3, label=f'Density at y ≈ {Y[j_slice, 0]:.2f}')
        plt.title(f"1D Density Profile from 2D Sod Tube at t={t_final:.2f}")
        plt.xlabel("x-coordinate")
        plt.ylabel("Density ($\\rho$)")
        plt.legend()
        plt.grid(True)
        plt.show()