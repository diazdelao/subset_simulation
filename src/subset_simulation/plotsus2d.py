"""
Plot Subset Simulation sample levels and performance contours in 2D.

The plotting routine colors samples by SuS level and overlays contours of the
active test function using batched grid evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from G import G_batch, get_test_function_name

def plotsus2d(x, L, Yi, d):
    """
    Plots the individual samples of each Subset Simulation level in 2D.

    Parameters:
    x : numpy array
        The array containing the samples of the subset simulation.
    L : int
        The number of levels in the subset simulation.
    Yi : array
        The intermediate thresholds.
    d : int
        The dimension of the data (2D or 3D).
    """
    if L <= 0 or x.size == 0:
        return None, None

    if d == 2:
        x_coords = x[0, :, :]
        y_coords = x[1, :, :]
    elif d == 3:
        x_coords = x[1, :, :]
        y_coords = x[2, :, :]
    else:
        raise ValueError("plotsus2d only supports d == 2 or d == 3.")

    # Set plot limits based on the full sample history.
    Xlims = [x_coords.min(), x_coords.max()]
    Ylims = [y_coords.min(), y_coords.max()]

    # For uniformly distributed grid
    npts = 100
    xx = np.linspace(Xlims[0], Xlims[1], npts)
    yy = np.linspace(Ylims[0], Ylims[1], npts)

    # Create a grid
    X_grid, Y_grid = np.meshgrid(xx, yy)
    grid_points = np.c_[X_grid.ravel(), Y_grid.ravel()]

    # Evaluate the contour grid in one batch instead of a Python loop.
    zz = G_batch(grid_points)
    zz = zz.reshape(X_grid.shape)

    # Plotting the samples and contours
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.cm.get_cmap("viridis", L)
    for level in range(L):
        ax.scatter(
            x_coords[:, level],
            y_coords[:, level],
            s=12,
            alpha=0.7,
            color=cmap(level),
            label=f"Level {level + 1}",
        )

    # Contour plot of the performance function G
    ax.contour(X_grid, Y_grid, zz, levels=15, colors='black',linewidths=1)

    # Labels and Title
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title(f"All SuS Levels ({get_test_function_name()})")
    ax.legend(title='Level', loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True)
    ax.axis('equal')
    return fig, ax
