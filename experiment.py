import torch
import matplotlib.pyplot as plt
from newton_shultz import newton_shultz_iterative
import numpy as np

def plot_matrix_transformation(matrices_torch, filename='matrix_transformation.png'):
    """
    Plots the matrix transformation over iterations:
    - Heatmaps of the matrix M.
    - Heatmaps of M @ M.T.
    - Singular values of M.
    """
    matrices = [m.cpu().float().numpy() for m in matrices_torch]
    
    num_steps = len(matrices)
    fig = plt.figure(figsize=(num_steps * 3.5, 10))
    gs = fig.add_gridspec(3, num_steps, height_ratios=[1, 1, 1.5])

    all_singular_values = []

    vmin_m, vmax_m = min(m.min() for m in matrices), max(m.max() for m in matrices)
    mmt_matrices = [m @ m.T for m in matrices]
    vmin_mmt, vmax_mmt = min(m.min() for m in mmt_matrices), max(m.max() for m in mmt_matrices)

    for i, M in enumerate(matrices):
        ax_m = fig.add_subplot(gs[0, i])
        im_m = ax_m.imshow(M, cmap='viridis', vmin=vmin_m, vmax=vmax_m, interpolation='nearest')
        ax_m.set_title(f'Step {i}\nM')
        ax_m.set_xticks([])
        ax_m.set_yticks([])

        ax_mmt = fig.add_subplot(gs[1, i])
        im_mmt = ax_mmt.imshow(mmt_matrices[i], cmap='magma', vmin=vmin_mmt, vmax=vmax_mmt, interpolation='nearest')
        ax_mmt.set_title(f'M @ M.T')
        ax_mmt.set_xticks([])
        ax_mmt.set_yticks([])
        
        _, S, _ = np.linalg.svd(M)
        all_singular_values.append(S)

    fig.colorbar(im_m, ax=fig.get_axes()[0:num_steps], shrink=0.7, location='right', pad=0.01)
    fig.colorbar(im_mmt, ax=fig.get_axes()[num_steps:2*num_steps], shrink=0.7, location='right', pad=0.01)

    ax_svd = fig.add_subplot(gs[2, :])
    for i, svals in enumerate(all_singular_values):
        ax_svd.plot(range(len(svals)), svals, 'o-', label=f'Step {i}')
    
    ax_svd.set_title('Singular Values Over Iterations')
    ax_svd.set_xlabel('Singular Value Index')
    ax_svd.set_ylabel('Value')
    ax_svd.legend()
    ax_svd.grid(True)
    ax_svd.set_ylim(bottom=0)
    ax_svd.axhline(1.0, color='r', linestyle='--', label='Target (1.0)')

    fig.suptitle('Newton-Schulz Iteration Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

if __name__ == "__main__":
    torch.manual_seed(42)
    
    matrix_size = 4
    random_matrix = torch.randn(matrix_size, matrix_size)
    
    print(f"Running Newton-Schulz iteration on a random {matrix_size}x{matrix_size} matrix.")
    
    num_iter_steps = 5
    matrices_torch = newton_shultz_iterative(random_matrix, steps=num_iter_steps)
    
    plot_matrix_transformation(matrices_torch)