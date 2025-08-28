import torch

@torch.compile
def newton_shultz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

def newton_shultz_iterative(G: torch.Tensor, steps: int = 5) -> list[torch.Tensor]:
    """
    Newton-Schulz iteration that returns the matrix at each step for visualization.
    This version is not compiled and returns intermediate steps.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    # Using float32 for easier visualization with matplotlib/numpy
    X = G.clone().to(dtype=torch.float32)

    was_transposed = False
    if G.size(-2) > G.size(-1):
        X = X.mT
        was_transposed = True

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    
    matrices = [X.clone()]

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
        matrices.append(X.clone())

    if was_transposed:
        matrices = [m.mT for m in matrices]

    return matrices