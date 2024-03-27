import torch

def normalize_3d_coordinate(xyz: torch.Tensor, padding: float):
    # xyz_norm = xyz / (1 + padding + 1e-3)  # [-1, 1] -> [-0.9, 0.9]
    xyz_norm = xyz * (1 - padding)  # [-1, 1], padding=0.1 -> [-0.9, 0.9]
    # xyz_norm = (xyz_norm + 1) / 2  # [-0.9, 0.9] -> [0.1, 1.9] -> [0.05, 0.95]
    return xyz_norm