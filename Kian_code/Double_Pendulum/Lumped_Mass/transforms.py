import torch
from torch import Tensor
from typing import Tuple


def transform_dynamical_matrices(M_q: Tensor, C_q: Tensor, G_q: Tensor, J_h: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    
    J_h_inv = torch.inverse(J_h)
    J_h_inv_trans = torch.transpose(J_h_inv)

    M_theta = J_h_inv_trans @ M_q @ J_h_inv
    C_theta = None
    G_theta = J_h_inv_trans @ G_q

    
    return M_theta, C_theta, G_theta


