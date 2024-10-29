import torch
from torch import Tensor
from typing import Tuple

def dynamical_matrices(rp: dict, q: Tensor, q_d: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Computes the dynamical matrices in the equation
    M_q @ q_dd + C_q @ q_d + G_q = tau_q

    Args:
        rp: dictionary of robot parameters
        q: link angles of shape (2, )
        q_d: link angular velocities of shape (2, )
    Returns:
        M_q: inertial matrix of shape (2, 2)
        C_q: coriolis and centrifugal matrix of shape (2, 2)
        G_q: gravitational matrix of shape (2, )     
    """
    c1 = torch.cos(q[0])
    c2 = torch.cos(q[1])
    s12 = torch.cos(q[0]-q[1])
    c12 = torch.cos(q[0]-q[1])

    M_q = torch.tensor(
        [[rp["l1"]**2 * rp["m"],                    rp["l1"] * rp["l2"] * rp["m"] * c12],
         [rp["l1"] * rp["l2"] * rp["m"] * c12,      rp["l2"]**2 * rp["m"]]]
         )

    C_q = torch.tensor(
        [[0,                    rp["l1"] * rp["l2"] * rp["m"] * q_d[1] * s12],
         [-rp["l1"] * rp["l2"] * rp["m"] * q_d[0] * s12,      0]]
         )
    
    G_q = torch.tensor(
        [[rp["g"] * rp["l1"] * rp["m"] * c1],
         [rp["g"] * rp["l2"] * rp["m"] * c2]]

    )

    return M_q, C_q, G_q 