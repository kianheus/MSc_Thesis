import torch
from torch import Tensor
from typing import Tuple
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dynamical_matrices(rp: dict, q: Tensor, q_d: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

    """
    Computes the dynamical matrices of a SEA with "naive" coordinates in the equation
    M_q @ q_dd + G_q = tau_q

    Args:
        rp: dictionary of robot parameters
        q: (motor angle, link angle - motor angle) of shape (2, )
    Returns:
        M_q: inertial matrix of shape (2, 2)
        G_q: gravitational matrix of shape (2, 2)     
    """
    
    M_q_0 = torch.cat((torch.tensor([rp["I0"] + rp["I1"]]), torch.tensor([rp["I1"]])), dim = 0).unsqueeze(0)
    M_q_1 = torch.cat((torch.tensor([rp["I1"]]), torch.tensor([rp["I1"]])), dim = 0).unsqueeze(0)
    M_q = torch.cat((M_q_0, M_q_1), dim = 0)

    G_q_0 = torch.cat((torch.tensor([0.]), torch.tensor([0.])), dim = 0).unsqueeze(0)
    G_q_1 = torch.cat((torch.tensor([0.]), torch.tensor([rp["k"]])), dim = 0).unsqueeze(0)
    G_q = torch.cat((G_q_0, G_q_1), dim = 0)


    return M_q, G_q 


def input_matrix() -> Tensor:

    """
    Computes the input matrix A (analytically) in the equation 
    M_q @ q_dd + C_q @ q_d + G_q = A_q @ u

    Args:
        None
    Returns:
        A_q: input matrix of shape (2, )   
    """

    A_q = torch.cat((torch.tensor([1.]), torch.tensor([0.])), dim = 0).unsqueeze(-1)

    return A_q


def jacobian() -> Tensor:

    """
    Computes the forward Jacobian of the analytic transform
    th = h(q) where th = (motor angle, link angle)
    Jh = d(q)/d(t)
    """

    J_h_0 = torch.cat((1., 0.), dim = 0).unsqueeze(0)
    J_h_1 = torch.cat((1., 1.), dim = 0).unsqueeze(0)
    J_h = torch.cat((J_h_0, J_h_1), dim = 0)
    print(J_h)
    
    return J_h
