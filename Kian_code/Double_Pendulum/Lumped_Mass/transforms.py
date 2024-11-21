import torch
from torch import Tensor
from typing import Tuple




def transform_dynamical_matrices(M_q: Tensor, C_q: Tensor, G_q: Tensor, J_h_inv: Tensor, 
                                J_h_inv_trans: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

    
    # @ performs batched matrix multiplication on the terms
    M_th = J_h_inv_trans @ M_q @ J_h_inv
    C_th = None
    G_th = J_h_inv_trans @ G_q

    
    return M_th, C_th, G_th


def analytic_theta_1(rp: dict, q: Tensor) -> Tensor:
    
    # h1 is defined as the length between the actuator attachment point and the mass of the double pendulum
    
    Rx = rp["xa"] - rp["l1"] * torch.cos(q[0]) - rp["l2"] * torch.cos(q[1])
    Ry = rp["ya"] - rp["l1"] * torch.sin(q[0]) - rp["l2"] * torch.sin(q[1])
    
    h1 = torch.sqrt(Rx**2 + Ry**2)
    
    return h1
    
    
def analytic_theta_2(rp: dict, q: Tensor) -> Tensor:
    
    # h2 is defined as the arctan between the vector from mass of double pendulum 
    # to the actuator point. 
    # This was Cosimo's hunch, and has been verified in the Mathematica code
    
    Rx = rp["xa"] - rp["l1"] * torch.cos(q[0]) - rp["l2"] * torch.cos(q[1])
    Ry = rp["ya"] - rp["l1"] * torch.sin(q[0]) - rp["l2"] * torch.sin(q[1])    
    
    h2 = torch.atan2(Ry,Rx)
    
    return h2