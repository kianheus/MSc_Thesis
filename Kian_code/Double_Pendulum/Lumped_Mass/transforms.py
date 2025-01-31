import torch
from torch import Tensor
from typing import Tuple



def transform_M(M_q: Tensor, J_h: Tensor, device: torch.device) -> Tensor:
    # Obtain inverse and transpose inverse Jacobian
    J_h_inv = J_h.transpose(1,2).to(device)
    J_h_inv_trans = J_h.to(device)

    # @ performs batched matrix multiplication on Mq
    M_th = J_h_inv_trans @ M_q @ J_h_inv
    
    return M_th

def transform_M_from_inverse(M_q: Tensor, J_h_inv: Tensor, 
                                J_h_inv_trans: Tensor) -> Tensor:

    # @ performs batched matrix multiplication on the terms
    M_th = J_h_inv_trans @ M_q @ J_h_inv

    return M_th

def transform_dynamical_matrices(M_q: Tensor, C_q: Tensor, G_q: Tensor, J_h: Tensor, 
                                device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:

    # Obtain inverse and transpose inverse Jacobian
    J_h_inv = J_h.transpose(1,2).to(device)
    J_h_inv_trans = J_h.to(device)

    # @ performs batched matrix multiplication on the terms
    M_th = J_h_inv_trans @ M_q @ J_h_inv
    C_th = None
    G_th = J_h_inv_trans @ G_q

    
    return M_th, C_th, G_th

def transform_dynamical_from_inverse(M_q: Tensor, C_q: Tensor, G_q: Tensor, theta: Tensor, theta_d: Tensor,
                                     J_h_inv: Tensor, J_h_inv_trans: Tensor) -> Tuple[Tensor, Tensor, Tensor]:


    # @ performs batched matrix multiplication on the terms

    M_th = J_h_inv_trans @ M_q @ J_h_inv

    C_th = torch.zeros(M_th.size()).to(M_q.device)
    for i in range(C_th.size(0)):
        for j in range(C_th.size(1)):
            for k in range(C_th.size(1)):
                M_th_dot_ijk = torch.autograd.grad(M_th[i,j], theta, create_graph=True, allow_unused=True)[0][0,k]
                M_th_dot_ikj = torch.autograd.grad(M_th[i,k], theta, create_graph=True, allow_unused=True)[0][0,j]
                M_th_dot_jki = torch.autograd.grad(M_th[j,k], theta, create_graph=True, allow_unused=True)[0][0,i]
                C_th[i, j] += 0.5 * (M_th_dot_ijk + M_th_dot_ikj - M_th_dot_jki) * theta_d[0, k]

    G_th = J_h_inv_trans @ G_q

    
    return M_th, C_th, G_th

def transform_input_matrix(A_q: Tensor, J_h: Tensor, device: torch.device) -> Tensor:

    # Obtain inverse Jacobian and calculate A_th
    J_h_inv = J_h.transpose(1,2).to(device)
    A_th = J_h_inv @ A_q

    return A_th

def transform_input_matrix_from_inverse_trans(A_q: Tensor, J_h_inv_trans: Tensor, device: torch.device) -> Tensor:

    A_th = J_h_inv_trans @ A_q

    return A_th


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