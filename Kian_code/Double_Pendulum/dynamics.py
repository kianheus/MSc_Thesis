import torch
from torch import Tensor
from typing import Tuple
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    c0 = torch.cos(q[0]).unsqueeze(0)
    c1 = torch.cos(q[1]).unsqueeze(0)
    s01 = torch.sin(q[0]-q[1]).unsqueeze(0)
    c01 = torch.cos(q[0]-q[1]).unsqueeze(0)
    
    M_q_00 = torch.tensor([rp["l0"]**2 * (rp["m0"] + rp["m1"])]).to(device) 
    M_q_0 = torch.cat((M_q_00, (rp["l0"] * rp["l1"] * rp["m1"] * c01).to(device)), dim = 0).unsqueeze(0)
    M_q_11 = torch.tensor([rp["l1"]**2 * rp["m1"]]).to(device) 
    M_q_1 = torch.cat(((rp["l0"] * rp["l1"] * rp["m1"] * c01).to(device), M_q_11), dim = 0).unsqueeze(0)
    M_q = torch.cat((M_q_0, M_q_1), dim = 0)
        
    C_q_00 = torch.tensor([0]).to(device) 
    C_q_0 = torch.cat((C_q_00, (rp["l0"] * rp["l1"] * rp["m1"] * q_d[1] * s01).to(device)), dim=0).unsqueeze(0)
    C_q_1 = torch.cat(((-rp["l0"] * rp["l1"] * rp["m1"] * q_d[0] * s01).to(device), C_q_00), dim=0).unsqueeze(0)
    C_q = torch.cat((C_q_0, C_q_1), dim = 0)
    
    G_q_0 = torch.tensor([rp["g"] * rp["l0"] * (rp["m0"] + rp["m1"])]).unsqueeze(0).to(device) * c0
    G_q_1 = torch.tensor([rp["g"] * rp["l1"] * rp["m1"]]).unsqueeze(0).to(device)  * c1
    G_q = torch.cat((G_q_0, G_q_1), dim = 0).to(device)


    return M_q, C_q, G_q 

def potential_matrix(rp, q):
    
    """
    Calculate just G_q
    """
    
    c0 = torch.cos(q[0]).unsqueeze(0)
    c1 = torch.cos(q[1]).unsqueeze(0)

    G_q_0 = torch.tensor([rp["g"] * rp["l0"] * (rp["m0"] + rp["m1"])]).unsqueeze(0).to(device) * c0
    G_q_1 = torch.tensor([rp["g"] * rp["l1"] * rp["m1"]]).unsqueeze(0).to(device)  * c1
    G_q = torch.cat((G_q_0, G_q_1), dim = 0).to(device)


    return G_q 


def add_spring_force_G_q(rp: dict, q: Tensor, G_q, k_spring: Tensor, rest_angles: Tensor) -> Tensor:

    """
    Adds spring force to the potential matrix based on spring constant k_spring.
    """
    k0 = k_spring[0]
    k1 = k_spring[1]

    offset0 = rest_angles[0]
    offset1 = rest_angles[1]

    K_q0 = torch.tensor([[-k0,  0],
                         [0,    0]]).to(device)

    K_q1 = torch.tensor([[-k1, k1], 
                         [k1, -k1]]).to(device)
    
    bias = torch.tensor([[k0 * offset0 - k1 * offset1],
                         [0.           + k1 * offset1]]).to(device)

    G_q_spring = (K_q0 + K_q1) @ q.T + bias
    G_q_total = G_q + G_q_spring

    return G_q_total


def input_matrix(rp: dict, q: Tensor) -> Tensor:

    """
    Computes the input matrix A (analytically) in the equation 
    M_q @ q_dd + C_q @ q_d + G_q = A_q @ u

    Args:
        rp: dictionary of robot parameters
        q: link angles of shape (2, )
    Returns:
        A_q: input matrix of shape (2, )   
    """

    Rx = rp["xa"] - rp["l0"] * torch.cos(q[0]) - rp["l1"] * torch.cos(q[1])
    Ry = rp["ya"] - rp["l0"] * torch.sin(q[0]) - rp["l1"] * torch.sin(q[1])

    A_q0 = ( (rp["l0"] * torch.sin(q[0]) * Rx - rp["l0"] * torch.cos(q[0]) * Ry) / torch.sqrt(Rx**2 + Ry**2) ).unsqueeze(0).unsqueeze(0)
    A_q1 = ( (rp["l1"] * torch.sin(q[1]) * Rx - rp["l1"] * torch.cos(q[1]) * Ry) / torch.sqrt(Rx**2 + Ry**2) ).unsqueeze(0).unsqueeze(0)
    A_q = torch.cat((A_q0, A_q1), dim = 0)

    return A_q.to(device)


def jacobian(rp: dict, q: Tensor) -> Tensor:

    """
    Computes the forward Jacobian of the analytic transform
    th = h(q) with th the partial inertial and input decoupling coordinates. 
    Jh = d(q)/d(t)
    """

    Rx = rp["xa"] - rp["l0"] * torch.cos(q[0]) - rp["l1"] * torch.cos(q[1])
    Ry = rp["ya"] - rp["l0"] * torch.sin(q[0]) - rp["l1"] * torch.sin(q[1])

    l = torch.sqrt(Rx**2 + Ry**2)
    alpha = torch.atan2(Ry, Rx)
    
    Jh = torch.cat((torch.autograd.grad(l, q, create_graph=True)[0].unsqueeze(0),
                       torch.autograd.grad(alpha, q, create_graph=True)[0].unsqueeze(0)), dim=0)
    
    return Jh

#def inverse_jacobian(rp: dict, th: Tensor) -> Tensor:
