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
    

    
    
    c1 = torch.cos(q[0]).unsqueeze(0)
    c2 = torch.cos(q[1]).unsqueeze(0)
    s12 = torch.cos(q[0]-q[1]).unsqueeze(0)
    c12 = torch.cos(q[0]-q[1]).unsqueeze(0)
    
    M_q_11 = torch.tensor([rp["l1"]**2 * rp["m"]]).to(device) 
    M_q_1 = torch.cat((M_q_11, rp["l1"] * rp["l2"] * rp["m"] * c12), dim = 0).unsqueeze(0)
    M_q_22 = torch.tensor([rp["l2"]**2 * rp["m"]]).to(device) 
    M_q_2 = torch.cat((rp["l1"] * rp["l2"] * rp["m"] * c12, M_q_22), dim = 0).unsqueeze(0)
    M_q = torch.cat((M_q_1, M_q_2), dim = 0)
        
    C_q_11 = torch.tensor([0]).to(device) 
    C_q_1 = torch.cat((C_q_11, rp["l1"] * rp["l2"] * rp["m"] * q_d[1] * s12), dim=0).unsqueeze(0)
    C_q_2 = torch.cat((-rp["l1"] * rp["l2"] * rp["m"] * q_d[0] * s12, C_q_11), dim=0).unsqueeze(0)
    C_q = torch.cat((C_q_1, C_q_2), dim = 0)
    
    G_q_1 = torch.tensor([rp["g"] * rp["l1"] * rp["m"]]).unsqueeze(0).to(device)  * c1
    G_q_2 = torch.tensor([rp["g"] * rp["l1"] * rp["m"]]).unsqueeze(0).to(device)  * c2
    G_q = torch.cat((G_q_1, G_q_2), dim = 0)


    return M_q, C_q, G_q 
    
    """

    M_q = torch.tensor(
        [[rp["l1"]**2 * rp["m"],                    rp["l1"] * rp["l2"] * rp["m"] * c12],
         [rp["l1"] * rp["l2"] * rp["m"] * c12,      rp["l2"]**2 * rp["m"]]]
         )

    C_q = torch.tensor(
        [[0,                                                  rp["l1"] * rp["l2"] * rp["m"] * q_d[1] * s12],
         [-rp["l1"] * rp["l2"] * rp["m"] * q_d[0] * s12,      0]]
         )
    
    G_q = torch.tensor(
        [[rp["g"] * rp["l1"] * rp["m"] * c1],
         [rp["g"] * rp["l2"] * rp["m"] * c2]]
    )
    

    return M_q, C_q, G_q 
    """


def dynamical_matrices_th(rp: dict, q: Tensor, q_d: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Computes the dynamical matrices in the equation (analytically)
    M_th @ th_dd + C_th @ th_d + G_th = tau_q

    Args:
        rp: dictionary of robot parameters
        q: link angles of shape (2, )
        q_d: link angular velocities of shape (2, )
    Returns:
        M_th: inertial matrix of shape (2, 2)
        C_th: coriolis and centrifugal matrix of shape (2, 2)
        G_th: gravitational matrix of shape (2, )     
    """
    
    A = rp["xa"] - rp["l1"] * torch.cos(q[0]) - rp["l2"] * torch.cos(q[1])
    B = rp["ya"] - rp["l1"] * torch.sin(q[0]) - rp["l2"] * torch.sin(q[1])
    
    tensor_0 = torch.tensor([0.]).to(device)
    
    M_th_11 = torch.tensor([rp["m"]]).to(device) 
    M_th_1 = torch.cat((M_th_11, tensor_0), dim = 0).unsqueeze(0)
    M_th_2 = torch.cat((tensor_0, (rp["m"]*(A**2+B**2)).unsqueeze(0)), dim = 0).unsqueeze(0)
    M_th = torch.cat((M_th_1, M_th_2), dim = 0)

    C_th = torch.tensor([float("nan")])
    
    G_th = torch.tensor([float("nan")])

    return M_th, C_th, G_th 