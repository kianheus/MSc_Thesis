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
    s12 = torch.sin(q[0]-q[1]).unsqueeze(0)
    c12 = torch.cos(q[0]-q[1]).unsqueeze(0)
    
    M_q_11 = torch.tensor([rp["l1"]**2 * rp["m"]]).to(device) 
    M_q_1 = torch.cat((M_q_11, (rp["l1"] * rp["l2"] * rp["m"] * c12).to(device)), dim = 0).unsqueeze(0)
    M_q_22 = torch.tensor([rp["l2"]**2 * rp["m"]]).to(device) 
    M_q_2 = torch.cat(((rp["l1"] * rp["l2"] * rp["m"] * c12).to(device), M_q_22), dim = 0).unsqueeze(0)
    M_q = torch.cat((M_q_1, M_q_2), dim = 0)
        
    C_q_11 = torch.tensor([0]).to(device) 
    C_q_1 = torch.cat((C_q_11, (rp["l1"] * rp["l2"] * rp["m"] * q_d[1] * s12).to(device)), dim=0).unsqueeze(0)
    C_q_2 = torch.cat(((-rp["l1"] * rp["l2"] * rp["m"] * q_d[0] * s12).to(device), C_q_11), dim=0).unsqueeze(0)
    C_q = torch.cat((C_q_1, C_q_2), dim = 0)
    
    G_q_1 = torch.tensor([rp["g"] * rp["l1"] * rp["m"]]).unsqueeze(0).to(device)  * c1.to(device)
    G_q_2 = torch.tensor([rp["g"] * rp["l2"] * rp["m"]]).unsqueeze(0).to(device)  * c2.to(device)
    G_q = torch.cat((G_q_1, G_q_2), dim = 0)


    return M_q, C_q, G_q 


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

    Rx = rp["xa"] - rp["l1"] * torch.cos(q[0]) - rp["l2"] * torch.cos(q[1])
    Ry = rp["ya"] - rp["l1"] * torch.sin(q[0]) - rp["l2"] * torch.sin(q[1])

    A_q1 = ( (rp["l1"] * torch.sin(q[0]) * Rx - rp["l1"] * torch.cos(q[0]) * Ry) / torch.sqrt(Rx**2 + Ry**2) ).unsqueeze(0).unsqueeze(0)
    A_q2 = ( (rp["l2"] * torch.sin(q[1]) * Rx - rp["l2"] * torch.cos(q[1]) * Ry) / torch.sqrt(Rx**2 + Ry**2) ).unsqueeze(0).unsqueeze(0)

    A_q = torch.cat((A_q1, A_q2), dim = 0)

    return A_q
