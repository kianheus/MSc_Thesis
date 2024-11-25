import torch
import torch.nn as nn
from torch import Tensor
import Double_Pendulum.Lumped_Mass.dynamics as dynamics

criterion = nn.MSELoss()




def loss_diagonality_geo_mean(M_th: Tensor, batch_size: int, device: torch.device) -> Tensor:

    """
    Uses the geometric mean (square root of the product) of the diagonal entries of 
    M_th as the denominator.
    """


    # Extract off-diagonal and diagonal elements
    off_dia = M_th[:, 0, 1]
    dia = M_th[:, [0, 1], [0, 1]]  # Shape [64, 2]
    # Compute the geometric mean of diagonal elements
    geo_mean = torch.sqrt(dia[:, 0] * dia[:, 1])


    # Try-except block in case components along diagonal are 0
    try:
        M_th_ratio = off_dia/geo_mean
    except:
        print("M_th_ratio broken")
        M_th_ratio = off_dia/(geo_mean + 1e-6)

    loss_diagonality_geo_mean = criterion(M_th_ratio, torch.zeros((batch_size)).to(device))

    return(loss_diagonality_geo_mean)


def loss_diagonality_trace(M_th: Tensor, batch_size: int, device: torch.device) -> Tensor: 

    """
    Uses the trace (sum of the diagonal entries) of M_th as the denominator.
    """

    # Extract off-diagonal elements and calculate trace of M_th
    off_dia = M_th[:, 0, 1]
    trace = torch.einsum('bii->b', M_th) 
    M_th_ratio = off_dia/trace

    loss_diagonality_geo_mean = criterion(M_th_ratio, torch.zeros((batch_size)).to(device))

    return(loss_diagonality_geo_mean)


def loss_diagonality_lowest(M_th: Tensor, batch_size: int, device: torch.device) -> Tensor: 
    
    """
    Uses the lowest value of the diagonal entries of M_th as the denominator.
    """

    # Extract off-diagonal elements and lowest diagonal element of M_th
    off_dia = M_th[:, 0, 1]
    lowest = torch.min(M_th[:, 0, 0], M_th[:, 1, 1])
    M_th_ratio = off_dia/lowest

    loss_diagonality_geo_mean = criterion(M_th_ratio, torch.zeros((batch_size)).to(device))

    return(loss_diagonality_geo_mean)   

def loss_J_h_unitary(J_h: Tensor, batch_size:int, device: torch.device) -> Tensor:
    
    """
    Loss term which imposes unitary-ness on the unactuated DoFs, as a measure to reduce
    "stretching" in the latent space.
    """

    # Obtain unactuated part of J_h and its transpose
    J_h_u = J_h[:, 1, :].unsqueeze(1)
    J_h_u_trans = J_h_u.transpose(-2,-1)

    # Calculate dot product
    dot_product = torch.bmm(J_h_u, J_h_u_trans).squeeze()

    # Dot product should equal 1 for unitary vector
    loss_J_h_unitary = criterion(dot_product, torch.ones((batch_size)).to(device))

    return loss_J_h_unitary

def loss_J_h_cheat(J_h: Tensor, J_h_ana: Tensor):

    """
    Calculates the loss on the Jacobian. This is "cheating" since we do not generally know
    the desired Jacobian.
    """
    loss_J_h_cheat = criterion(J_h, J_h_ana)

    return loss_J_h_cheat

def loss_reconstruction(q: Tensor, q_hat: Tensor) -> Tensor:

    """
    Calculates the reconstruction loss, normally the most important term for an 
    auto-encoder.
    """
    loss_reconstruction = criterion(q_hat, q)

    return(loss_reconstruction)
    

def loss_l1(model):

    """
    Calculates the l_1 (regularization) loss on the model parameters.
    """
    
    loss_l1 = sum(p.abs().sum() for p in model.parameters())
    return(loss_l1)

def loss_M_th_cheat(M_th: Tensor, rp: dict, q: Tensor, q_d: Tensor, batch_size: int) -> Tensor:

    """
    Calculates loss on M_th with the "desired" analytical M_th. This is again "cheating"
    as M_th_ana is not generally known.
    """

    # Obtain M_th_ana and resize to compare with M_th
    matrices_th_vmap = torch.vmap(dynamics.dynamical_matrices_th, 
                                in_dims=(None, 0, 0))
    M_th_ana, _, _ = matrices_th_vmap(rp, q, q_d)

    # Used to be this loss, but other seems more physical
    #loss_M_th_cheat = torch.linalg.norm(M_th - M_th_ana)

    loss_M_th_cheat = criterion(M_th, M_th_ana)

    return loss_M_th_cheat