import torch
from torch import Tensor
from typing import Tuple
from functools import partial


def transform_dynamical_from_inverse(M_q: Tensor, G_q: Tensor, theta: Tensor, theta_d: Tensor,
                                     J_h_inv: Tensor, J_h_inv_trans: Tensor) -> Tuple[Tensor, Tensor, Tensor]:


    """
    Transforms dynamical matrices based on the inverse jacobian (Jh^-1) and inverse transpose jacobian (Jh^-T)

    1) The mass matrix transform follows from conservation of energy and can be written as:
    M_th = Jh^-T @ M_q @ Jh^-1

    2) The Coriolis & Centrifugal matrix follows from the condition that [M' - 2C] be skew-symmetric. 
    An example calculation can be found in https://arxiv.org/pdf/2010.01033:
    C_th = SUM(Gamma_ijk(q) * q_dot)
    Where Gamma_ijk(q) are the Christoffel symbols of the first kind.

    3) The potential matrix can be calculated based on virtual work as:
    G_th = Jh^-T @ G_q
    """

    M_th = J_h_inv_trans @ M_q @ J_h_inv

    C_th = torch.zeros(M_th.size()).to(M_q.device)
    for i in range(C_th.size(0)):
        for j in range(C_th.size(1)):
            for k in range(C_th.size(1)):
                M_th_dot_ijk = torch.autograd.grad(M_th[i,j], theta, create_graph=True)[0][0,k]
                M_th_dot_ikj = torch.autograd.grad(M_th[i,k], theta, create_graph=True)[0][0,j]
                M_th_dot_jki = torch.autograd.grad(M_th[j,k], theta, create_graph=True)[0][0,i]
                C_th[i, j] += 0.5 * (M_th_dot_ijk + M_th_dot_ikj - M_th_dot_jki) * theta_d[0, k]

    G_th = J_h_inv_trans @ G_q

    
    return M_th, C_th, G_th

def transform_input_matrix_from_inverse_trans(A_q: Tensor, J_h_inv_trans: Tensor) -> Tensor:

    """
    Similarly to the calculation of G_th in the function above, the input matrix is calculated as:
    A_th = Jh^-T @ A_q
    """

    A_th = J_h_inv_trans @ A_q

    return A_th


def analytic_theta_0(rp: dict, q: Tensor) -> Tensor:
    
    """
    th0 is defined the same as q0, namely the motor angle
    """
    
    th0 = q[0]
    
    return th0
    
    
def analytic_theta_1(rp: dict, q: Tensor) -> Tensor:
    
    """
    th1 is defined as the link angle, and is thus equal to q1 - 0.
    """

    th1 = q[1] + q[0]
    
    return th1

def analytic_theta(rp:dict, q: Tensor) -> Tensor:

    """
    Combines the individual coordinates into a single set of coordinates. 
    """

    th0 = analytic_theta_0(rp, q)
    th1 = analytic_theta_1(rp, q)

    th = torch.stack([th0, th1], dim=-1)

    return th


def analytic_inverse(rp: dict, th: Tensor) -> Tuple:

    """
    Inverse kinematics from theta to q, based on the end-effector
    position (xend, yend). 
    Returns a tuple with two sets of joint angles, one for clockwise
    and one for counter-clockwise configuration.
    """

    q0 = th[0]
    q1 = th[0] + th[1]
    q = torch.stack([q0, q1], dim=-1)

    return q


def wrap_to_pi(q):

    """
    Returns a tensor with elements wrapped between -pi and pi.
    """

    return (q + torch.pi) % (2 * torch.pi) - torch.pi