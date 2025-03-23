import torch
from torch import Tensor
from typing import Tuple
from functools import partial



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

def transform_dynamical_from_inverse(M_q: Tensor, C_q: Tensor, G_q: Tensor, theta: Tensor, theta_d: Tensor,
                                     J_h_inv: Tensor, J_h_inv_trans: Tensor) -> Tuple[Tensor, Tensor, Tensor]:


    # @ performs batched matrix multiplication on the terms

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
    
    th1 = torch.sqrt(Rx**2 + Ry**2)
    
    return th1
    
    
def analytic_theta_2(rp: dict, q: Tensor) -> Tensor:
    
    # h2 is defined as the arctan between the vector from mass of double pendulum 
    # to the actuator point. 
    # This was Cosimo's hunch, and has been verified in the Mathematica code
    
    Rx = rp["xa"] - rp["l1"] * torch.cos(q[0]) - rp["l2"] * torch.cos(q[1])
    Ry = rp["ya"] - rp["l1"] * torch.sin(q[0]) - rp["l2"] * torch.sin(q[1])    
    
    th2 = torch.atan2(Ry,Rx)
    
    return th2

def analytic_theta(rp:dict, q: Tensor) -> Tensor:
    th1 = analytic_theta_1(rp, q)
    th2 = analytic_theta_2(rp, q)

    th = torch.stack([th1, th2], dim=-1)

    return th


def analytic_inverse(rp: dict, th: Tensor) -> Tuple:

    """
    Inverse kinematics from theta to q, based on the end-effector
    position (xend, yend). 
    Returns a tuple with two sets of joint angles, one for clockwise
    and one for counter-clockwise configuration.
    """

    # Obtain end effector position.

    xend = rp["xa"] - th[0]*torch.cos(th[1])
    yend = rp["ya"] - th[0]*torch.sin(th[1])

    # Calculate the inside angle of the two joints, used to determine q1. Epsilon prevents NaN.
    epsilon = 0.00001

    numerator = (xend**2 + yend**2 - rp["l1"]**2 - rp["l2"]**2)
    denominator = torch.tensor(2*rp["l1"]*rp["l2"])
    fraction = numerator/denominator

    """
    if torch.abs(fraction - 1) > 1.1:
        raise ValueError("End effector outside of robot reach, inverse cannot be calculated")
    else:
    """
    
    epsilon = 1e-6
    fraction = torch.clamp(fraction, -1.0 + epsilon, 1.0 - epsilon)

    beta = torch.arccos(fraction)

    # Determine primary angles.
    q1 = torch.atan2(yend, xend + epsilon) - torch.atan2(rp["l2"]*torch.sin(beta), epsilon + rp["l1"] + rp["l2"]*torch.cos(beta))
    q2 = q1 + beta

    # Determine secondary angles.
    q1_alt = torch.atan2(yend, xend) + torch.atan2(rp["l2"]*torch.sin(beta), epsilon + rp["l1"] + rp["l2"]*torch.cos(beta))
    q2_alt = q1_alt - beta 

    # Normalize values between -pi and pi.
    q1 = (q1 + torch.pi) % (2 * torch.pi) - torch.pi
    q2 = (q2 + torch.pi) % (2 * torch.pi) - torch.pi
    q1_alt = (q1_alt + torch.pi) % (2 * torch.pi) - torch.pi
    q2_alt = (q2_alt + torch.pi) % (2 * torch.pi) - torch.pi

    q = torch.stack([q1, q2], dim=-1)
    q_alt = torch.stack([q1_alt, q2_alt], dim=-1)

    # Check whether the primary angle is clockwise. Otherwise, swap with secondary.
    q_cw = q
    q_ccw = q_alt
    
    return q_cw, q_ccw


def forward_kinematics(rp, q):

    x_end = rp["l1"] * torch.cos(q[0]) + rp["l2"] * torch.cos(q[1])
    y_end = rp["l1"] * torch.sin(q[0]) + rp["l2"] * torch.sin(q[1])

    pos_end = torch.stack([x_end, y_end], dim=-1)

    x_elbow = rp["l1"] * torch.cos(q[0])
    y_elbow = rp["l1"] * torch.sin(q[0])

    pos_elbow = torch.stack([x_elbow, y_elbow], dim=-1)

    return pos_end, pos_elbow

def inverse_kinematics(pos, rp, is_clockwise):
    xend = pos[0]
    yend = pos[1]

    numerator = (xend**2 + yend**2 - rp["l1"]**2 - rp["l2"]**2)
    denominator = torch.tensor(2*rp["l1"]*rp["l2"])
    fraction = numerator/denominator

    beta = torch.arccos(fraction)

    # Determine primary angles.
    q1 = torch.atan2(yend, xend) - torch.atan2(rp["l2"]*torch.sin(beta), rp["l1"] + rp["l2"]*torch.cos(beta))
    q2 = q1 + beta

    # Determine secondary angles.
    q1_alt = torch.atan2(yend, xend) + torch.atan2(rp["l2"]*torch.sin(beta), rp["l1"] + rp["l2"]*torch.cos(beta))
    q2_alt = q1_alt - beta 

    # Normalize values between -pi and pi.
    q1 = (q1 + torch.pi) % (2 * torch.pi) - torch.pi
    q2 = (q2 + torch.pi) % (2 * torch.pi) - torch.pi
    q1_alt = (q1_alt + torch.pi) % (2 * torch.pi) - torch.pi
    q2_alt = (q2_alt + torch.pi) % (2 * torch.pi) - torch.pi

    q = torch.stack([q1, q2], dim=-1)
    q_alt = torch.stack([q1_alt, q2_alt], dim=-1)

    # Check whether the primary angle is clockwise. Otherwise, swap with secondary.
    q_cw = q_alt
    q_ccw = q

    if is_clockwise:
        q_out = q_cw
    else:
        q_out = q_ccw
    
    return q_out


def wrap_to_pi(q):
    return (q + torch.pi) % (2 * torch.pi) - torch.pi

def shift_q(q, clockwise = False):
    with torch.no_grad():
        if clockwise:
            shift_mask = (q[:, 0] < 0) & (q[:, 1] > 0)
            q[shift_mask, 1] -= 2 * torch.pi
        else:
            shift_mask = (q[:, 0] > 0) & (q[:, 1] < 0)
            q[shift_mask, 1] += 2 * torch.pi 
    return q

def check_clockwise(q):
    if (q[1] >= q[0] and q[1] <= q[0] + torch.pi) or (q[1] >= q[0] - 2 * torch.pi and q[1] <= q[0] - torch.pi):
        clockwise = False
    else:
        clockwise = True
    return clockwise


# Function to flip joint angles to their (counter)clockwise equivalent, depending on input "flilp_to_cw".
def flip_q(rp, q, flip_to_cw):
    pos, _ = forward_kinematics(rp, q)
    q_flipped = inverse_kinematics(pos, rp, is_clockwise=flip_to_cw)
    if torch.allclose(q, q_flipped):
        print("WARNING: q flipped but retained same value, did you select the right orientation?")
    return q_flipped

# Function to flip joint velocities to their (counter)clockwise equivalent, depending on input "flilp_to_cw".
def flip_q_d(rp, q, q_d, flip_to_cw):
    pos, _ = forward_kinematics(rp, q)
    ik_partial = partial(inverse_kinematics, rp = rp, is_clockwise=flip_to_cw)
    J = torch.autograd.functional.jacobian(ik_partial, inputs=pos)
    q_d = (J @ q_d.T).T
    return q_d
