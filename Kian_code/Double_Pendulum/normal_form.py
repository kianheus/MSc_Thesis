import torch


def calculate_Y(th, th_d, M_th, C_th, G_th, device):
    
    M, C, G = M_th, C_th, G_th
  
    M0 = M[0,0]
    M1 = M[1,1]

    # Define used Coriolis terms as such, since we assume that diagonal terms of M
    # are only a function of their own DoF.
    C0 = 0.5 * torch.autograd.grad(M0, th, create_graph=True)[0][0, 0] * th_d[0, 0]
    C1 = 0.5 * torch.autograd.grad(M1, th, create_graph=True)[0][0, 1] * th_d[0, 1]

    G0 = G[0]
    G1 = G[1]

	
    dM0dth0 = torch.autograd.grad(M0, th, create_graph=True)[0][0,0]
    dM1dth1 = torch.autograd.grad(M1, th, create_graph=True)[0][0,1]
    ddM1ddth1 = torch.autograd.grad(dM1dth1, th, create_graph=True)[0][0,1]
    dddM1dddth1 = torch.autograd.grad(ddM1ddth1, th, create_graph=True)[0][0,1]
    dG1dth0 = torch.autograd.grad(G1, th, create_graph=True)[0][0,0]
    dG1dth1 = torch.autograd.grad(G1, th, create_graph=True)[0][0,1]
    ddG1ddth0 = torch.autograd.grad(dG1dth0, th, create_graph=True)[0][0,0]
    ddG1dth0dth1 = torch.autograd.grad(dG1dth0, th, create_graph=True)[0][0,1]
    ddG1ddth1 = torch.autograd.grad(dG1dth1, th, create_graph=True)[0][0,1]
    
    print("\n")
    print("dM0dth0:", dM0dth0)
    print("dM1dth1", dM1dth1)
    print("ddM1ddth1", ddM1ddth1)
    print("dddM1dddth1", dddM1dddth1)
    print("dG1dth0", dG1dth0)
    print("dG1dth1", dG1dth1)
    print("ddG1ddth0", ddG1ddth0)
    print("ddG1dth0dth1", ddG1dth0dth1)
    print("ddG1ddth1", ddG1ddth1)
    print("\n")

    y = th[0,1]
    y_i = th_d[0,1]
    y_ii = 1/M1 * (-C1 * th_d[0,1] - G1)
    y_iii = 1/(M1**2) * (1/2 * dM1dth1**2 * th_d[0,1]**3 + G1 * dM1dth1 * th_d[0,1]) - \
            1/M1 * (1/2 * ddM1ddth1 * th_d[0,1]**3 + dM1dth1 * th_d[0,1] * y_ii + dG1dth0 * th_d[0,0] + dG1dth1 * th_d[0,1])

    Y = torch.tensor([[y],
                      [y_i],
                      [y_ii],
                      [y_iii]]).to(device)
    
    return Y


def calculate_alpha_beta(th, th_d, M_th, C_th, G_th, A_th, Y):

    M, C, G, A = M_th, C_th, G_th, A_th

    y_ii = Y[2, 0]
    y_iii = Y[3, 0]
  
    M0 = M[0,0]
    M1 = M[1,1]

    # Define used Coriolis terms as such, since we assume that diagonal terms of M
    # are only a function of their own DoF. 
    C0 = 0.5 * torch.autograd.grad(M0, th, create_graph=True)[0][0, 0] * th_d[0, 0]
    C1 = 0.5 * torch.autograd.grad(M1, th, create_graph=True)[0][0, 1] * th_d[0, 1]

    G0 = G[0]
    G1 = G[1]    

    dM0dth0 = torch.autograd.grad(M0, th, create_graph=True)[0][0,0]
    dM1dth1 = torch.autograd.grad(M1, th, create_graph=True)[0][0,1]
    ddM1ddth1 = torch.autograd.grad(dM1dth1, th, create_graph=True)[0][0,1]
    dddM1dddth1 = torch.autograd.grad(ddM1ddth1, th, create_graph=True)[0][0,1]
    dG1dth0 = torch.autograd.grad(G1, th, create_graph=True)[0][0,0]
    dG1dth1 = torch.autograd.grad(G1, th, create_graph=True)[0][0,1]
    ddG1ddth0 = torch.autograd.grad(dG1dth0, th, create_graph=True)[0][0,0]
    ddG1dth0dth1 = torch.autograd.grad(dG1dth0, th, create_graph=True)[0][0,1]
    ddG1ddth1 = torch.autograd.grad(dG1dth1, th, create_graph=True)[0][0,1]

    alpha = 1/(M1**3) * (dM1dth1**3 * th_d[0,1]**4 + 2 * G1 * dM1dth1**2 * th_d[0,1]**2) + \
           1/(M1**2) * (ddM1ddth1 * dM1dth1 * th_d[0,1]**3 + 3/2 * dM1dth1 ** 2 * th_d[0,1]**2 * y_ii +
                        dG1dth0 * dM1dth1 * th_d[0,0] * th_d[0,1] + dG1dth1 * dM1dth1 * th_d[0,1]**2 + 
                        G1 * ddM1ddth1 * th_d[0,1]**2 + G1 * dM1dth1 * y_ii) + \
           (dM1dth1 * th_d[0,1])/(M1**2) * (1/2 * ddM1ddth1 * th_d[0,1]**3 + dM1dth1 * th_d[0,1] * y_ii + 
                                          dG1dth0 * th_d[0,0] + dG1dth1 * th_d[0,1]) - \
           1/M1 * (1/2 * dddM1dddth1 * th_d[0,1]**4 + 5/2 * ddM1ddth1 * th_d[0,1]**2 * y_ii +
                   dM1dth1 * y_ii**2 + dM1dth1 * th_d[0,1] * y_iii + 
                   ddG1ddth0 * th_d[0,0]**2 + ddG1dth0dth1 * th_d[0,0] * th_d[0,1] + dG1dth0 * 1/M0 * (- 1/2 * dM0dth0 * th_d[0,0]**2 - G0) + 
                   ddG1dth0dth1 * th_d[0,0] * th_d[0,1] + ddG1ddth1 * th_d[0,1]**2 + dG1dth1 * y_ii)

    beta = - 1/M1 * dG1dth0 * 1/M0 * A[0]

    return alpha, beta


def calculate_v(Y, Y_des, K):
    
    v = torch.matmul(K, Y-Y_des)

    return v


def calculate_u(alpha, beta, v):
    
    u = 1/beta * (-alpha + v)
    
    return u


def calculate_y_iv(alpha, beta, u):
    
    y_iv = alpha + beta * u

    return y_iv


    """
    y_iv = 1/(M1**3) * (dM1dth1**3 * th_d[1]**4 + 2 * G1 * dM1dth1**2 * th_d[1]**2) + \
           1/(M1**2) * (ddM1ddth1 * dM1dth1 * th_d[1]**3 + 3/2 * dM1dth1 ** 2 * th_d[1]**2 * y_ii +
                        dG1dth0 * dM1dth1 * th_d[0] * th_d[1] + dG1dth1 * dM1dth1 * th_d[1]**2 + 
                        G1 * ddM1ddth1 * th_d[1]**2 + G1 * dM1dth1 * y_ii) + \
           (dM1dth1 * th_d[1])/(M1**2) * (1/2 * ddM1ddth1 * th_d[1]**3 + dM1dth1 * th_d[1] * y_ii + 
                                          dG1dth0 * th_d[0] + dG1dth1 * th_d[1]) - \
           1/M1 * (1/2 * dddM1dddth1 * th_d[1]**4 + 5/2 * ddM1ddth1 * th_d[1]**2 * y_ii +
                   dM1dth1 * y_ii**2 + dM1dth1 * th_d[1] * y_iii + 
                   ddG1ddth0 * th_d[0]**2 + ddG1dth0dth1 * th_d[0] * th_d[1] + dG1dth0 * th_dd0 + 
                   ddG1dth0dth1 * th_d[0] * th_d[1] + ddG1ddth1 * th_d[1]**2 + dG1dth1 * y_ii)
    """