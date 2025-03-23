import torch


def calculate_Y(th, th_d, M_th, C_th, G_th, A_th, K, device):
    
    M, C, G, A = M_th, C_th, G_th, A_th

    M0 = M[0,0]
    M1 = M[1,1]

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

    #th_dd0 = 1/M0 * (A_th[0] * u - 1/2 * dM0dth0 * th_d[0]**2 - G0)
    #TODO: Check if we can indeed omit u in this statement because we are writing in normal form

    y = th[0,1]
    y_i = th_d[0,1]
    y_ii = 1/M1 * (-C1 * th_d[0,1] - G1)
    y_iii = 1/(M1**2) * (1/2 * dM1dth1**2 * th_d[0,1]**3 + G1 * dM1dth1 * th_d[0,1]) - \
            1/M1 * (1/2 * ddM1ddth1 * th_d[0,1]**3 + dM1dth1 * th_d[0,1] * y_ii + dG1dth0 * th_d[0,0] + dG1dth1 * th_d[0,1])

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

    Y = torch.tensor([[y],
                      [y_i],
                      [y_ii],
                      [y_iii]]).to(device)

    v = torch.matmul(K, Y)

    print("v", v)
    u = 1/beta * (-alpha + v)
    print("u:", u)

    y_iv = alpha + beta * u
    
    return Y, y_iv, v, u

#def calculate_y_iv(th, th_d, M, C, G, A):

