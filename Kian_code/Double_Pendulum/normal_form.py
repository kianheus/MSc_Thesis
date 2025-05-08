import torch


def calculate_Y(th, th_d, M_th, G_th, device):


    M, G = M_th, G_th
  
    M0 = M[0,0]
    M1 = M[1,1]

    # Define used Coriolis terms as such, since we assume that diagonal terms of M
    # are only a function of their own DoF.
    C0 = 0.5 * torch.autograd.grad(M0, th, create_graph=True)[0][0, 0] * th_d[0, 0]
    C1 = 0.5 * torch.autograd.grad(M1, th, create_graph=True)[0][0, 1] * th_d[0, 1]

    G0 = G[0]
    G1 = G[1]

	
    dM1dth1 = torch.autograd.grad(M1, th, create_graph=True)[0][0,1]
    ddM1ddth1 = torch.autograd.grad(dM1dth1, th, create_graph=True)[0][0,1]
    dG1dth0 = torch.autograd.grad(G1, th, create_graph=True)[0][0,0]
    dG1dth1 = torch.autograd.grad(G1, th, create_graph=True)[0][0,1]

    if False:
        print("dM1dth1:", dM1dth1)
        print("ddM1ddth1:", ddM1ddth1)
        print("dG1dth0:", dG1dth0)
        print("dG1dth1:", dG1dth1)
        print("C0:", C0)
        print("C1:", C1)
        print("G0:", G0)
        print("G1:", G1)
        print("M0:", M0)
        print("M1:", M1)
    


    #THIS CALCULATION HAS BEEN VERIFIED ANALYTICALLY WITH WOLFRAM MATHEMATICA
    y = th[0,1]
    y_i = th_d[0,1]
    y_ii = 1/M1 * (-C1 * th_d[0,1] - G1)
    y_iii = 1/(M1**2) * (1/2 * dM1dth1**2 * th_d[0,1]**3 + G1 * dM1dth1 * th_d[0,1]) - \
            1/M1 * (1/2 * ddM1ddth1 * th_d[0,1]**3 + dM1dth1 * th_d[0,1] * y_ii + dG1dth0 * th_d[0,0] + dG1dth1 * th_d[0,1])
    
    if False:
        print("y_ii:", y_ii.item())
        print("y_iii:", y_iii.item())


    Y = torch.tensor([[y],
                      [y_i],
                      [y_ii],
                      [y_iii]]).to(device)
    
    return Y

def calculate_Y_inverse(Y, M_th, C0, C1, G0, G1, device):

    th1 = Y[0,0]
    th1_d = Y[1,0]
    

def calculate_alpha_beta(th, th_d, M_th, G_th, A_th, Y):

    M, G, A = M_th, G_th, A_th

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
    """
    ddG1ddth0 = torch.autograd.grad(dG1dth0, th, create_graph=True)[0][0,0]
    ddG1dth0dth1 = torch.autograd.grad(dG1dth0, th, create_graph=True)[0][0,1]
    ddG1ddth1 = torch.autograd.grad(dG1dth1, th, create_graph=True)[0][0,1]
    """
    
    try:
        ddG1ddth0 = torch.autograd.grad(dG1dth0, th, create_graph=True)[0][0,0]
    except RuntimeError:
        print("Second derivative ddG1ddth0 does not require grad — setting to zero.")
        ddG1ddth0 = torch.zeros_like(dG1dth0)

    try:
        ddG1dth0dth1 = torch.autograd.grad(dG1dth0, th, create_graph=True)[0][0,1]
    except RuntimeError:
        print("Mixed derivative ddG1dth0dth1 does not require grad — setting to zero.")
        ddG1dth0dth1 = torch.zeros_like(dG1dth0)

    try:
        ddG1ddth1 = torch.autograd.grad(dG1dth1, th, create_graph=True)[0][0,1]
    except RuntimeError:
        print("Second derivative ddG1ddth1 does not require grad — setting to zero.")
        ddG1ddth1 = torch.zeros_like(dG1dth1)
    

    if False:
        print("dM0dth0:", dM0dth0.item())
        print("dM1dth1:", dM1dth1.item())
        print("ddM1ddth1:", ddM1ddth1.item())
        print("dddM1dddth1:", dddM1dddth1.item())
        print("A0:", A[0].item())
        print("A1:", A[1].item())
        print("G0:", G0.item())
        print("G1:", G1.item())
        print("dG1dth0:", dG1dth0.item())
        print("dG1dth1:", dG1dth1.item())
        print("ddG1ddth0:", ddG1ddth0.item())
        print("ddG1dth0dth1:", ddG1dth0dth1.item())
        print("ddG1ddth1:", ddG1ddth1.item())
    

    AA = M1**(-2)
    BB = 1/2 * th_d[0,1]**3 * dM1dth1**2
    CC = th_d[0,1] * dM1dth1 * G1
    DD = M1**(-1)
    EE = 1/2 * th_d[0,1]**3 * ddM1ddth1
    FF = th_d[0,1] * y_ii * dM1dth1
    GG = th_d[0,0] * dG1dth0
    HH = th_d[0,1] * dG1dth1

    dAAdt = -2 * th_d[0,1] * dM1dth1 / (M1**3)
    dBBdt = 3/2 * th_d[0,1]**2 * y_ii * dM1dth1**2 + th_d[0,1]**4 * dM1dth1 * ddM1ddth1
    dCCdt = th_d[0,1] * dM1dth1 * (th_d[0,0] * dG1dth0 + th_d[0,1] * dG1dth1) + th_d[0,1]**2 * ddM1ddth1*G1 + y_ii * dM1dth1 * G1
    dDDdt = - th_d[0,1] * dM1dth1 / (M1**2)
    dEEdt = 1/2 * dddM1dddth1 * th_d[0,1]**4 + 3/2 * th_d[0,1]**2 * y_ii * ddM1ddth1
    dFFdt = th_d[0,1]**2 * y_ii * ddM1ddth1 + y_ii**2 * dM1dth1 + y_iii * th_d[0,1] * dM1dth1
    dGGdt_pas = th_d[0,0] * (th_d[0,0] * ddG1ddth0 + th_d[0,1] * ddG1dth0dth1) + M0**(-1) *(- 0.5 * dM0dth0 * th_d[0,0]**2 - G0) * dG1dth0
    dGGdt_act = M0**(-1) * A[0] * dG1dth0
    #dGGdt = dGGdt_pas + dGGdt_act
    dHHdt = y_ii * dG1dth1 + th_d[0,1] * (th_d[0,0] * ddG1dth0dth1 + th_d[0,1] * ddG1ddth1)

    

    alpha = dAAdt * (BB + CC) + AA * (dBBdt + dCCdt) - dDDdt * (EE + FF + GG + HH) - DD * (dEEdt + dFFdt + dGGdt_pas + dHHdt)
    beta = - DD * dGGdt_act

    if False: #alpha > 4.5:
        print("yii:", y_ii.item())
        print("y_iii:", y_iii.item())
        print("AA", AA.item())
        print("BB", BB.item())
        print("CC", CC.item())
        print("DD", DD.item())
        print("EE", EE.item())
        print("FF", FF.item())
        print("GG", GG.item())
        print("HH", HH.item())

        print("dAAdt", dAAdt.item())
        print("dBBdt", dBBdt.item())
        print("dCCdt", dCCdt.item())
        print("dDDdt", dDDdt.item())
        print("dEEdt", dEEdt.item())
        print("dFFdt", dFFdt.item())
        #print("dGGdt", dGGdt.item())
        print("dGGdt_pas", dGGdt_pas.item())
        print("dGGdt_act", dGGdt_act.item())
        print("dHHdt", dHHdt.item())

        print("Block 1:", (dAAdt * (BB + CC)).item())
        print("Block 2:", (AA * (dBBdt + dCCdt)).item())
        print("Block 3:", (-dDDdt * (EE + FF + GG + HH)).item())
        print("Block 4:", (- DD * (dEEdt + dFFdt + dGGdt_pas + dHHdt)).item())


    return alpha, beta


def calculate_v(Y, Y_des, K):
    
    v = torch.matmul(K, Y_des - Y)

    return v


def calculate_u(alpha, beta, v):
    
    u = 1/beta * (-alpha + v)
    
    return u


def calculate_y_iv(alpha, beta, u):
    
    y_iv = alpha + beta * u

    return y_iv

def check_stable_gains(K):

    K = K.squeeze(0)
    stable = True
    for i in range(4):
        print("K[" + str(i) + "] =", round(K[i].item(), 11))
        if K[i] <= 0:
            stable = False
            print("Unstable gains because K[" + str(i) + "] is <= 0.")
    print("\nFor K3 * K2 > K1 criterion:")
    print("K3 * K2 =", round((K[3] * K[2]).item(), 11))
    print("K1 = ", round((K[1]).item(), 11))

    print("\nFor K3 * K2 * K1 > K1**2 + K3**2 * K1 criterion:")
    print("K3 * K2 * K1 =", round((K[3] * K[2] * K[1]).item(), 11))
    print("K1**2 =", round((K[1]**2).item(), 11))
    print("K3**2 * K0 =", round((K[3]**2 * K[0]).item(), 3), "\n")
    if K[3] * K[2] < K[1]:
        stable = False
        print("Unstable because K[3] * K[2] < K[1]")
    if K[3] * K[2] * K[1] < K[1]**2 + K[3]**2 * K[0]:
        stable = False
        print("Unstable because K[3] * K[2] * K[1] < K[1]**2 + K[3]**2 * K[0]")
        print(K[3] * K[2] * K[1] - K[1]**2 + K[3]**2 * K[0])
    return stable