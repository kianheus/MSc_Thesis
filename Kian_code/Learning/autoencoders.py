import torch
import torch.nn as nn
from functools import partial

import Double_Pendulum.Lumped_Mass.transforms as transforms

"""
This file contains the architectures of a number of Autoencoders used in my thesis. 
Currently it only contains a single, set architecture for learning both h1 and h2, 
however, this can be extended and made modular depending on __init__ conditions.
"""


class Autoencoder_double(nn.Module):
    def __init__(self, rp):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(2, 16),
            nn.Softplus(),
            nn.Linear(16, 16),
            nn.Softplus(),
            nn.Linear(16, 2)
        )
        
        
        self.dec = nn.Sequential(
            nn.Linear(2, 16),
            nn.Softplus(),
            nn.Linear(16, 16),
            nn.Softplus(),
            nn.Linear(16, 2)
        )
        
        self.rp = rp
        
    def encoder_theta_1_ana(self, q):
        theta_1 = transforms.analytic_theta_1(self.rp, q).unsqueeze(0)
        return theta_1, theta_1
    
    #This function is not used in the forward pass, but is useful for comparing learned to analytic theta_2
    def encoder_theta_2_ana(self, q):
        theta_2 = transforms.analytic_theta_2(self.rp, q).unsqueeze(0)
        return theta_2, theta_2
    
    def theta_ana(self, q):
        theta_ana = torch.vmap(transforms.analytic_theta, in_dims=(None, 0))(self.rp, q)
        #theta_1_ana, _ = torch.vmap(self.encoder_theta_1_ana)(q)
        #theta_2_ana, _ = torch.vmap(self.encoder_theta_2_ana)(q)
        #theta_ana = torch.cat((theta_1_ana, theta_2_ana), dim=1)
        return theta_ana
    
    def encoder(self, q):
        theta = self.enc(q)
        return theta
    
    def decoder(self, theta, clockwise=None):
        q_hat = self.dec(theta)
        return q_hat

    def encoder_vmap(self, q):
        theta = self.encoder(q)
        return theta
    
    def decoder_vmap(self, theta, clockwise=None):
        q_hat = self.decoder(theta)
        return q_hat
    
    def jacobian_enc(self, q, clockwise=None):
        J_h = torch.vmap(torch.func.jacfwd(self.encoder_vmap, has_aux=False))(q)
        return J_h.squeeze(0)

    def jacobian_dec(self, theta, clockwise=None):
        J_h_dec = torch.vmap(torch.func.jacfwd(self.decoder_vmap, has_aux=False))(theta)
        return J_h_dec.squeeze(0)
    
    def forward(self, q):
        
        J_h_1_ana, theta_1_ana = torch.vmap(torch.func.jacfwd(self.encoder_theta_1_ana, has_aux=True))(q)
        J_h_2_ana, theta_2_ana = torch.vmap(torch.func.jacfwd(self.encoder_theta_2_ana, has_aux=True))(q)
        J_h_ana = torch.cat((J_h_1_ana, J_h_2_ana), dim=1).float()
        
        theta = self.encoder(q)
        J_h = self.jacobian_enc(q) 
        #J_h = torch.vmap(torch.func.jacfwd(self.encoder_vmap, has_aux=False))(q)
        q_hat = self.decoder(theta)
        J_h_dec = self.jacobian_dec(theta)
        #J_h_dec, q_hat = torch.vmap(torch.func.jacfwd(self.decoder_vmap, has_aux=True))(theta)

        return(theta, J_h, q_hat, J_h_dec, J_h_ana)
    

class Analytic_transformer():

    """
    This class follows the same notation as the Autoencoders, but provides the analytic solution instead. 
    """

    def __init__(self, rp):
        self.rp = rp


    def encoder(self, q):
        theta_ana = transforms.analytic_theta(self.rp, q)
        return theta_ana
    
    def decoder(self, theta, clockwise):
        q_ccw, q_cw = torch.vmap(transforms.analytic_inverse, in_dims=(None, 0))(self.rp, theta)
        if clockwise:
            return q_cw
        else:
            return q_ccw
        
    def decoder_cw(self, theta):
        q_ccw, q_cw = transforms.analytic_inverse(self.rp, theta)
        return q_cw
    
    def decoder_ccw(self, theta):
        q_ccw, q_cw = transforms.analytic_inverse(self.rp, theta)
        return q_ccw
    
    def encoder_theta_1_ana(self, q):
        theta_1 = transforms.analytic_theta_1(self.rp, q).unsqueeze(0)
        return theta_1, theta_1
    
    #This function is not used in the forward pass, but is useful for comparing learned to analytic theta_2
    def encoder_theta_2_ana(self, q):
        theta_2 = transforms.analytic_theta_2(self.rp, q).unsqueeze(0)
        return theta_2, theta_2
    
    def theta_ana(self, q):
        theta_ana = torch.vmap(transforms.analytic_theta, in_dims=(None, 0))(self.rp, q)
        return theta_ana

    def encoder_vmap(self, q):
        theta = torch.vmap(self.encoder)(q)
        return theta
    
    def decoder_vmap(self, theta, clockwise):
        if clockwise:
            q_hat = torch.vmap(self.decoder_cw)(theta)
            return q_hat
        else:
            q_hat = torch.vmap(self.decoder_ccw)(theta)
            return q_hat

    def jacobian_enc(self, q, clockwise=None):
        J_h = torch.vmap(torch.func.jacfwd(self.encoder, has_aux=False))(q)
        return J_h.squeeze(0).float()

    def jacobian_dec(self, theta, clockwise=None):
        if clockwise:
            J_h_dec = torch.vmap(torch.func.jacfwd(self.decoder_cw, has_aux=False))(theta)
        else:
            J_h_dec = torch.vmap(torch.func.jacfwd(self.decoder_ccw, has_aux=False))(theta)
        return J_h_dec.squeeze(0).float()
    

    def forward(self, q, clockwise):

        J_h_ana = self.jacobian_enc(q)

        theta = self.theta_ana(q)
        q_hat = self.decoder_vmap(theta, clockwise)
        J_h = torch.vmap(torch.func.jacfwd(self.encoder_nn, has_aux=True))(q)
        J_h_dec, q_hat = torch.vmap(torch.func.jacfwd(self.decoder_nn_cw, has_aux=True))(theta)
        print("Only taking cw decoder, are you sure you want this?")
        # Change depending on your use case

        return theta, J_h, q_hat, J_h_dec, J_h_ana
