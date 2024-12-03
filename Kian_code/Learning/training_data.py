import torch

# Number of samples
n_samples = 1000

# Data range
q1_low  = torch.pi/4
q1_high = torch.pi
q2_low  = -torch.pi/2
q2_high = torch.pi/2
q1_d_low  = -torch.pi
q1_d_high = torch.pi
q2_d_low  = -torch.pi
q2_d_high = torch.pi


# Generate uniformly distributed points for q1 and q2
q1 = q1_low + (q1_high - q1_low) * torch.rand(n_samples)
q2 = q2_low + (q2_high - q2_low) * torch.rand(n_samples)
q1_d = q1_d_low + (q1_d_high - q1_d_low) * torch.rand(n_samples)
q2_d = q2_d_low + (q2_d_high - q2_d_low) * torch.rand(n_samples)

# Stack q1 and q2 to get the 2D coordinates
points = torch.stack([q1, q2, q1_d, q2_d], axis=1)