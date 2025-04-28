import torch

# Number of samples
n_samples = 100000

# Data range
q0_low  = -torch.pi
q0_high = torch.pi
q1_low  = -torch.pi
q1_high = 2*torch.pi

# Generate uniformly distributed points for q1 and q2
q0 = q0_low + (q0_high - q0_low) * torch.rand(n_samples)
q1 = q1_low + (q1_high - q1_low) * torch.rand(n_samples)

# Stack q1 and q2 to get the 2D coordinates
points = torch.stack([q0, q1], axis=1)