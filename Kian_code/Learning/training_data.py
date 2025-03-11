import torch

# Number of samples
n_samples = 100000

# Data range
q1_low  = -torch.pi
q1_high = torch.pi
q2_low  = -torch.pi
q2_high = 2*torch.pi

# Generate uniformly distributed points for q1 and q2
q1 = q1_low + (q1_high - q1_low) * torch.rand(n_samples)
q2 = q2_low + (q2_high - q2_low) * torch.rand(n_samples)

# Stack q1 and q2 to get the 2D coordinates
points = torch.stack([q1, q2], axis=1)
#points = torch.stack([q1, q2, q1_d, q2_d], axis=1)

