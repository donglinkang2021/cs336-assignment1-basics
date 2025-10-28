import torch
import torch.nn.functional as F

x1 = torch.randn(4,2)
x2 = torch.randn(3,2)

# M = torch.randn(4, 3)  # random matrix
M = x1 @ x2.T
U, S, Vh = torch.linalg.svd(M)

print(U, S, Vh)

# take the first pair of singular vectors
sigma1 = S[0]
u1 = U[:, 0]
v1 = Vh[0, :]

# optimal rank-1 approximation
M_approx = sigma1 * u1.unsqueeze(-1) @ v1.unsqueeze(0)

print("Original matrix M:\n", M)
print("Approximate matrix M_approx:\n", M_approx)
print("Error:", torch.norm(M - M_approx))
print("MSE:", F.mse_loss(M_approx, M))
