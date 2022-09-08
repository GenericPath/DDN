import torch
from torch.autograd import grad
from torch.autograd import gradcheck

from node import *
from nc import NormalizedCuts

import numpy as np

# similar to https://github.com/anucvml/ddn/blob/master/tests/testPnPNode.py

def is_hermitian_positive_semidefinite(X):
    if X.shape[0] != X.shape[1]: # must be a square matrix
        print("not square")
        return False

    if not torch.all( X - X.H == 0 ): # must be a symmetric or hermitian matrix
        print("not symmetric or hermitian")
        return False

    try: # Cholesky decomposition fails for matrices that are NOT positive definite.

        # But since the matrix may be positive SEMI-definite due to rank deficiency
        # we must regularize.
        regularized_X = X + torch.eye(X.shape[0]) * 1e-14

        np.linalg.cholesky(regularized_X)
    except np.linalg.LinAlgError:
        print("not positive semi-definite")
        return False

    return True

# generate data
W = torch.randn(32,1024,1024, requires_grad=False, device='cpu') # as if it was a batch of 32 32x32 rgb images
W = torch.nn.functional.relu(W) # enforce positive constraint
W_t = torch.einsum('bij->bji', W) # transpose of batched matrix
W = torch.matmul(W, W_t) # to create a positive semi-definite matrix

# check properties of W
one_instance = W[0]
is_hermitian_positive_semidefinite(one_instance)

# Create node
node = NormalizedCuts()
DL = DeclarativeLayer(node)

W = W.requires_grad_()

# forward pass
y = DL(W)

# compute objective function
f = node.objective(W,y)

# compute gradient
Dy = grad(y, (W), grad_outputs=torch.ones_like(y))

# run gradcheck
test = gradcheck(DL, (W), eps=1e-4, atol=1e-4, rtol=1e-4, raise_exception=True)
print(f'Gradcheck passed {test}')
