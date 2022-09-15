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
W = torch.randn(2,64,64, requires_grad=False, device='cpu') # as if it was a batch of 32 32x32 rgb images
W = torch.nn.functional.relu(W) # enforce positive constraint
W_t = torch.einsum('bij->bji', W) # transpose of batched matrix
W = torch.matmul(W, W_t) # to create a positive semi-definite matrix
W = W.type(torch.double)

# check properties of W
one_instance = W[0]
is_hermitian_positive_semidefinite(one_instance)

# Create node
node = NormalizedCuts()
DL = DeclarativeLayer(node)

W = W.requires_grad_()

# forward pass
y = DL(W)
y = y.type(torch.double)

# compute objective function
f = node.objective(W,y)
eq1 = node.equality_constraints(W,y)
print(f'f\n{f}')
print(f'eq1 {eq1}')

# compute gradient
# Dy = grad(y, (W), grad_outputs=torch.ones_like(y))

# run gradcheck
# test = gradcheck(DL, (W), eps=1e-4, atol=1e-4, rtol=1e-4, raise_exception=False)
# print(f'Gradcheck passed {test}')


# maybe try the actual solution thing
# inf thing from paper?




# y = y.flatten(-2) # 2, 64
# # y = # make it a column vector
# print(y.shape)

# d = W.sum(1) # Row sum: where axis 0 is batch, 1 is row, 2 is col
# D = torch.diag_embed(d)
# print(D.shape)

# ONE = torch.ones_like(y)


# eqconst = torch.matmul(y, D)
# print(eqconst.shape)
# print(eqconst)

# y = y.flatten(-2) # converts to the vector with shape = (32, 1, N) 
# b, c, N = y.shape
# y = y.reshape(b,c,N) # convert to a col vector (y^T)

# d = W.sum(0) # eqv to x.sum(0) --- d vector
# D = torch.diag_embed(d).to(device=y.device) # D = matrix with d on diagonal
# ONE = torch.ones_like(y).to(device=y.device) # create a vector of ones
# fConst = torch.einsum('...IK,...KJ->...IJ', torch.einsum('...IK,...KJ->...IJ',y, D), ONE).squeeze(-2)
# print(fConst)

# print(y.shape)
# a,b = y.shape
# y = y.reshape(a,1,b)
# ONE = torch.ones((a,b,1), dtype=torch.double)

# b,1,N
# b,N,N

# print(f'y shape: {y.shape}')
# print(f'D shape: {D.shape}')
# eqconst = torch.bmm(y,D)
# print(f'eqconst shape: {eqconst.shape}')
# print(eqconst)

# eqconst2 = torch.bmm(eqconst, ONE)
# print(eqconst2.shape)
# print(eqconst2)



# y2 = torch.sign(y)
# print(f'y2: {y2}')
# f2 = node.objective(W,y2)
# print(f'f2: {f2}')
# eqconst3 = torch.bmm(y2,D)
# eqconst3 = torch.bmm(eqconst3, ONE)
# print(f'eqconst3: {eqconst3}')

