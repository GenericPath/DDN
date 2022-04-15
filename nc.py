from ddn.pytorch.node import *
import torch
import numpy as np
from PIL import Image
import cv2

class NormalizedCuts(EqConstDeclarativeNode):
    """
    A declarative node to embed Normalized Cuts into a Neural Network
    
    Normalized Cuts and Image Segmentation https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf
    Shi, J., & Malik, J. (2000)
    """
    def __init__(self, chunk_size=None):
        super().__init__(chunk_size=chunk_size) # input is divided into chunks of at most chunk_size
        
    def objective(self, x, y):
        """
        f(W,y) = y^T * (D-W) * y / y^T * D * y

        output: shape - (b, c, x)
        """
        y = y.flatten(-2) # converts to the vector with shape = (32, 1, N) 
        b, c, N = y.shape
        y = y.reshape(b,c,1,N) # convert to a col vector

        d = torch.einsum('bcij->bcj', x) # eqv to x.sum(0) --- d vector
        D = torch.diag_embed(d) # D = matrix with d on diagonal

        L = D-x

        return torch.div(
            torch.einsum('bcij,bckj->bcik', torch.einsum('bcij,bckj->bcik', y, L), y),
            torch.einsum('bcij,bckj->bcik', torch.einsum('bcij,bckj->bcik', y, D), y)
        ).squeeze(-2)

    
    def equality_constraints(self, x, y):
        """
        y = solution matrix - shape: (b,c,x,y)
        x = input affinity/weight matrix - shape: (b,c,N,N) for N = x*y
        subject to y^T * D * 1 = 0

        output: size (b, c, x)
        """
        y = y.flatten(-2) # converts to the vector with shape = (32, 1, N) 
        b, c, N = y.shape
        y = y.reshape(b,c,1,N) # convert to a col vector

        d = torch.einsum('bcij->bcj', x) # eqv to x.sum(0) --- d vector
        D = torch.diag_embed(d) # D = matrix with d on diagonal
        ONE = torch.ones(b,c,1,N) # create a col vector of ones - shape = (32, 1, 1, N)

        # No need to transpose y, as the vector multiplication will be the same for 1D tensors

        # return the constraint calculation, squeezed to output size
        return torch.einsum('bcij,bckj->bcik', torch.einsum('bcij,bckj->bcik', y, D), ONE).squeeze(-2)

    def solve(self, A):
        """ TODO: pass a parameter to avoid hardcoded output dimensions """
        out_size = 32
        # Implementation notes:
        # - requires einsum's to act on batch. Otherwise torch complains about tensors not being in graph being differentiated
        # - D_inv_sqrt calculates the inv sqrt of diagonal only to avoid division by zero

        # obtain the batch and image size (sqrt of x/y to get out_size?)
        b,c,x,y = A.shape

        # can also replace bc with ...
        d = torch.einsum('bcij->bcj', A) # eqv to A.sum(0) --- d vector
        D = torch.diag_embed(d) # D = matrix with d on diagonal
        D_inv_sqrt = torch.diag_embed(d.pow(-0.5)) # Don't calculate inverse sqrt of 0 entries (non diagonals)

        L = (D-A) # Laplacian matrix
        # The symmetrically normalized laplacian can be calculated as D^-0.5 * L * D^-0.5 or eqv. I - D^-0.5 * A * D^-0.5 
        L_norm = torch.einsum('bcij,bcjk->bcik', torch.einsum('bcij,bcjk->bcik', D_inv_sqrt , L) , D_inv_sqrt)

        # Solve eigenvectors and eigenvalues
        (w, v) = torch.linalg.eigh(L_norm)
        
        # Returns the second smallest eigenvector (and possibly more if num_eigs > 1)
        # eigenvector(s) reshaped to match original image size
        num_eigs = 1
        return v.narrow(-2, 1, num_eigs).squeeze(1).reshape(b, num_eigs, out_size, out_size), None
    
    def test(self, x, y):
        """ Test gradient """
        # Evaluate objective function at (xs,y):
        f = torch.enable_grad()(self.objective)(x, y=y) # b

        # Compute partial derivative of f wrt y at (xs,y):
        fY = grad(f, y, grad_outputs=torch.ones_like(f), create_graph=True)
        return fY