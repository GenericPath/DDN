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
        self.constant = 1e-5
        
    def objective(self, x_batch, y):
        """
        f(W,y) = y^T * (D-W) * y / y^T * D * y
        """
        for i in range(len(x_batch)): # code in ddn-3 for the torch.einsum version of this, may switch for efficiency later...
            x = x_batch[i][0].add(self.constant)
            y_i = y[i][0].add(self.constant)#.flatten()
            # W is an NxN symmetrical matrix with W(i,j) = w_ij
            D = x.sum(1).diag() # D is an NxN diagonal matrix with d on diagonal, for d(i) = sum_j(w(i,j))
            L = D - x

            y_t = torch.t(y_i)
            x_batch[i][0] = torch.div(torch.matmul(torch.matmul(y_t, L),y_i),torch.matmul(torch.matmul(y_t,D),y_i))
        return x_batch
    
    def equality_constraints(self, x_batch, y_batch):
        """
        subject to y^T * D * 1 = 0
        """
        for i in range(len(x_batch)):
            x = x_batch[i][0].add(self.constant)
            y = y_batch[i][0].add(self.constant)#.flatten()
            # Ensure correct size and shape of y... scipy minimise flattens y         
            N = x.size(dim=0)
            
            #x is an NxN symmetrical matrix with W(i,j) = w_ij
            D = x.sum(1).diag() # D is an NxN diagonal matrix with d on diagonal, for d(i) = sum_j(w(i,j))
            ONE = torch.ones(N,1)   # Nx1 vector of all ones
            y_t = torch.t(y)
            x_batch[i][0] = torch.matmul(torch.matmul(y_t,D), ONE)
        return x_batch

    def solve(self, W_batch):
        # W_batch = torch.squeeze(W_batch, 1)
        output = torch.zeros(32,1,1024) #, dtype=torch.float32, requires_grad=True)
        for i in range(len(W_batch)):
            W = torch.add(W_batch[i,0], self.constant) # Each batch is passed as [batch, channels, width, height], add a small constant to avoid NaN

            D = torch.diag(torch.sum(W, 0))
            D_half_inv = torch.diag(1.0 / torch.sqrt(torch.sum(W, 0)))
            M = torch.matmul(D_half_inv, torch.matmul((D - W), D_half_inv))

            # M is the normalised laplacian

            (w, v) = torch.linalg.eigh(M)
            # TODO: Operate eigh across batch dimensions (it only operates on the last two dimensions...)
            # TODO: ... this means changing everything into einsum operations! this way it will all work?
            # TODO: because the without batch version has grad_fn with eigh as part of it.. therefore it should probably work!

            #find index of second smallest eigenvalue
            index = torch.argsort(w)[1] # arg sort not neccessary, as this eigh returns sorted already...

            v_partition = v[:, index]
            # instead of the sign of a digit being the binary split, let the NN learn it
            # v_partition = torch.sign(v_partition)
        
            # return the eigenvector and a blank context
            output[i, 0] = v_partition#.view(32,32)
        return output, None
    
    def test(self, x, y):
        """ Test gradient """
        # Evaluate objective function at (xs,y):
        f = torch.enable_grad()(self.objective)(x, y=y) # b

        # Compute partial derivative of f wrt y at (xs,y):
        fY = grad(f, y, grad_outputs=torch.ones_like(f), create_graph=True)
        return fY