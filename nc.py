import torch
import numpy as np
import matplotlib.pyplot as plt

# local imports
from node import *

def manual_weight(name, r=1, min=False):
    """
    I = Image name
    r = radius for connections (defaults to 4-way connection with r=1)
    """
    if type(name) == str: I = plt.imread(name)
    else: I = name
    x,y = I.shape
    
    N = x*y
    W = torch.zeros((N,N))

    I = I.flatten()

    for u in range(N):
        for v in range(N):
            if np.linalg.norm(u-v) > r: # 4-way connection
                continue
            W[u][v] = 1 if I[u] == I[v] else 0 # np.linalg.norm(I[u]-I[v])
    
    if min:
        diags = r+2
        out = torch.zeros((diags,N))
        for index, i in enumerate(range(-(diags//2), (diags//2)+1)):
            temp_diag = torch.diag(W,i)
            pad = torch.zeros(N-len(temp_diag))
            out[index] = torch.cat([temp_diag, pad])
        W = out
    return W

def de_minW(out):
    """
    Returns the reconstructed weight matrix from a smaller version.
    """
    #b,c,
    diags, N = out.shape
    reconst = torch.zeros((N,N))
    for index, i in enumerate(range(-(diags//2), (diags//2)+1)):            
        temp = torch.diag(out[b][c][index][:N-abs(i)], i)
        reconst = torch.add(reconst, temp)
    return reconst.reshape(b,c,N,N)

class NormalizedCuts(AbstractDeclarativeNode):
    """
    A declarative node to embed Normalized Cuts into a Neural Network
    
    Normalized Cuts and Image Segmentation https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf
    Shi, J., & Malik, J. (2000)
    """
    def __init__(self, chunk_size=None, eps=1e-12):
        super().__init__(chunk_size=chunk_size, eps=eps) # input is divided into chunks of at most chunk_size
        
    def objective(self, x, y):
        """
        f(W,y) = y^T * (D-W) * y / y^T * D * y

        Arguments:
            y: (b, c, x, y) Torch tensor,
                batch, channels of solution tensors

            x: (b, c, N, N) Torch tensor,
                batch, channels of affinity/weight tensors (N = x * y)        

        Return value:
            objectives: (b, c, x) Torch tensor,
                batch, channels of objective function evaluations
        """
        i,j,k,l = x.shape
        if k != l:
            # then it is a minW style weight matrix
            x = de_minW(x)

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
        subject to y^T * D * 1 = 0

        Arguments:
            y: (b, c, x, y) Torch tensor,
                batch, channels of solution tensors

            x: (b, c, N, N) Torch tensor,
                batch, channels of affinity/weight tensors (N = x * y)        

        Return value:
            equalities: (b, c, x) Torch tensor,
                batch, channel of constraint calculation scalars
        """
        i,j,k,l = x.shape
        if k != l:
            # then it is a minW style weight matrix
            x = de_minW(x)

        y = y.flatten(-2) # converts to the vector with shape = (32, 1, N) 
        b, c, N = y.shape
        y = y.reshape(b,c,1,N) # convert to a col vector (y^T)

        d = torch.einsum('bcij->bcj', x) # eqv to x.sum(0) --- d vector
        D = torch.diag_embed(d).to(device=y.device) # D = matrix with d on diagonal
        ONE = torch.ones(b,c,N,1).to(device=y.device) # create a vector of ones

        # MATRIX SIZES (for sanity check)
        # B,C,1,N
        # B,C,N,N
        # = B,C,1,N
        # B,C,N,1
        # = B,C,1,1

        # return the constraint calculation, squeezed to output size
        return torch.einsum('bcIK,bcKJ->bcIJ', torch.einsum('bcIK,bcKJ->bcIJ',y, D), ONE).squeeze(-2)

    def solve(self, A):
        """ 
        Solve the normalized cuts using eigenvectors (produces single cut, no recursion yet)

        Arguments:
            x: (b, c, N, N) Torch tensor,
                batch, channels of affinity/weight tensors (N = x * y from orignal x,y images)

        TODO: pass a parameter to avoid hardcoded output dimensions
        """
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
        # TODO: verify if the eig is the correct one... (narrowed to index 1 not index 0, but possible it already removes this one)
        return v.narrow(-2, 1, num_eigs).squeeze(1).reshape(b, num_eigs, out_size, out_size), None
    
    def test(self, x, y):
        """ Test gradient """
        # Evaluate objective function at (xs,y):
        f = torch.enable_grad()(self.objective)(x, y=y) # b

        # Compute partial derivative of f wrt y at (xs,y):
        fY = grad(f, y, grad_outputs=torch.ones_like(f), create_graph=True)
        return fY

if __name__ == "__main__":
    A = torch.randn(32,32)
    W_1 = manual_weight(A, 1, False)
    W_2 = manual_weight(A, 1, True)
    W_3 = de_minW(W_2)
    print(W_1 == W_2)


    # 1. Confirm the node can calculate a first derivative (eg. does pytorch complain about anything?)
    A = torch.randn(32,1,1024,1024, requires_grad=True) # real 32x32 image input
    b,c,x,y = A.shape

    A = torch.nn.functional.relu(A) # enforce positive constraint
    A_t = torch.einsum('bcij->bcji', A) # transpose of batched matrix
    A = torch.matmul(A, A_t) # to create a positive semi-definite matrix

    node = NormalizedCuts()
    y,misc = node.solve(A)
    node.test(A,y=y)

    # 2. Confirm the node solves the correct problems
    eqconstBool = issubclass(NormalizedCuts, EqConstDeclarativeNode) # If AbstractDeclarativeNode then false, remove this later?...
    fSolved = node.objective(A, y)
    print(f'Max: {torch.max(fSolved)}, Min: {torch.min(fSolved)}, Mean: {torch.mean(fSolved)}, std var: {torch.std(fSolved)}')
    if eqconstBool:
        # TODO: verify why this is not correct?
        fConsts = node.equality_constraints(A, y)
        print(f'Max: {torch.max(fConsts)}, Min: {torch.min(fConsts)}, Mean: {torch.mean(fConsts)}, std var: {torch.std(fConsts)}')