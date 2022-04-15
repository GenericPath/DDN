import torch
import torch.nn as nn

def main():
    main = True
    constant = 1e-5

    if main: # Batched matrix version
        A = torch.randn(32,1,1024,1024) # real 32x32 image input
        b,c,x,y = A.shape

        A = torch.nn.functional.relu(A, inplace=True) # enforce positive constraint
        A_t = torch.einsum('bcij->bcji', A) # transpose of batched matrix
        A = torch.matmul(A, A_t) # to create a positive semi-definite matrix

        # B = torch.randn(32,1,32) # real 32x32 image output

        # can also replace bc with ...
        d = torch.einsum('bcij->bcj', A) # == A.sum(0) --- d vector
        D = torch.diag_embed(d) # D = matrix with d on diagonal
        D_inv_sqrt = torch.diag_embed(d.pow(-0.5)) # Don't calculate inverse sqrt of 0 entries (non diagonals)

        L = (D-A) # Laplacian matrix
        # The symmetrically normalized laplacian can be calculated as D^-0.5 * L * D^-0.5 or eqv. I - D^-0.5 * A * D^-0.5 
        L_norm = torch.einsum('bcij,bcjk->bcik', torch.einsum('bcij,bcjk->bcik', D_inv_sqrt , L) , D_inv_sqrt)

        # Solve eigenvectors and eigenvalues
        (w, v) = torch.linalg.eigh(L_norm)
        
        # Returns the computed eigenvectors for the normalised laplacian
        # this is the second smallest eigenvector, and above.

        # the eigenvector is reshaped to match the expected inputs for a conv net
        # each of these new images represents some binary segmentation
        # defaults to the second smallest eigenvector only, and as such is a binary segmentation
        num_eigs = 1
        return v.narrow(v, -2, 1, num_eigs).squeeze(1).reshape(b, num_eigs, x, y)

    else: # No batches version
        # Generate a random positive semi-definite matrix (M = A*A^T)
        A = torch.tensor(([1,2,3],[4,5,6],[7,8,9]), dtype=torch.float32)
        M = torch.matmul(A, A.T)
        # Add a small constant factor (as is done for the NN version.. without positive semi-definite guarantees)
        M = M.add(constant)

        # Calculate d from M
        d = M.sum(0)
        # From d, form D
        D = torch.diag_embed(d) # defaults to embedding -2, -1 dimenions (last two)



        # torch.einsum('bcij->bcji', A) # transpose A
        # torch.einsum('bcij,bcjk->bcik', A , B) # matrix multiplication


    # torch.einsum('bcij->bcji', A) # transpose A
    # torch.einsum('bcij,bcjk->bcik', A , B) # matrix multiplication


    # def general_eigen(self, A, y):
    # """ f = y^T A y """
    
    # # Batch         
    # yT = torch.einsum('bij->bji', y)
    # # Batch matrix multiplication
    # return torch.einsum('bij,bjk->bik', torch.einsum('bij,bjk->bik', yT, A), y)
    
    # # For single problem...        
    # # return torch.matmul(torch.matmul(y.t()), A), y)
    
    # def objective(self, x, y):
    #     """ f(x,y) = y^T (D-W) y """
    #     D = torch.einsum('bij->bj', x)
    #     D = torch.diag_embed(D)
    #     L = D - x # Laplacian matrix
    #     return self.general_eigen(L, y)


if __name__ == '__main__':
    main()
