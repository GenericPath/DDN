import torch
import torch.nn as nn

def main():
    main = True
    constant = 1e-5

    if main: # Batched matrix version
        A = torch.randn(32,1,32,32) # real 32x32 image input
        A = torch.nn.functional.relu(A, inplace=True) # enforce positive constraint
        A_t = torch.einsum('bcij->bcji', A) # transpose of batched matrix
        A = torch.matmul(A, A_t) # to create a positive semi-definite matrix

        # B = torch.randn(32,1,32) # real 32x32 image output

        # These were an attempt to multiple by an I of the correct size, but realised mistakes
        # b,c,x,y = A.shape # last two dimensions of input
        # I = torch.eye(x,y).reshape(1,1,x,y).repeat(b,c,1,1) # batch of identity matrices

        # can also replace bc with ...
        d = torch.einsum('bcij->bcj', A) # == A.sum(0) --- d vector
        D = torch.diag_embed(d) # D = matrix with d on diagonal
        D_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(d))

        L = (D-A)
        L_norm = torch.einsum('bcij,bcjk->bcik', torch.einsum('bcij,bcjk->bcik', D_inv_sqrt , L) , D_inv_sqrt)

        # Approach 1
        # D_inv_sqrt = torch.linalg.inv(torch.sqrt(D)) # eqv. to torch.inverse
        # torch.where(D_inv_sqrt == float('inf'), # etc

        # Approach 2
        # from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/get_laplacian.html
        # deg_inv_sqrt = deg.pow_(-0.5)
        # deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

        # Approach 3
        # but instead...
        # https://pytorch.org/docs/master/generated/torch.linalg.inv.html#torch.linalg.inv
        # suggests torch.linalg.solve to be more stable.

        # solves AX = B for X.. so gives inverse(A)*B
        # L_norm = torch.linalg.solve(torch.sqrt(D), (D-A))# Normalized laplacian (D^-0.5 * (D-W) * D^-0.5)
        # TODO: check if D-A or just A...
        # to solve for XA = B, pass inputs A and B tranposed, and tranpose solution
        # L_norm = torch.transpose(torch.linalg.solve(torch.transpose(torch.sqrt(D), -2, -1), torch.transpose(L_norm, -2, -1)), -2, -1)

        # Solve eigenvectors and eigenvalues
        (w, v) = torch.linalg.eigh(L_norm)

        print('done')


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

        print(c)



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
