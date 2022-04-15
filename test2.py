import torch
import torch.nn as nn

from nc import NormalizedCuts

def main():
    main = True

    if main: # Batched matrix version
        A = torch.randn(32,1,1024,1024, requires_grad=True) # real 32x32 image input
        b,c,x,y = A.shape

        A = torch.nn.functional.relu(A) # enforce positive constraint
        A_t = torch.einsum('bcij->bcji', A) # transpose of batched matrix
        A = torch.matmul(A, A_t) # to create a positive semi-definite matrix

        node = NormalizedCuts()
        y,misc = node.solve(A)
        node.test(A,y=y)

if __name__ == '__main__':
    main()
