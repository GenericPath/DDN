from net_argparser import net_argparser
from data import *
from nc import NormalizedCuts


from scipy.sparse.linalg import eigsh, eigs

# TODO:
# implement scipy.sparse.linalg.eigsh as a solve option
# implement scipy.sparse.linalg.lobpcg as a solve option
# (OPTIONAL) implement scipy.optimize.minimize as a solve option
# Check the output part... as this may need to be adjusted to solve the nc nodes objective (and not just any valid eigenvector)

def new_solve(A, expected=None):
    """ 
    Solve the normalized cuts using eigenvectors (produces single cut, no recursion yet)

    Arguments:
        A: (b, N, N) Torch tensor,
            batch of affinity/weight tensors (N = x * y from orignal x,y images)

    TODO: pass a parameter to avoid hardcoded output dimensions
    """        
    # Implementation notes:
    # - requires einsum's to act on batch. Otherwise torch complains about tensors not being in graph being differentiated
        # TODO: check if above claim is true still since refactoring
    # - inf in D_inv_sqrt don't matter as other functions used seem to handle it fine, previously avoided by only inverting the diagonal

    A = A.detach() # TODO : verify if this breaks anything

    A = de_minW(A) # check if needs to be converted from minVer style
    b,x,y = A.shape
    out_size = int(np.sqrt(x)) # NOTE: assumes it is square..
    output_size = (b,out_size,out_size)

    # can also replace bc with ...
    d = torch.einsum('bij->bj', A) # eqv to A.sum(0) --- d vector
    D = torch.diag_embed(d) # D = matrix with d on diagonal
    D_inv_sqrt = torch.diag_embed(d.pow(-0.5)) # previously had pow inside diag

    L = D-A # Laplacian matrix

    # if self.symm_norm_L:
    #     # The symmetrically normalized laplacian can be calculated as D^-0.5 * L * D^-0.5 or eqv. I - D^-0.5 * A * D^-0.5 
    #     L_norm = torch.einsum('...ij,...jk->...ik', torch.einsum('...ij,...jk->...ik', D_inv_sqrt , L) , D_inv_sqrt)
    #     # L_norm = L_norm.to(A.device)
    # else:
    L_norm = L

    # # # # # # # # # #
    # Solve eigenvectors and eigenvalues
    # TODO: replace this bit
    # old bit
    # (w, v) = torch.linalg.eigh(L_norm.cpu())
    # new bit
    max_iter = 10000 # just guarantee some type of convergence to machine precision (tol=0)
    output = []
    
    for i in range(b):
        (w,v) = eigsh(A.detach().cpu().numpy()[i], maxiter=max_iter, tol=0, which='SM', k=1)
        output.append(v)
    
    # Returns the second smallest eigenvector
    # output = v[:,:,1,None].reshape(output_size)
    output = np.asarray(output)
    output = output.reshape(output_size)
    # DNN NOTE: Detach inputs from graph, attach only the output (or if using optimisation to solve you can with torch.enable_grad() ( ... optim ))
    
    # if self.bipart:
    #     output = partition(output)


    # TODO: put the {POST EIG} code here



    if expected is not None:
        print('TODO: make this plot the visual of it :)')


    # # # # # # # # # #
    # TODO: this is the bit to make it the correct numbers... so can scale and it will still be a valid eigenvector... need to work out which approach works best for this.
    output *= (out_size)

    # remove any inversion of groups A,B (so either doesn't flip sign)
    if output[0][0][0] > 0:
        output *= -1
        
    output = torch.tensor(output)
    return output.to(A.device).requires_grad_(True), None

def main():
    args = net_argparser(ipynb=True)
    args.network = 1
    args.total_images = 10
    args.minify = False 
    # args.bipart = False 
    # args.symm_norm_L = False
    args.radius = 100
    args.img_size = [16,16] # the default is 32,32 anyway

    train_dataset = SimpleDatasets(args, transform=transforms.ToTensor())

    true = train_dataset.get_segmentation(0)
    true[true > 0] = 1
    true[true <= 0] = -1

    if true[0][0][0] > 0:
        true *= -1

    W_true = train_dataset.get_weights(0).double()
    # print(W_true)    
    
    node = NormalizedCuts(eps=1e-3)#, bipart=args.bipart, symm_norm_L=args.symm_norm_L)

    y, _ = new_solve(W_true)
    node.gradient(W_true.requires_grad_(True), y=y)
    
    
    
    # {POST EIG} code.. TODO: move to the solve.. and then put the solve into the 
    
    print('donezo')

if __name__ == '__main__':
    main()