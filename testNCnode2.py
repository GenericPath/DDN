from nc import NormalizedCuts
from node import DeclarativeLayer
from net_argparser import net_argparser
from data import *
import torch
from torch.autograd import grad

cmap_name = 'jet'

node = NormalizedCuts(eps=1e-3)
DL = DeclarativeLayer(node)

args = net_argparser(ipynb=True)
args.network = 1
args.total_images = 3
args.minify = True
args.radius = 20
args.img_size = [16,16] # the default is 32,32 anyway

train_dataset = SimpleDatasets(args, transform=transforms.ToTensor())

A = train_dataset.get_weights(2)
A = A.requires_grad_()

if True:
    ##### SOLVE eigh
    A = de_minW(A) # check if needs to be converted from minVer style
    b,x,y = A.shape
    out_size = int(np.sqrt(x)) # NOTE: assumes it is square..
    output_size = (b,out_size,out_size)
    A = A.type(torch.double)


    # can also replace bc with ...
    d = torch.einsum('bij->bj', A) # eqv to A.sum(0) --- d vector
    D = torch.diag_embed(d) # D = matrix with d on diagonal
    D_inv_sqrt = torch.diag_embed(d.pow(-0.5)) # previously had pow inside diag

    L = D-A # Laplacian matrix
    # The symmetrically normalized laplacian can be calculated as D^-0.5 * L * D^-0.5 or eqv. I - D^-0.5 * A * D^-0.5 
    L_norm = torch.einsum('...ij,...jk->...ik', torch.einsum('...ij,...jk->...ik', D_inv_sqrt , L) , D_inv_sqrt)

    # Solve eigenvectors and eigenvalues
    w, v = torch.linalg.eigh(L_norm)


    #### SOLVE svd
    U,S,V=torch.svd(L_norm)

    

eigs = {
    'eigh_1_None' : v[:,:,1,None].reshape(output_size).requires_grad_(),
    'eigh_0_None' : v[:,:,0,None].reshape(output_size).requires_grad_(),
    'svd_1_None'  : U[:,:,1,None].reshape(output_size).requires_grad_(),
    'svd_0_None'  : U[:,:,0,None].reshape(output_size).requires_grad_(),
}

print(eigs.keys())

W = A

plots = [W]
labels = ['objective:\neqconst:\nW']

derivatives = [W]

for name, y in eigs.items():
    print(f'y shape {y.shape}, W shape {W.shape}')
    print(name)

    # y = torch.sign(y)

    f = node.objective(W, y)
    eq = node.equality_constraints(W,y)
    plots.append(y)
    labels.append(f'{f.item():.5f}\n{eq.item():.5f}\n{name}')

    # Dy = grad(y, (W), grad_outputs=torch.ones_like(y))

    # derivatives.append(Dy)

plot_multiple_images('test-eigs', plots, figsize=args.img_size, ipynb=False, cmap_name=cmap_name, labels=labels)
plot_multiple_images('test-Dy', derivatives, figsize=args.img_size, ipynb=False, cmap_name=cmap_name)

print(eigs['svd_0_None'])