from nc import NormalizedCuts
from node import DeclarativeLayer
from net_argparser import net_argparser
from data import *
import torch
import torch.linalg
from torch.autograd import grad

import scipy.linalg
import scipy.sparse
import torch.nn.functional as F

# TODO: https://math.stackexchange.com/questions/3853424/what-does-the-value-of-eigenvectors-of-a-graph-laplacian-matrix-mean#comment7950799_3853794
# basically need to check these, otherwise everything doesnt work
# maybe use the psuedoinverse...
# https://en.wikipedia.org/wiki/Laplacian_matrix
# https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse

cmap_name = 'jet'

node = NormalizedCuts(eps=1e-3)
DL = DeclarativeLayer(node)


def partition(eigenvectors):
    """
    eigenvectors : (b, x, y)
    for dim = (x,y)
    """
    b,x,y = eigenvectors.shape

    output = []

    # eigenvectors = F.normalize(eigenvectors)

    for i in range(b):
        eigenvec = torch.clone(eigenvectors[i]).flatten()

        # Using average point to compute bipartition 
        second_smallest_vec = eigenvectors[i].flatten()
        avg = torch.sum(second_smallest_vec).item() / torch.numel(second_smallest_vec)

        bipartition = second_smallest_vec > avg
        seed = torch.argmax(torch.abs(second_smallest_vec))

        if bipartition[seed] != 1:
            eigenvec = eigenvec * -1
            bipartition = torch.logical_not(bipartition)
        bipartition = bipartition.reshape(args.img_size[0], args.img_size[1]).type(torch.double)
        
        output.append(bipartition.numpy())
    # output.
    return torch.tensor(output)


args = net_argparser(ipynb=True)
args.network = 1
args.total_images = 3
args.minify = False
args.radius = 100
args.img_size = [16,16] # the default is 32,32 anyway

train_dataset = SimpleDatasets(args, transform=transforms.ToTensor())

A = train_dataset.get_weights(0)
A = A.requires_grad_()

simple = True

if True:
    ##### SOLVE eigh
    A = de_minW(A) # check if needs to be converted from minVer style
    b,x,y = A.shape
    out_size = int(np.sqrt(x)) # NOTE: assumes it is square..
    output_size = (b,out_size,out_size)
    A = A.type(torch.double)


    A_p = torch.where(A == 0, 1e-5, A)
    d_p = A_p.sum(1)
    D_p = torch.diag_embed(d_p)

    # can also replace bc with ...
    d = torch.einsum('bij->bj', A) # eqv to A.sum(0) --- d vector
    D = torch.diag_embed(d) # D = matrix with d on diagonal
    D_inv_sqrt = torch.diag_embed(d.pow(-0.5))
    D_pinv = torch.linalg.pinv(D_p, hermitian=True)




    L = D-A # Laplacian matrix
    # The symmetrically normalized laplacian can be calculated as D^-0.5 * L * D^-0.5 or eqv. I - D^-0.5 * A * D^-0.5 
    
    L_norm = torch.einsum('...ij,...jk->...ik', torch.einsum('...ij,...jk->...ik', D_inv_sqrt , L) , D_inv_sqrt)
    L_norm_p = torch.bmm(torch.linalg.lstsq(D, L).solution, D_pinv)
    print("pause")

    if simple:
        _, a = torch.linalg.eigh(L)
        _, b = torch.linalg.eigh(L_norm)
        _, c = torch.linalg.eigh(L_norm_p)



        W = A[0].detach().numpy()

        L_1 = scipy.sparse.csgraph.laplacian(W, symmetrized=True, form="lo")  
        L_2 = scipy.sparse.csgraph.laplacian(W, normed=True, symmetrized=True, form="lo")        
      
        rng = np.random.default_rng()


        X = rng.random((x, 2))
        Y = np.ones((x, 1))

        _, eves_1 = scipy.sparse.linalg.lobpcg(L_1, X, Y=Y, largest=False, tol=1e-3)
        eves_1 *= np.sign(eves_1[0, 0])

        _, eves_2 = scipy.sparse.linalg.lobpcg(L_2, X, Y=Y, largest=False, tol=1e-3)
        eves_2 *= np.sign(eves_2[0, 0])


    else:
        # solve normal way
        _, v = torch.linalg.eigh(L_norm) # from my guess work
        _,v_L = torch.linalg.eigh(L)

        #### SOLVE svd
        U, _, _=torch.svd(L_norm)
        U_L, _, _ =torch.svd(L)

        ### solve lobpcg
        a, vec_lobpcg = torch.lobpcg(L_norm, k=2, B=D, largest=False)
        a, vec_lobpcg_L = torch.lobpcg(L, k=2, largest=False)

        ### solve general eigenvector problem with scipy
        vec_scipy_egih = []
        vec_scipy_egih_L = []

        for i in range(A.shape[0]):
            L_i = (D[i] - A[i]).detach().numpy()
            D_i = D[i].detach().numpy()

            D_i_sqrt = np.sqrt(D_i, where=[D_i!= 0])[0]
            D_i_sqrt_inv = np.linalg.inv(D_i_sqrt)

            L_i_norm = D_i_sqrt_inv @ L_i @ D_i_sqrt_inv

            _, eigenvector = scipy.linalg.eigh(L_i_norm, None, subset_by_index=[1,2])
            _, eigenvector_L = scipy.linalg.eigh(L_i, D_i, subset_by_index=[1,2])
            vec_scipy_egih.append(eigenvector)
            vec_scipy_egih_L.append(eigenvector_L)
        vec_scipy_egih = torch.tensor(vec_scipy_egih)
        vec_scipy_egih_L = torch.tensor(vec_scipy_egih_L)


if not simple:
    eigs = { # could also add the size of the eigenvectors..
        'eigh_0_L_norm'   : v[:,:,0,None].reshape(output_size),
        'eigh_1_L_norm'   : v[:,:,1,None].reshape(output_size),
        'svd_0_L_norm'    : U[:,:,0,None].reshape(output_size),
        'svd_1_L_norm'    : U[:,:,1,None].reshape(output_size),
        'lobpcg_0_L_norm'      : vec_lobpcg[:,:,0].reshape(output_size),
        'lobpcg_1_L_norm'      : vec_lobpcg[:,:,1].reshape(output_size),
        'sci_eigh_0_L_norm'  : vec_scipy_egih[:,:,0].reshape(output_size),
        'sci_eigh_1_L_norm'  : vec_scipy_egih[:,:,1].reshape(output_size),

        'eigh_0_L'   : v_L[:,:,0,None].reshape(output_size),
        'eigh_1_L'   : v_L[:,:,1,None].reshape(output_size),
        'svd_0_L'    : U_L[:,:,0,None].reshape(output_size),
        'svd_1_L'    : U_L[:,:,1,None].reshape(output_size),
        'lobpcg_0_L'      : vec_lobpcg_L[:,:,0].reshape(output_size),
        'lobpcg_1_L'      : vec_lobpcg_L[:,:,1].reshape(output_size),
        'scipy_eigh_0_L'  : vec_scipy_egih_L[:,:,0].reshape(output_size),
        'scipy_eigh_1_L'  : vec_scipy_egih_L[:,:,1].reshape(output_size),
    }

    for key, value in eigs.copy().items():
        eigs[key+'_bipart'] = partition(value)

    # W = F.normalize(A)
    W = A

    plots = []
    labels = []

    derivatives = [W]

    for name, y in eigs.items():
        f = node.objective(W, y)
        eq = node.equality_constraints(W,y)
        plots.append(y)
        labels.append(f'{f.item():.5f}\n{eq.item():.5f}\n{name}')

        # Dy = grad(y, (W), grad_outputs=torch.ones_like(y))
        # derivatives.append(Dy)

    splits = 8
    plots_split = [plots[x:x+splits] for x in range(0, len(plots), splits)]
    labels_split = [labels[x:x+splits] for x in range(0, len(labels), splits)]

    plots_split.append(W)
    labels_split.append([f'objective:\neqconst:\nW: r {args.radius}'])

    plot_multiple_images('test-eigs', plots_split, figsize=args.img_size, ipynb=False, cmap_name=cmap_name, labels=labels_split)
    # plot_multiple_images('test-Dy', derivatives, figsize=args.img_size, ipynb=False, cmap_name=cmap_name)

else:
    eigs = { # could also add the size of the eigenvectors..
        'L'         : a[:,:,0,None].reshape(output_size),
        'L_norm'    : b[:,:,0,None].reshape(output_size),
        'L_norm_p'  : c[:,:,0,None].reshape(output_size),
        'new-scipy_1' : torch.tensor(eves_1[:,0]).reshape(output_size),
        'new-scipy_2' : torch.tensor(eves_2[:,0]).reshape(output_size),
    }

    for key, value in eigs.copy().items():
        eigs[key+'_bipart'] = partition(value)

    W = A

    plots = []
    labels = []

    derivatives = [W]

    for name, y in eigs.items():
        f = node.objective(W, y)
        eq = node.equality_constraints(W,y)
        plots.append(y)
        labels.append(f'{f.item():.5f}\n{eq.item():.5f}\n{name}')

        # Dy = grad(y, (W), grad_outputs=torch.ones_like(y))
        # derivatives.append(Dy)

    splits = 3
    plots_split = [plots[x:x+splits] for x in range(0, len(plots), splits)]
    labels_split = [labels[x:x+splits] for x in range(0, len(labels), splits)]

    plots_split.append(W)
    labels_split.append([f'objective:\neqconst:\nW: r {args.radius}'])

    plot_multiple_images('test-eigs', plots_split, figsize=args.img_size, ipynb=False, cmap_name=cmap_name, labels=labels_split)
    # plot_multiple_images('test-Dy', derivatives, figsize=args.img_size, ipynb=False, cmap_name=cmap_name)
