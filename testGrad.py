from net_argparser import net_argparser
from data import *
from nc import NormalizedCuts
from torch.autograd import grad

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

    y, _ = node.solve(W_true) # vs new_solve which should be the same now :)
    node.gradient(W_true.requires_grad_(True), y=y)
    
    f = torch.enable_grad()(node.objective)(W_true, y=y)
    fY = grad(f, y, grad_outputs=torch.ones_like(f), create_graph=True)[0]
    # fY = torch.enable_grad()(fY.reshape)(self.b, -1) # bxm

    # {POST EIG} code.. TODO: move to the solve.. and then put the solve into the 
    
    print('donezo')

if __name__ == '__main__':
    main()