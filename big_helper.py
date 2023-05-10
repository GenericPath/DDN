import numpy as np

# different valid image sets
VALID_DATASETS = ['baby', 'MNIST', 'BW', 'TEXCOL']
def get_images(name='baby', length=1, size=(28,28)):
    if name not in VALID_DATASETS:
        raise ValueError(f"name must be one of {VALID_DATASETS}")
    
    from skimage.io import imread
    from skimage.transform import resize
    if name == 'baby':
        img_baby = imread("data/test/3.jpg",0)
        img_baby = resize(img_baby, size)
        return [img_baby]
    
    elif name == 'MNIST':
        from torchvision import datasets
        mnist = datasets.MNIST('data', train=True, download=True)
        imgs = [resize(np.asarray(mnist[i][0]), size) for i in range(length)]
        return imgs
    
    import os, pickle
    if name == 'BW':
        path = 'data/BWv1.pkl'
        if os.path.isfile(path):
            with open (path, 'rb') as fp:
                imgs = pickle.load(fp)
                diff = length - len(imgs)
                if diff < 0:
                    return imgs[:length]
                else:
                    return imgs + create_bw(diff, size)
        else:
            imgs = create_bw(length, size)
            with open(path, 'wb') as fp:
                pickle.dump(imgs, fp)
            return imgs
    elif name == 'TEXCOL':
        # currently doesnt check if the texture stuff isnt present...
        # TODO: check and maybe put it all into generate_datasets stuff?
        # TODO: make a check func in generate_dataset
        import glob
        from generate_dataset import make_texture_colour_image
        cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
        folder_path = './data/textures/Normalized Brodatz'
        extension = '.tif'
        imgs = glob(os.path.join(folder_path,'*'+extension))
        
        outputs = []
        for i in range(length):
            outputs.append(make_texture_colour_image(imgs, cmaps))
        # TODO: check these outputs... 
        # TODO: do the same as above with working within a pkl file instead of images to disk... (currently generated every time)
        return outputs 
    
    
def create_bw(length, size, ratio=None): # TODO: enforce a ratio between the B and W portions of BW image...
    import random
    
    outputs = []
    for i in range(length):
        w,h = (random.randint(size[0]//3, size[0]), random.randint(size[0]//3, size[0]))
        x,y = (random.randint(size[0]//3, size[0]), random.randint(size[0]//3, size[0]))

        xy = [(x-w//2,y-h//2), (x+w//2,y+h//2)]
        answer = np.zeros(size)
        answer[xy[0][0]:xy[1][0], xy[0][1]:xy[1][1]] = 255
        outputs.append(answer)
        
    return outputs 


def add_headers(
    fig,
    *,
    row_headers=None,
    col_headers=None,
    row_pad=1,
    col_pad=5,
    rotate_row_headers=True,
    **text_kwargs
):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )

def plot_images(imgs, labels=None, row_headers=None, col_headers=None, colmns=None, title=None):
    """
    imgs = [img1, img2]
    labels = ['label1', 'label2']
    colmns = 2 (so will be a 1x2 size display)
    """
    import matplotlib.pyplot as plt
    
    num = len(imgs)
    # Calculate the given number of subplots, or use colmns count to get a specific output
    if colmns == None:
        ay = np.ceil(np.sqrt(num)).astype(int) # this way it will prefer rows rather than columns
        ax = np.rint(np.sqrt(num)).astype(int)
    else:
        ax = np.ceil(num / colmns).astype(int)
        ay = colmns
        
    fig = plt.figure()
    if title is not None:
        fig.suptitle(title, fontsize=16)
    
    for i in range(1, num+1):
        sub = fig.add_subplot(ax,ay,i)
        if labels is not None:
            sub.set_title(f'{labels[i-1]}')
            
        sub.axis('off')
        sub.imshow(imgs[i-1])
        
    add_headers(fig, row_headers=row_headers, col_headers=col_headers, rotate_row_headers=False)
    # plt.tight_layout()

# different weighting functions
# intens, position, affinity, intens * position...

# different weightings with W/np.max(W) (e.g. do we normalize the weights before doing laplace?)

# different laplace solvers
# cheap, expensive, symmetric/none...
# how to handle 0's in D? or... how to handle 0's in d?

# different eigensolvers
# initially try for just one eig solver?
# Time the eigensolvers

# linalg.eig(L)
# np.eig(L)

# linalg.eigh(L)
# linalg.eig(L)
# linalg.eigh(L,D)
# linalg.eig(L,D)
# + the gvd variants etc....

# outputs
# - * 28, * eigval, * np.sqrt(D)
# sign of outputs to make it an indicator vector
# objective, eqconst