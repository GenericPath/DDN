import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import linalg
import networkx as nx

import math

def test_cost(a,b, sigma):
    cost = 100 * math.exp(- pow(a - b, 2) / (2 * pow(sigma, 2))) # TODO check if needs to be abs of (a-b)
    # TODO: version with *= 1/dist(a,b) type of thing
    # like in https://www.csd.uwo.ca/~yboykov/Papers/ijcv06.pdf
    return cost

def final_weight_test(img, r, sigma, cost=test_cost):
    # following from https://github.com/julie-jiang/image-segmentation/blob/master/imagesegmentation.py
    X,Y = img.shape
    N = X*X
    W = np.zeros((N, N))
    
    for i in range(X):
        for j in range(Y):
            x = i * Y + j
            if i + 1 < X: # pixel below # TODO: understand what these ifs are doing?
                y = (i + 1) * Y + j
                W[x][y] = W[y][x] = cost(img[i][j], img[i + 1][j], sigma)
            if j + 1 < Y: # pixel to the right
                y = i * Y + j + 1
                W[x][y] = W[y][x] = cost(img[i][j], img[i][j + 1], sigma)
    return W

def colour_diff(image, pixel1, pixel2):
    # # Extract the color values of the two pixels
    # two approaches for colour vs black and white
    if len(image.shape) == 2:
        intensity1 = image[pixel1[0], pixel1[1]]
        intensity2 = image[pixel2[0], pixel2[1]]
        return np.abs(intensity1 - intensity2)
    else:
        color1 = image[pixel1[0], pixel1[1], :]
        color2 = image[pixel2[0], pixel2[1], :]
        return np.sqrt(np.sum((color1 - color2)**2))

def texture_diff(image, pixel1, pixel2, neighborhood_size):
    # was previously using some stuff from skimage.feature, but no more
    # Extract the neighborhood around the first pixel
    x1, y1 = pixel1
    neighborhood1 = image[x1-neighborhood_size:x1+neighborhood_size+1,
                              y1-neighborhood_size:y1+neighborhood_size+1]
    # Extract the neighborhood around the second pixel
    x2, y2 = pixel2
    neighborhood2 = image[x2-neighborhood_size:x2+neighborhood_size+1,
                              y2-neighborhood_size:y2+neighborhood_size+1]
    # Calculate the texture difference
    if len(image.shape) == 2:  # Grayscale
        texture_diff = np.abs(neighborhood1 - neighborhood2).mean()
    else:  # Colour
        texture_diff = np.sqrt(np.sum((neighborhood1 - neighborhood2)**2, axis=(0,1))).mean()
    return texture_diff

def generic_weight(img, radius, func, sigmaI, sigmaX):
    """Generic function for weighting, takes a func which computes difference measure

    Args:
        img (Array): img to compute all weights for
        radius (int): The radius of neighbourhood to compute weights for
        func (function): Used to compute the different between two pixels, 
                        use partial for additional params
        sigmaI (float): weighting of the function
        sigmaX (float): weighting of spatial location

    Returns:
        Array: W, weights matrix. shape: (X**2, Y**2)
    """
    X,Y = img.shape
    W = np.zeros((X*X, Y*Y))
    for x in range(X):
        for y in range(Y):
            for dx in range(max(0,x-radius), min(X,x+radius)):
                for dy in range(max(0,y-radius), min(Y,y+radius)):
                    # W = compare (x,y) with (d,y)
                    W[x*X + dx][y*Y + dy] = np.exp(-np.abs(func(img, (x,y),(dx,dy)))/sigmaI) # function to weight them
                    W[x*X + dx][y*Y + dy] *= np.exp(-np.abs(math.dist((x,y),(dx,dy)))/sigmaX) # distance
                    continue
    return W

def generic_weight_noexp(img, radius, func, sigmaX):
    # Same as above, without the np.exp part of it for the weighting function :)
    X,Y = img.shape
    W = np.zeros((X*X, Y*Y))
    for x in range(X):
        for y in range(Y):
            for dx in range(max(0,x-radius), min(X,x+radius)):
                for dy in range(max(0,y-radius), min(Y,y+radius)):
                    # W = compare (x,y) with (d,y)
                    W[x*X + dx][y*Y + dy] = func(img, (x,y),(dx,dy)) # function to weight them
                    W[x*X + dx][y*Y + dy] *= np.exp(-np.abs(math.dist((x,y),(dx,dy)))/sigmaX) # distance
                    continue
    return W

def generic_weight_rawfunc(img, radius, func):
    # Same as above, without the np.exp part of it for the weighting function :)
    X,Y = img.shape
    W = np.zeros((X*X, Y*Y))
    for x in range(X):
        for y in range(Y):
            for dx in range(max(0,x-radius), min(X,x+radius)):
                for dy in range(max(0,y-radius), min(Y,y+radius)):
                    # W = compare (x,y) with (d,y)
                    W[x*X + dx][y*Y + dy] = func(img, (x,y),(dx,dy)) # function to weight them
                    continue
    return W

def weight_tot(img, radius, sigmaI, sigmaX):
    # TODO: fix this relative to the new weight_int
    X,Y = img.shape
    W = np.zeros((X*X, Y*Y))
    for x in range(X):
        for y in range(Y):
            for dx in range(max(0,x-radius), min(X,x+radius)):
                for dy in range(max(0,y-radius), min(Y,y+radius)):
                    # W = compare (x,y) with (d,y)
                    W[x*X + dx][y*Y + dy] = np.exp(-np.abs(img[x][y]-img[dx][dy])/sigmaI) # intensity
                    W[x*X + dx][y*Y + dy] *= np.exp(-np.abs(math.dist((x,y),(dx,dy)))/sigmaX) # distance
                    continue
    return W

def weight_int_broken(img, radius, sigmaI):
    X,Y = img.shape
    W = np.zeros((X*X, Y*Y))
    for x in range(X):
        for y in range(Y):
            for dx in range(max(0,x-radius), min(X,x+radius)):
                for dy in range(max(0,y-radius), min(Y,y+radius)):
                    W[x*X + dx][y*Y + dy] = np.exp(-np.abs(img[x][y]-img[dx][dy])/sigmaI) # intensity
                    continue
    return W

def weight_int_broken2(img, radius, sigmaI):
    X,Y = img.shape
    W = np.zeros((X*X, Y*Y))
    
    radius = min(X // 2, radius)
    
    for x in range(X):
        for y in range(Y):
            for dx in range(max(0,x-radius), min(X-x,x+radius)):
                for dy in range(max(0,y-radius), min(Y-y,y+radius)):
                    u,v = x + dx, y + dy
                    W[y*Y+x][u*Y + v] = np.exp(-np.abs(img[x][y]-img[u][v])/sigmaI) # intensity
                    continue
    return W

def weight_int(img, radius, sigmaI=0.1):
    X,Y = img.shape
    N = X*X
    W = np.zeros((N, N))
    
    radius = min(N // 2, radius)
    
    I = img.flatten()
    for u in range(N-1): # could use step size of r to improve speed?
        end = min(u+radius+1, N) # upper triangle, only traverse as far as needed
        for v in range(u,end): # end is exclusive bound
            x1,y1 = u // X, u % Y
            x2,y2 = v // X, v % Y
            coord1 = np.array([x1,y1])
            coord2 = np.array([x2,y2])
            if np.linalg.norm(coord1-coord2) > radius: # how far of a connection to add
                continue
            else:
                W[u][v] = W[v][u] = np.exp(-np.abs(img[x1][y1]-img[x2][y2])/sigmaI) # intensity
    return W

def weight_dist(img, radius, sigmaX):
    # TODO: fix this relative to the new weight_int
    X,Y = img.shape
    W = np.zeros((X*X, Y*Y))
    for x in range(X):
        for y in range(Y):
            for dx in range(max(0,x-radius), min(X,x+radius)):
                for dy in range(max(0,y-radius), min(Y,y+radius)):
                    W[x*X + dx][y*Y + dy] = np.exp(-np.abs(math.dist((x,y),(dx,dy)))/sigmaX) # distance
                    continue
    return W

def within_percentage(x,y,percentage):
    diff = abs(x-y)
    thresh = (percentage/100) * max(x,y)
    return diff <= thresh

def manual_weights_binary(img, r=300, percentage=40):
    # assumes grayscale
    X,Y = img.shape
    N = X*Y
    W = np.zeros((N,N))

    r = min(N//2, r) # ensure the r value doesn't exceed the axes of the outputs

    I = img.flatten()
    for u in range(N-1): # could use step size of r to improve speed?
        end = min(u+r+1, N) # upper triangle, only traverse as far as needed
        for v in range(u,end):
            coord1 = np.array([u // X, u % Y])
            coord2 = np.array([v // X, v % Y])
            if np.linalg.norm(coord1-coord2) > r: # 4-way connection
                continue
            else:
                W[u][v] = W[v][u] = within_percentage(I[u],I[v],percentage)
                # W[u][v] = W[v][u] = not I[u] == I[v] # Symmetric (0 if same, 1 if different)
    return W

def manual_weights_binary2(img, r=300, percentage=40):
    X, Y = img.shape
    N = X * Y
    r = min(N // 2, r)

    I = img.flatten()
    indices = np.arange(N)

    coord1 = np.array([indices // X, indices % Y]).T
    coord2 = coord1.reshape((1, N, 2))

    distances = np.linalg.norm(coord1 - coord2, axis=2)
    within_radius = distances <= r

    W = np.zeros((N, N))
    # W[np.logical_not(within_radius)] = 0.0

    for u in range(N-1):
        end = min(u + r + 1, N)
        for v in range(u + 1, end):
            if within_radius[u, v]:
                W[u, v] = W[v, u] = within_percentage(I[u], I[v], percentage)

def manual_weights_abs(img, r=300):
    # assumes grayscale
    X,Y = img.shape
    N = X*Y
    W = np.zeros((N,N))

    r = min(N//2, r) # ensure the r value doesn't exceed the axes of the outputs

    I = img.flatten()
    for u in range(N-1): # could use step size of r to improve speed?
        end = min(u+r+1, N) # upper triangle, only traverse as far as needed
        for v in range(u,end): # end is exclusive bound
            coord1 = np.array([u // X, u % Y])
            coord2 = np.array([v // X, v % Y])
            if np.linalg.norm(coord1-coord2) > r: # r-way connection
                continue
            else:
                W[u][v] = W[v][u] = np.abs(I[u] - I[v]) # Symmetric
    return W

def manual_weight_abs2(img, r=300):
    X,Y = img.shape
    N = X*Y
    W = np.zeros((N,N))

    r = min(N//2, r) # ensure the r value doesn't exceed the axes of the outputs

    I = img.flatten()
    for u in range(N-1): # could use step size of r to improve speed?
        end = min(u+r+1, N) # upper triangle, only traverse as far as needed
        for v in range(u,end): # end is exclusive bound
            coord1 = np.array([u // X, u % Y])
            coord2 = np.array([v // X, v % Y])
            if np.linalg.norm(coord1-coord2) > r: # r-way connection
                continue
            else:
                W[u][v] = W[v][u] = np.abs(I[u] - I[v]) # Symmetric
    return W

def manual_weights_abs_upper(img, r=300):
    N = img.shape[0] * img.shape[1]
    W = np.zeros((N,N))

    r = min(N//2, r) # ensure the r value doesn't exceed the axes of the outputs

    I = img.flatten()
    for u in range(N-1): # could use step size of r to improve speed?
        end = min(u+r+1, N) # upper triangle, only traverse as far as needed
        for v in range(u,end):
            if np.linalg.norm(u-v) > r: # 4-way connection
                continue
            W[u][v] = np.abs(I[u] - I[v]) # Upper only
    return W

def intensity_weight_matrix(img, r=None): # blank arg R to match syntax of others with minimal code changes
  weight = np.abs(np.float32(img.flatten()[:, np.newaxis]) - np.float32(img.flatten()[np.newaxis, :]))
  W = np.exp(-weight/10)*255
  return W

def positional_weight_matrix(img): 
  m,n = img.shape                                                                                                     
  X, y = np.meshgrid(np.arange(m), np.arange(n))                                                                 
  X = X.flatten()
  Y = y.flatten()

  distance = np.sqrt((X[:, np.newaxis] - X[np.newaxis, :])**2 + (Y[:, np.newaxis] - Y[np.newaxis, :])**2)
  W = np.exp(-distance/5)
  W =W*(W>0.58)
  return W

def intens_posit_wm(img):
    """
    No ratio intens and positonal version
    """
    return intensity_weight_matrix(img) * positional_weight_matrix(img)

def weights_2(img, r=2, sigma_I=0.2, sigma_X=1):
    channel = 1
    n_row, n_col = img.shape
    
    N = n_row*n_col
    W = np.zeros((N,N))
    
    for row_count, row in enumerate(img):
        for col_count, v in enumerate(row):
            index = row_count * n_col + col_count

            search_w = r * 2 + 1
            start_row = row_count - r
            start_col = col_count - r

            for d_row in range(search_w):
                for d_col in range(search_w):
                    new_row = start_row + d_row
                    new_col = start_col + d_col
                    dst = (new_row - row_count) ** 2 + (new_col - col_count) ** 2
                    if 0 <= new_col < n_col and 0 <= new_row < n_row:
                        if dst >= r ** 2:
                            continue

                        cur_index = int(new_row * n_col + new_col)

                        F = img[row_count, col_count] - img[new_row, new_col]
                        if channel == 3:
                            F_diff = F[0]**2 + F[1]**2 + F[2]**2  
                        else:
                            F_diff = np.abs(F) #**2

                        w = np.exp(-((F_diff / (sigma_I ** 2)) + (dst / (sigma_X ** 2))))
                        W[index, cur_index] = w

    return W

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
    
    num = len(imgs)
    # Calculate the given number of subplots, or use colmns count to get a specific output
    if colmns is None:
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

def DW_matrices(graph):
    # using networkx graph to get D and W
    W = nx.to_numpy_array(graph)
    d = W.sum(axis=0)
    D = np.diag(d) 
    
    return D, W

def D_matrix(W):
    d = W.sum(axis=0)
    D = np.diag(d) 
    
    return D

def argmin2(array):
    # O(n) to find second smallest argmin, instead of sorting O(n^2)
    min1 = np.inf
    min2 = np.inf
    min_idx1 = min_idx2 = i = 0
    n = array.shape[0]
    
    for i in range(n):
        x = array[i]
        if x < min1:
            min2 = min1
            min_idx2 = min_idx1
            min1 = x
            min_idx1 = i
        elif x > min1 and x < min2:
            min2 = x
            min_idx2 = i
        i += 1

    return min_idx2

def cut_cost(cut, W):
    total_weight = 0
    for i in range(len(cut)):
        for j in range(len(cut)): # should be square
            if cut[i] != cut[j]:
                total_weight += W[i][j] # assumes all weights are non-negative
    return total_weight

def ncut_cost(cut, D, W):
    cut = np.array(cut)
    cc = cut_cost(cut, W)

    # D has elements only along the diagonal, one per node, so we can directly
    # index the data attribute with cut.
    # ~ is a bitwise negation operator (flip bits)
    assoc_a = D[cut].sum()
    assoc_b = D[~cut].sum()

    return (cc / assoc_a) + (cc / assoc_b)

def get_min_ncut(ev, d, w, num_cuts):
    mcut = np.inf
    mn = ev.min()
    mx = ev.max()

    # If all values in `ev` are equal, it implies that the graph can't be
    # further sub-divided. In this case the bi-partition is the the graph
    # itself and an empty set.
    min_mask = np.zeros_like(ev, dtype=bool)
    if np.allclose(mn, mx):
        return min_mask, mcut

    # Refer Shi & Malik 2001, Section 3.1.3, Page 892
    # Perform evenly spaced n-cuts and determine the optimal one.
    for t in np.linspace(mn, mx, num_cuts, endpoint=False):
        mask = ev > t
        cost = ncut_cost(mask, d, w)
        if cost < mcut:
            min_mask = mask
            mcut = cost

    return min_mask, mcut

def solve_ncut(D,W):
    d2 = D.copy()
    d2 = np.reciprocal(np.sqrt(d2.data, where=d2>0), where=d2>0) # avoid nans and infs using where - using out=d2.data may be more efficient?

    A = d2 * (D - W) * d2

    m = W.shape[0]

    random_state = np.random.default_rng(0)
    v0 = random_state.random(A.shape[0])
    vals, vectors = linalg.eigsh(A, which='SM', v0=v0, # if it is sparse this converges quickly, otherwise it doesn't really
                                        k=min(100, m - 2))

    vals, vectors = np.real(vals), np.real(vectors)
    index2 = argmin2(vals)
    ev = vectors[:, index2]
        
    return ev


def _deterministic_vector_sign_flip(u):
    """Modify the sign of vectors for reproducibility.

    Flips the sign of elements of all the vectors (rows of u) such that
    the absolute maximum element of each vector is positive.

    Parameters
    ----------
    u : ndarray
        Array with vectors as its rows.

    Returns
    -------
    u_flipped : ndarray with same shape as u
        Array with the sign flipped vectors as its rows.
    """
    # from https://github.com/scikit-learn/scikit-learn/blob/2a2772a87b6c772dc3b8292bcffb990ce27515a8/sklearn/utils/extmath.py#L1093
    # used in SpectralClustering
    
    
    max_abs_rows = np.argmax(np.abs(u), axis=1)
    signs = np.sign(u[range(u.shape[0]), max_abs_rows])
    u *= signs[:, np.newaxis]
    return u

def partition_by_step(input, D, W):
    step = 50
    pos = input.copy()
    max_value = pos.max()
    min_value = pos.min()
    setp = (max_value - min_value) / step
    dict = {}
    for i in range(1, step):
        partition = (min_value + i * setp)
        temp_pos = pos < partition


        k = (np.sum(W[temp_pos])) / (np.sum(D))
        b = k / (1 - k)

        y = temp_pos.astype('float64') * 2 - b * (temp_pos == False).astype('float64') * 2

        ncut = (y @ (D - W) @ y.T) / (y @ D @ y.T)
        dict[i] = ncut

    min_partition = min_value + min(dict, key=dict.get) * setp
    pos[pos >= min_partition] = 255
    pos[pos < min_partition] = 0

    pos = pos.reshape((28, 28))

    return pos.astype('uint8')

def partition_by_zero(input):
    input = input.reshape((28,28)).astype('float64')   
    input[input>0] = 255
    input[input<=0] = 0
    return input.astype('uint8')

def partition_by_avg(input):
    return partition_by_zero(input - np.average(input))

def partition_by_avg_nocut(input):
    return input - np.average(input)