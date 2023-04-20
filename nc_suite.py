import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import linalg
import networkx as nx


def manual_weights_binary(img, r=1):
    N = img.shape[0] * img.shape[1]
    W = np.zeros((N,N))

    r = min(N//2, r) # ensure the r value doesn't exceed the axes of the outputs

    I = img.flatten()
    for u in range(N-1): # could use step size of r to improve speed?
        end = min(u+r+1, N-1) # upper triangle, only traverse as far as needed
        for v in range(u,end):
            if np.linalg.norm(u-v) > r: # 4-way connection
                continue
            W[u][v] = W[u][v] = not I[u] == I[v] # Symmetric (0 if same, 1 if different)
    return W

def manual_weights_abs(img, r=1):
    N = img.shape[0] * img.shape[1]
    W = np.zeros((N,N))

    r = min(N//2, r) # ensure the r value doesn't exceed the axes of the outputs

    I = img.flatten()
    for u in range(N-1): # could use step size of r to improve speed?
        end = min(u+r+1, N-1) # upper triangle, only traverse as far as needed
        for v in range(u,end):
            if np.linalg.norm(u-v) > r: # 4-way connection
                continue
            W[u][v] = W[u][v] = np.abs(I[u] - I[v]) # Symmetric
    return W

def intensity_weight_matrix(img):                                                                                 
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

def weights_2(img):
    channel = 1
    n_row, n_col = img.shape
    
    N = n_row*n_col
    W = np.zeros((N,N))
    
    r = 2
    sigma_I = 0.2
    sigma_X = 1
    
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