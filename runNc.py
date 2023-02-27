import numpy as np
import scipy.sparse.linalg

# https://github.com/lin-902/Normalized_cuts/blob/master/ncut/ncut.py#L77
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html

# https://github.com/rcf97/Image-Segmentation/blob/main/image_segmentation.py
# https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/future/graph/graph_cut.py#L72-L148


def laplacian(D, A, symm=True):
    """
    D - Degree matrix
    A - Weights matrix (can be referred to as W)
    """
    if symm:
        return D-A
    else:
        return D-A
    
    
def cut(input):
    scipy.sparse.linalg.eigs()
    
def weights(img):
    channel = 1
    n_row, n_col = img.shape
    
    N = n_row*n_col
    W = np.zeros((N,N))
    
    r = 2
    sigma_I = 0.1
    sigma_X = 1.0
    
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
                        F_diff = F[0]**2 + F[1]**2 + F[2]**2 if channel == 3 else F[0]**2

                        W = np.exp(-((F_diff / (sigma_I ** 2)) + (dst / (sigma_X ** 2))))
                        W[index, cur_index] = W

    return W