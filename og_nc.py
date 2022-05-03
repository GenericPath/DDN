import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
from PIL import Image
import scipy
import scipy.spatial

# To compare against the acual NC answers
from nc import NormalizedCuts, manual_weight

def main():
    # A = ["a","b","c","d","e","f"]
    # N = len(A)
    # W = np.empty((N,N), dtype=object)
    # W[:] = ""
    # for u in range(N):
    #     for v in range(N):
    #         if np.linalg.norm(u-v) > 2: # 4-way connection
    #             continue
    #         W[u][v] = A[u]+A[v]
    # print(W)
    # return



    # I = plt.imread('data/test/bwVert.png')
    # I = 0.2989 * I[:, :, 0] + 0.5870 * I[:, :, 1] + 0.1140 * I[:, :, 2] # convert RGB to grayscale
    I = plt.imread('data/test/img2.png')
    # plt.imshow(I, cmap="gray")
    # plt.show()

    (nx, ny) = I.shape
    mesh = np.meshgrid(np.linspace(0, nx - 1, nx), np.linspace(0, ny - 1, ny))
    V = np.array([mesh[0].ravel(), mesh[1].ravel()]).T

    W = create_weights(I)
    W_2 = manual_weight(I)
    (A, B) = ncut(V, W)


    plt.imshow(I, cmap="gray")
    plt.scatter(A[:, 0], A[:, 1])
    plt.scatter(B[:, 0], B[:, 1])
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.title("Segmentation")
    plt.show()
    
    return

# From https://github.com/MichaelKarpe/ponts-paristech-projects/blob/master/computer-vision/image-segmentation/normalized-cuts/Avec%20une%20image.ipynb
def ncut(V, W):
    """
    Applique l'algorithme de séparation en deux clusters pour un critère x < 0, x > 0
    """
    d = np.sum(W, axis=0)
    D = np.diag(d)
    D_tmp = np.diag(d ** (-0.5))
    M = D_tmp.dot(D - W).dot(D_tmp)
    (eigen_values, eigen_vectors) = scipy.linalg.eig(M)
    idx = (-eigen_values).argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
    x_temp = eigen_vectors[:, 1]
    x = np.where(x_temp < 0, -1, 1)
    A = V[np.where(x < 0)[0]]
    B = V[np.where(x > 0)[0]]

    return (A, B)

def create_weights(I):
    W_cond = scipy.spatial.distance.pdist(I.ravel().reshape(-1, 1))
    W = scipy.spatial.distance.squareform(W_cond)
    return W

if __name__ == '__main__':
    main()
      