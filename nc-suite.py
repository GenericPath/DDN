import colorsys
import math
import numpy as np

def intensity_weight(pixel):
    """
    Input the slice of the pixel (so 1, or 3, if RGB)
    - For RGB will use brightness (value in HSV)
    """
    if len(pixel) == 3:
        h,s,v = colorsys.rgb_to_hsv(*pixel)
        return v
    elif len(pixel) == 1:
        return pixel
    else:
        raise Exception("invalid input to intensity_weight()")
            
def color_weights(pixel):
    """
    Input RGB for a colour based function
    """
    h,s,v = colorsys.rgb_to_hsv(*pixel)
    return [v, v * s * math.sin(h), v * s * math.cos(h)]

def texture_weights(pixel):
    raise Exception("texture weights isn't implemented yet, see Malik 1990 Texture Discrimination")
    


def new_weights(img, F, r, sigma_I, sigma_X):
    """
    Given an img, use F(i) to compare different between two pixels and return the weights matrix W
    """

    RGB = (len(img) == 3)
    
    
    for row in img:
        for col in row:
            return





# def weights(img, weighting_function):
#     channel = 1
#     n_row, n_col = img.shape
    
#     N = n_row*n_col
#     W = np.zeros((N,N))
    
#     r = 2
#     sigma_I = 0.1
#     sigma_X = 1
    
#     for row_count, row in enumerate(img):
#         for col_count, v in enumerate(row):
#             index = row_count * n_col + col_count

#             search_w = r * 2 + 1
#             start_row = row_count - r
#             start_col = col_count - r

#             for d_row in range(search_w):
#                 for d_col in range(search_w):
#                     new_row = start_row + d_row
#                     new_col = start_col + d_col
#                     dst = (new_row - row_count) ** 2 + (new_col - col_count) ** 2
#                     if 0 <= new_col < n_col and 0 <= new_row < n_row:
#                         if dst >= r ** 2:
#                             continue

#                         cur_index = int(new_row * n_col + new_col)

#                         F = img[row_count, col_count] - img[new_row, new_col]
#                         if channel == 3:
#                             F_diff = F[0]**2 + F[1]**2 + F[2]**2  
#                         else:
#                             F_diff = F**2

#                         w = np.exp(-((F_diff / (sigma_I ** 2)) + (dst / (sigma_X ** 2))))
#                         W[index, cur_index] = w

#     return W

if __name__ == '__main__':
    r = 2
    sigma_I = 0.1
    sigma_X = 5