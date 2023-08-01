import numpy as np

def save_imgs():
    return

def save_data():
    return

def get_images(size=(28,28), filename='data/test/3.jpg'):
    """ Currently only generates a single test image """
    
    # from skimage.io import imread
    # from skimage.transform import resize
    # from skimage.color import rgb2gray
    import cv2
    import numpy as np

    img_cv2 = np.array(cv2.resize(cv2.imread(filename,0), size), dtype=np.float32) 
    # float32 to remove overflows in weights matrix

    # img_sk = imread("data/test/3.jpg",0)
    # img_sk_gray = rgb2gray(img_sk)
    # img_sk = resize(img_sk, size)
    # img_sk_gray = resize(img_sk_gray, size)

    img_cv2_norm = img_cv2 / np.max(img_cv2)

    imgs = [img_cv2_norm]
    imgs_text = ["cv2_norm"]

    # imgs = [img_cv2,img_cv2_norm,img_sk_gray, img_sk]
    # imgs_text = ["cv2(0,255)", "cv2/max(0,1)", "sk(gray)", "sk"]
    print(imgs_text)
    print(f'{np.min(img):.4f}, {np.max(img):.4f}\n{img.dtype}' for img in imgs)
    return imgs, imgs_text
    
    
def get_weights(image, radii=[1,-1]):
    """ Given an image, compute a number of weights """
    from nc_suite import manual_weight_abs2, manual_weights_binary, intens_posit_wm
    from nc_suite import intensity_weight_matrix
    
    methods = [manual_weight_abs2, manual_weights_binary]
    method_names = ["manual_weight_abs2", "manual_weights_binary"]
    
    weight_labels = [] 
    weights = []
    for radius in radii:
        if radius == -1: # -1 corresponds to full size
            radius = image.size # or np.prod(np.shape)
        for method, method_name in zip(methods, method_names):
            weights.append(method(image, radius))
            weight_labels.append(f"{str(radius)}\n{method_name}")
    weights.append(intens_posit_wm(image))
    weight_labels.append(f"{image}\nintens_posit_wm")
    weights.append(intensity_weight_matrix(image))
    weight_labels.append(f'{image}\nintensity_weight_matrix')
    
    return weights, weight_labels
    
def get_laplaces(W, nums=[0],W_zerods=[True], L_zerods=[False]):
    
    outputs = []
    outputs_text = []
    for num in nums:
        for W_zerod in W_zerods:
            for L_zerod in L_zerods:
                if W_zerod:
                    np.fill_diagonal(W,0)
                d = np.sum(W, 1)
                D = np.diag(d)
                
                if num == 0: # expensive
                    sqrt_D_inv = np.diag(np.reciprocal(np.sqrt(d), where=d!=0)) # assumes D is 1 dimensional vector
                    L = sqrt_D_inv @ (D - W) @ sqrt_D_inv
                    
                
                if num == 1: # non symmetrically normalized
                    L = D-W
            
                # if num == 2: # cheap
                #     shift = 0.5
                    
                #     sqrt_D = np.diag(np.sqrt(d)) # assumes D is 1 dimensional vector
                #     D = np.diag(d)
                #     L = sqrt_D @ np.linalg.inv(D * (1-shift) - W) @ sqrt_D # no matmul for the D multiplied by constant factor
                    
                if L_zerod:
                    np.fill_diagonal(L, 0)
                outputs.append(L)
                outputs_text.append(f'{num}\nW-{W_zerod}\nL-{L_zerod}')
    return outputs, outputs_text

def get_eigfuncs():
    import numpy as np
    # import scipy
    
    # TODO: add lobpcg and a bunch of others :)
    # TODO: generalized and non-generalized forms?
    
    return [np.linalg.eigh], ['np.eigh']
    
def experiment():
    import numpy as np
    
    truth = np.zeros((28,28)) # placeholder for now
    
    radii = [1,-1]
    nums = [0]
    W_zerods = [False,True]
    L_zerods = [False,True]
    indicies = [0, 1] # smallest and second smallest... should we avoid similar ones? or avoid near 0's?
    
    e_funcs, e_funcs_text = get_eigfuncs()
    
    imgs, imgs_text = get_images()
    for img, img_name in zip(imgs, imgs_text):
        weights, weights_text = get_weights(img, radii=radii)
        for weight, weight_name in zip(weights,weights_text):
            
            plot_output = []
            
            laplaces, laplaces_text = get_laplaces(weight, nums, W_zerods, L_zerods)
            for laplace, l_name in zip(laplaces, laplaces_text):
                for eig_func, eig_name in zip(e_funcs, e_funcs_text):
                    try:
                        w,v = eig_func(laplace)
                        # TODO: calculate eigenspectrum (spread of eigenvalues)
                        
                        # compute sorted indicies
                        idx = np.argsort(w)  # index of second smallest
                        
                        for index in indicies:
                            vec = v[:,idx[index]]
                            # for vec and binarized vec (either nc cut point or split on 0?)
                                # - calculate kl divergence, l1 and l2 from known sample...
                                # - calculate absolute cosine similarity from known sample...
                                # - store histogram... maybe plot it maybe do something with?
                                # - calculate NC cutting point..?
                                # - calculate objective function(s) and constraint function(s)
                    except:
                        for index in indicies:
                            plot_output.append(np.zeros_like(img))
                        # make a blank vector to use instead
                        continue
                    



# after - test the determinism from gould paper...

