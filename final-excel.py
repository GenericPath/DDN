import numpy as np
import matplotlib.pyplot as plt
import os

def save_imgs(file_path, image_list):
    with open(file_path, 'ab') as file:
        for image in image_list:
            np.save(file, image)

def save_data(file_path, strings):
    with open(file_path, 'a') as file:
        for string in strings:
            file.write(string + '\n')
                   
def save_plot_imgs(image_list, labels=None, output_path='folder/', output_name='image.png', grid_size=None):
    """
    Plots all the images together in a grid layout and saves the plot as a single image.

    Parameters:
        image_list (list of numpy arrays): List of images as numpy arrays.
        labels (list of str): List of labels for each image (optional). Default is None.
        output_path (str): Path to save the output image. Default is 'output_image.png'.
        grid_size (tuple of int): Size of the grid (rows, columns) to arrange the images. If None, it is calculated based on the number of images. Default is None.
        square (bool): Whether to make the images square by padding if needed. Default is True.
    """

    # Calculate the grid size if not provided
    num_images = len(image_list)
    if grid_size is None:
        num_rows = np.ceil(np.sqrt(num_images)).astype(int)
        num_cols = np.ceil(num_images / num_rows).astype(int)
    else:
        num_rows, num_cols = grid_size

    # Create the figure and axis
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(30, 30))  # Adjust the figsize if needed
    
    output_path = os.path.join(output_path, output_name)
    fig.suptitle(output_name, fontsize=16)

    # Iterate through each image
    for i in range(num_rows):
        for j in range(num_cols):
            idx = i * num_cols + j
            if idx < num_images:
                image = image_list[idx]

                # Show the image
                axs[i, j].imshow(image)
                axs[i, j].axis('off')

                # Add label if provided
                if labels is not None and idx < len(labels):
                    axs[i, j].set_title(labels[idx])

    # Remove any remaining empty subplots
    for idx in range(num_images, num_rows * num_cols):
        fig.delaxes(axs.flatten()[idx])

    # Save the plot to a file
    plt.tight_layout()  # Adjust spacing and layout
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)  # Adjust dpi if needed
    plt.close()

def save_plot_histograms(images, output_path='folder/', output_name='image.png', num_bins=256, grid_size=None):
    num_images = len(images)
    if grid_size is None:
        num_rows = np.ceil(np.sqrt(num_images)).astype(int)
        num_cols = np.ceil(num_images / num_rows).astype(int)
        
        grid_size = (num_rows, num_cols)
    else:
        num_rows, num_cols = grid_size
    
    fig, axes = plt.subplots(*(grid_size), figsize=(30, 30), sharex=True, sharey=True)
    
    output_path = os.path.join(output_path, output_name)
    fig.suptitle(output_name, fontsize=16)

    for i, ax in enumerate(axes.ravel()):
        if i < len(images):
            image = images[i]
            # Normalize image to the range [0, 255]
            image_normalized = (image - image.min()) / (image.max() - image.min())

            # Calculate histogram
            hist, bins = np.histogram(image_normalized.flatten(), bins=num_bins, range=[0, 1])
            # Normalize histogram for better visualization
            hist = hist / hist.max()

            # Plot the histogram with a filled step plot
            ax.fill_between(bins[:-1], hist, alpha=0.75)
            ax.set_xlim([0, 256])
            ax.set_ylim([0, 1])
            # ax.set_title(f"Image {i+1}")
            # ax.set_xlabel("Pixel Value")
            # ax.set_ylabel("Normalized Frequency")

    # Remove any empty subplots
    for i in range(len(images), grid_size[0] * grid_size[1]):
        axes.ravel()[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the position of the main title

    # Save the plot to a file if the output_file is provided
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close()

def get_images(size=(28,28), filename='data/test/3.jpg'):
    """ Currently only generates a single test image """
    from skimage.io import imread
    from skimage.transform import resize
    from skimage.color import rgb2gray
    import cv2

    img_cv2 = np.array(cv2.resize(cv2.imread(filename,0), size), dtype=np.float32) 
    # float32 to remove overflows in weights matrix

    img_sk = imread("data/test/3.jpg",0)
    img_sk_gray = rgb2gray(img_sk)
    img_sk = resize(img_sk, size)
    img_sk_gray = resize(img_sk_gray, size)

    img_cv2_norm = img_cv2 / np.max(img_cv2)

    # imgs = [img_cv2_norm]
    # imgs_text = ["cv2_norm"]
    
    sk_norm = img_sk / np.max(img_sk)
    sk_255 = sk_norm * 255

    imgs = [img_cv2,img_cv2_norm,img_sk_gray, img_sk, sk_norm, sk_255]
    imgs_text = ["cv2(0,255)", "cv2(0,1)", "sk(gray)", "sk(decimal)", "sk(0,1)", "sk(0,255)"]
    print(imgs_text)
    print([f'{np.min(img):.4f}, {np.max(img):.4f}\n{img.dtype}' for img in imgs])
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
            try:
                weights.append(method(image, radius))
                weight_labels.append(f"{str(radius)},{method_name}")
            except:
                weights.append(np.zeros((784,784)))
                weight_labels.append(f"{str(radius)},{method_name}")
    try:
        weights.append(intens_posit_wm(image))
        weight_labels.append(f"{image.size},intens_posit_wm")
        weights.append(intensity_weight_matrix(image))
        weight_labels.append(f'{image.size},intensity_weight_matrix')
    except:
        weights.append(np.zeros((784,784)))
        weight_labels.append(f"{image.size},intens_posit_wm")
        weights.append(np.zeros((784,784)))
        weight_labels.append(f'{image.size},intensity_weight_matrix')
    
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
                outputs_text.append(f'{num},W-{W_zerod},L-{L_zerod}')
    return outputs, outputs_text

def get_eigfuncs():
    import scipy.linalg as linalg
    import scipy.sparse.linalg as sparse_linalg
    
    # TODO: add pytorch.linalg.eigh
    # TODO: add lobpcg and a bunch of others :)
    # TODO: generalized and non-generalized forms?
    eig_funcs = [np.linalg.eig, np.linalg.eigh, linalg.eig, linalg.eigh, sparse_linalg.eigs, sparse_linalg.eigsh]
    eig_names = ["np.eig", "np.eigh", "scipy.eig", "scipy.eigh", "scipy.sparse.eigs", "scipy.sparse.eigsh"]
    return eig_funcs, eig_names
    
def compute_kl_divergence(image1, image2):
    # Flatten the images into 1D arrays
    # Compute histograms of the pixel values
    hist1, _ = np.histogram(image1.flatten(), bins=np.linspace(0, 1, num=256)) # range of 0,1 originally from 0,255
    hist2, _ = np.histogram(image2.flatten(), bins=np.linspace(0, 1, num=256))

    total_pixels = np.sum(hist1) # normalize by the total number of pixels (so it sums to 1)
    hist1 = hist1/total_pixels
    hist2 = hist2/total_pixels

    # small constant factor to avoid division by zero, none on the first hist1 as multiply by zero is fine
    kl_divergence = np.sum(hist1 * np.log((hist1 + 1e-10) / (hist2 + 1e-10)))

    return kl_divergence   

def normalize_image(image1):
    # Normalize image1 to the range [0, 1]
    normalized_image1 = (image1 - np.min(image1)) / (np.max(image1) - np.min(image1))
    return normalized_image1 

def get_objfuncs():
    # TODO: implement this
    return [], []

def experiment():
    import os
    from datetime import datetime
    
    size = (28,28)
    imgs, imgs_text = get_images(size)
    
    truth = np.copy(imgs[1]) # cv2_norm
    truth[truth>0.5] = 1
    truth[truth<=0.5] = 0 # TODO: use a truth from a calculated one.. but this works for now
        
    radii = [1,10,784//4,-1]
    nums = [0,1]
    W_zerods = [False,True]
    L_zerods = [False,True]
    indicies = [0, 1] # smallest and second smallest... should we avoid similar ones? or avoid near 0's?
    
    e_funcs, e_funcs_text = get_eigfuncs()
    obj_funcs, const_funcs = get_objfuncs() # TODO: implement this
    
    # Create a folder name using the current date and time
    save_dir = 'results'
    date_string = datetime.now().strftime('%Y%m%d-%H%M%S')
    save_dir = os.path.join(save_dir, date_string)
    os.makedirs(save_dir, exist_ok=True)
    
    imgs = imgs+[truth.astype(float), truth*255]
    imgs_text = imgs_text+['truth(0,1)', 'truth(0,255)']
    save_plot_imgs(imgs, labels=imgs_text, output_path=save_dir,output_name=f'1images.png')
    
    for img, img_name in zip(imgs, imgs_text):
        weights, weights_text = get_weights(img, radii=radii)
        save_plot_imgs(weights, labels=weights_text, output_path=save_dir,output_name=f'{img_name},WEIGHTS.png')
        for weight, weight_name in zip(weights,weights_text):
            
            plot_output = [] # a smaller subset to plot per set of weight
            plot_labels = []
            # 'img_name, weight_name, l_name, eig_name, index, columnar, kl, l1, l2, val_min, val_max'
            data = [] # just add above column labels manually to the excel spreadsheet
            
            laplaces, laplaces_text = get_laplaces(weight, nums, W_zerods, L_zerods)
            save_plot_imgs(laplaces, labels=laplaces_text, output_path=save_dir,output_name=f'{img_name},{weight_name},LAPLACES.png')
            for laplace, l_name in zip(laplaces, laplaces_text):
                for eig_func, eig_name in zip(e_funcs, e_funcs_text):
                    try:
                        w,v = eig_func(laplace) # laplace @ v = w * laplace
                        # calculate eigenspectrum (spread of eigenvalues)
                        w_min = np.min(w)
                        w_max = np.max(w)
                        
                        idx = np.argsort(w)
                        
                        for index in indicies:
                            for columnar in [True, False]:
                                if columnar:
                                    # NOTE: np.allclose(laplace @ v[:,0],w[0] * laplace, 1e-20,1e-14)
                                    vec = v[:,idx[index]] # np.linalg.eigh is column based
                                else:
                                    vec = v[idx[index]] # some others (like lobpcg) are row..?

                                vec = np.real(vec.reshape(size))
                                
                                plot_output.append(vec)
                                plot_labels.append(f'{l_name}\n{eig_name}\n{index},{columnar}')

                                # NOTE: i think we normalize for kl but is it neccessary? and does it help other metrics?
                                vec = normalize_image(vec)
                            
                                # TODO
                                # - calculate absolute cosine similarity from known sample...
                                # - store histogram... maybe plot it maybe do something with?
                                # - calculate NC cutting point..?
                                # - calculate objective function(s) and constraint function(s)
                                kl = compute_kl_divergence(vec, truth)
                                l1 = np.abs(vec - truth).sum()
                                l2 = np.linalg.norm(vec - truth)
                                
                                data.append(f'{img_name},{weight_name},{l_name},{eig_name},{index},{columnar},{kl},{l1},{l2},{w_min},{w_max}')
                    except:
                        for index in indicies:
                            plot_output.append(np.zeros_like(img))
                        # make a blank vector to use instead
                        continue
                    
            save_imgs(os.path.join(save_dir,'data_images.np'), plot_output)
            save_data(os.path.join(save_dir,'data_output.txt'), data+['\n\n'])
            save_plot_imgs(plot_output, labels=plot_labels, output_path=save_dir,output_name=f'{img_name},{weight_name}.png')
            save_plot_histograms(plot_output, output_path=save_dir,output_name=f'{img_name},{weight_name},HISTOGRAMS.png')
                    

experiment()

# after - test the determinism from gould paper...

