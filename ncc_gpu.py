import cupy as cp
import imageio
from scipy.signal import correlate2d
import numpy as np

def calculate_normalized_cross_correlation_gpu(image1_path, image2_path):
    # Read the images using imageio and convert them to CuPy arrays
    image1 = cp.asarray(imageio.imread(image1_path))
    image2 = cp.asarray(imageio.imread(image2_path))

    # Normalize the images
    image1 = (image1 - cp.mean(image1)) / cp.std(image1)
    image2 = (image2 - cp.mean(image2)) / cp.std(image2)

    # Calculate the cross-correlation using CuPy's signal processing functions
    cross_correlation = correlate2d(image1, image2, mode='same', boundary='wrap')

    # Normalize the cross-correlation
    normalized_cross_correlation = cross_correlation / cp.max(cross_correlation)

    # Convert the result back to a NumPy array if necessary for further CPU processing
    return cp.asnumpy(normalized_cross_correlation)

def calculate_mutual_information_gpu(image1_path, image2_path):
    # Read the images and convert to CuPy arrays
    image1 = cp.asarray(imageio.imread(image1_path))
    image2 = cp.asarray(imageio.imread(image2_path))

    # Calculate the joint histogram
    joint_histogram, _, _ = cp.histogram2d(
        image1.ravel(),
        image2.ravel(),
        bins=(int(image1.max()) + 1, int(image2.max()) + 1)
    )

    # Calculate the marginal histograms
    marginal_histogram_image1, _ = cp.histogram(image1, bins=(int(image1.max()) + 1))
    marginal_histogram_image2, _ = cp.histogram(image2, bins=(int(image2.max()) + 1))

    # Normalize the histograms to get probabilities
    joint_prob = joint_histogram / cp.sum(joint_histogram)
    marginal_prob_image1 = marginal_histogram_image1 / cp.sum(marginal_histogram_image1)
    marginal_prob_image2 = marginal_histogram_image2 / cp.sum(marginal_histogram_image2)

    # Calculate the joint entropy
    joint_entropy = -cp.sum(joint_prob * cp.log2(joint_prob + cp.finfo(cp.float64).eps))

    # Calculate the marginal entropies
    marginal_entropy_image1 = -cp.sum(marginal_prob_image1 * cp.log2(marginal_prob_image1 + cp.finfo(cp.float64).eps))
    marginal_entropy_image2 = -cp.sum(marginal_prob_image2 * cp.log2(marginal_prob_image2 + cp.finfo(cp.float64).eps))

    # Calculate the mutual information
    mutual_information = marginal_entropy_image1 + marginal_entropy_image2 - joint_entropy

    # Convert the result back to a NumPy array if necessary
    return cp.asnumpy(mutual_information)
image1_path = '/nfs2/baos1/rudravg/GCA007TIB_TISSUE01_DAPI_DAPI_12ms_ROUND_14_initial_reg.tif'
image2_path = '/nfs2/baos1/rudravg/GCA007TIB_TISSUE01_DAPI_DAPI_12ms_ROUND_15_initial_reg.tif'
result_ncc = calculate_normalized_cross_correlation_gpu(image1_path, image2_path)
#result_mi = calculate_mutual_information(image1_path, image2_path)
print(result_ncc)
#print(result_mi)