import numpy as np
from scipy.signal import correlate2d
import imageio

def calculate_normalized_cross_correlation(image1_path, image2_path):
    # Read the images
    image1 = imageio.imread(image1_path)
    image2 = imageio.imread(image2_path)

    # Normalize the images
    image1 = (image1 - np.mean(image1)) / np.std(image1)
    image2 = (image2 - np.mean(image2)) / np.std(image2)

    # Calculate the cross-correlation
    cross_correlation = correlate2d(image1, image2, mode='same')

    # Normalize the cross-correlation
    normalized_cross_correlation = cross_correlation / np.max(cross_correlation)

    return normalized_cross_correlation

#Write function to calculate mutual information between two images
def calculate_mutual_information(image1_path, image2_path):
    # Read the images
    image1 = imageio.imread(image1_path)
    image2 = imageio.imread(image2_path)

    # Calculate the joint histogram
    joint_histogram, _, _ = np.histogram2d(
        image1.ravel(),
        image2.ravel(),
        bins=(image1.max() + 1, image2.max() + 1)
    )

    # Calculate the marginal histograms
    marginal_histogram_image1, _ = np.histogram(image1, bins=(image1.max() + 1))
    marginal_histogram_image2, _ = np.histogram(image2, bins=(image2.max() + 1))

    # Calculate the joint entropy
    joint_entropy = -np.sum(joint_histogram * np.ma.log(joint_histogram).filled(0))

    # Calculate the marginal entropies
    marginal_entropy_image1 = -np.sum(marginal_histogram_image1 * np.ma.log(marginal_histogram_image1).filled(0))
    marginal_entropy_image2 = -np.sum(marginal_histogram_image2 * np.ma.log(marginal_histogram_image2).filled(0))

    # Calculate the mutual information
    mutual_information = marginal_entropy_image1 + marginal_entropy_image2 - joint_entropy

    return mutual_information

# Example usage
image1_path = '/nfs2/baos1/rudravg/GCA007TIB_TISSUE01_DAPI_DAPI_12ms_ROUND_14_initial_reg.tif'
image2_path = '/nfs2/baos1/rudravg/GCA007TIB_TISSUE01_DAPI_DAPI_12ms_ROUND_15_initial_reg.tif'
result_ncc = calculate_normalized_cross_correlation(image1_path, image2_path)
#result_mi = calculate_mutual_information(image1_path, image2_path)
print(result_ncc)
#print(result_mi)