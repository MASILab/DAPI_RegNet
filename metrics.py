import numpy as np
from skimage.metrics import structural_similarity as ssim

class Metrics:
    def __init__(self, img1, img2):
        self.image1 = img1
        self.image2 = img2
    def calculate_normalized_cross_correlation(self):
        # Normalize the images
        image1 = (self.image1 - np.mean(self.image1)) / np.std(self.image1)
        image2 = (self.image2 - np.mean(self.image2)) / np.std(self.image2)

        # Calculate the sum of absolute differences
        sum_of_differences = np.sum(np.abs(image1 - image2))

        return sum_of_differences
    
    def calculate_normalized_mutual_information(self):
        hist_img1, _ = np.histogram(self.image1, bins=256, range=(0, 256))
        hist_img2, _ = np.histogram(self.image2, bins=256, range=(0, 256))
        hist_2d, _, _ = np.histogram2d(self.image1.ravel(), self.image2.ravel(), bins=256, range=[[0, 256], [0, 256]])

        # Normalize histograms to get probability distributions
        hist_img1 = hist_img1 / hist_img1.sum()
        hist_img2 = hist_img2 / hist_img2.sum()
        hist_2d = hist_2d / hist_2d.sum()

        # Calculate marginal entropies
        marginal_entropy_img1 = -np.sum(hist_img1[hist_img1 > 0] * np.log2(hist_img1[hist_img1 > 0]))
        marginal_entropy_img2 = -np.sum(hist_img2[hist_img2 > 0] * np.log2(hist_img2[hist_img2 > 0]))

        # Calculate joint entropy
        joint_entropy = -np.sum(hist_2d[hist_2d > 0] * np.log2(hist_2d[hist_2d > 0]))

        # Calculate mutual information
        mutual_information = marginal_entropy_img1 + marginal_entropy_img2 - joint_entropy

        # Calculate normalized mutual information
        nmi = 2 * mutual_information / (marginal_entropy_img1 + marginal_entropy_img2)

        return nmi
    
    def calculate_ssim(self):
        # Calculate the structural similarity index
        ssim_index = ssim(self.image1, self.image2, data_range=self.image2.max() - self.image2.min(), use_sample_covariance=False)
        return ssim_index
    
    def calculate_dice(self, threshold=0.5):
        # Calculate the Dice similarity coefficient
        self.image1 = (self.image1>0).astype(int)
        self.image2 = (self.image2>0).astype(int)
        #Apply mask to the images
        intersection = np.sum(self.image1*self.image2)
        union = np.sum(self.image1) + np.sum(self.image2)
        dice = 2*intersection/union
        return dice
    def jaccard_index(self):
        # Calculate the Jaccard index
        self.image1 = (self.image1>0).astype(int)
        self.image2 = (self.image2>0).astype(int)
        #Apply mask to the images
        intersection = np.sum(self.image1*self.image2)
        union = np.sum(self.image1) + np.sum(self.image2) - intersection
        jaccard = intersection/union
        return jaccard
    def overlap_coefficient(self):
        # Calculate the overlap coefficient
        self.image1 = (self.image1>0).astype(int)
        self.image2 = (self.image2>0).astype(int)
        #Apply mask to the images
        intersection = np.sum(self.image1*self.image2)
        overlap = 2*intersection/min(np.sum(self.image1), np.sum(self.image2))
        return overlap