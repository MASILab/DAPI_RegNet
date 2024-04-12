import torch
import torch.nn.functional as F
import numpy as np
import math


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()
class NMI:
    """
    Local (over window) normalized mutual information loss.
    """

    def __init__(self, bin_centers, sigma_ratio=0.5, epsilon=1e-5):
        """
        :param bin_centers: A tensor of shape [num_bins] that represents the center of each histogram bin.
        :param sigma_ratio: The ratio of the standard deviation to the mean for the Gaussian window used in density estimation.
        :param epsilon: A small value to avoid division by zero.
        """
        self.bin_centers = bin_centers
        self.sigma_ratio = sigma_ratio
        self.epsilon = epsilon

    def compute_histogram(self, x, bin_centers, sigma):
        """
        Computes the smoothed histogram for a batch of images or volumes.
        :param x: The input tensor of shape [batch_size, *vol_shape, nb_feats].
        :param bin_centers: The centers of the histogram bins.
        :param sigma: The standard deviation for the Gaussian window.
        :return: A tensor representing the smoothed histogram of the input.
        """
        x_flat = x.flatten(start_dim=1)  # Flatten the spatial dimensions
        dists = (x_flat.unsqueeze(2) - bin_centers.unsqueeze(0).unsqueeze(0))**2  # Compute the squared distances
        kernel_vals = torch.exp(-0.5 * dists / (sigma**2))  # Apply Gaussian kernel
        histogram = kernel_vals.sum(dim=1) / x_flat.size(1)  # Sum over the flattened spatial dimensions
        return histogram

    def loss(self, y_true, y_pred):
        """
        Computes the normalized mutual information loss.
        :param y_true: The ground truth tensor.
        :param y_pred: The predicted tensor.
        :return: The NMI loss value.
        """
        sigma = self.sigma_ratio * (self.bin_centers[1] - self.bin_centers[0])
        true_hist = self.compute_histogram(y_true, self.bin_centers, sigma)
        pred_hist = self.compute_histogram(y_pred, self.bin_centers, sigma)

        # Compute joint histogram
        joint_hist = torch.einsum('bi,bj->bij', true_hist, pred_hist)
        joint_hist /= joint_hist.sum() + self.epsilon  # Normalize the joint histogram

        # Compute marginal histograms
        true_marginal = true_hist.sum(dim=0) + self.epsilon
        pred_marginal = pred_hist.sum(dim=0) + self.epsilon

        # Compute entropies
        joint_entropy = -torch.sum(joint_hist * torch.log(joint_hist + self.epsilon))
        true_entropy = -torch.sum(true_marginal * torch.log(true_marginal))
        pred_entropy = -torch.sum(pred_marginal * torch.log(pred_marginal))

        # Compute NMI
        nmi = (true_entropy + pred_entropy) / joint_entropy

        return -nmi  # Return negative NMI as a loss to minimize