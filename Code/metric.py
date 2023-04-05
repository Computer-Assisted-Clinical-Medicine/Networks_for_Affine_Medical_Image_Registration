import numpy as np
import SimpleITK as sitk

def dice_coefficient(output, target, threshold=0.5, smooth=1e-5):
    """
    Calculates the Dice coefficient between two binary masks.

    Args:
    output (numpy.ndarray): Predicted binary mask
    target (numpy.ndarray): Ground truth/Label binary mask
    threshold (float): Threshold value for binarization (default: 0.5)
    smooth (float): Smoothing factor to avoid division by zero (default: 1e-5)

    Returns:
    float: Dice coefficient between the predicted and ground truth/label binary masks
    """
    output = output[:, :, :] > threshold
    target = target[:, :, :] > threshold
    inse = np.count_nonzero(np.logical_and(output, target))
    l = np.count_nonzero(output)
    r = np.count_nonzero(target)
    hard_dice = (2 * inse + smooth) / (l + r + smooth)
    return hard_dice

### Numpy

def mutual_information_np(output, target, bins=200):
    """
    Calculates the Mutual Information between two images using NumPy.

    Args:
    output (numpy.ndarray): Predicted image
    target (numpy.ndarray): Ground truth/Label image
    bins (int): Number of bins for histogram calculation (default: 200)

    Returns:
    float: Mutual Information between the predicted and ground truth/label images
    """
    #I(X, Y) = H(X) + H(Y) - H(X,Y)
    # https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
    output = output.ravel()
    target = target.ravel()

    c_XY = np.histogram2d(output, target, bins)[0]
    c_X = np.histogram(output, bins)[0]
    c_Y = np.histogram(target, bins)[0]

    H_X = _shannon_entropy(c_X)
    H_Y = _shannon_entropy(c_Y)
    H_XY = _shannon_entropy(c_XY)

    MI = H_X + H_Y - H_XY
    return MI

def normalized_mutual_information_np(output, target, bins=200):
    """
        Calculates the Normalized Mutual Information between two images using NumPy.

        Args:
        output (numpy.ndarray): Predicted image
        target (numpy.ndarray): Ground truth/Label image
        bins (int): Number of bins for histogram calculation (default: 200)

        Returns:
        float: Normalized Mutual Information between the predicted and ground truth/label images
        """
    # symmetric uncertainty
    mm = mutual_information_np(output, target, bins)
    c_X = np.histogram(output, bins)[0]
    c_Y = np.histogram(target, bins)[0]
    H_X = _shannon_entropy(c_X)
    H_Y = _shannon_entropy(c_Y)
    return (2 * mm) / (H_Y + H_X)

### SITK
# http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/34_Segmentation_Evaluation.html

def hausdorff_metric_sitk(output, target):
    """
            Calculates the Hausdorff Distance between two images using NumPy.

            Args:
            output (numpy.ndarray): Predicted image
            target (numpy.ndarray): Ground truth/Label image

            Returns:
            float: Hausdorff Distance between the predicted and ground truth/label images
            """
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter.Execute(target, output)
    return hausdorff_distance_filter.GetHausdorffDistance()

### Helper Functions

def _shannon_entropy(c):
    """
    Args:
        c: numpy array, 1D histogram.

    Returns:
        H: float, Shannon entropy of the input histogram.

    """
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))
    return H
