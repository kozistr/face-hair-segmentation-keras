import numpy as np
import pydensecrf.densecrf as dcrf

from pydensecrf.utils import unary_from_labels
from skimage.color import gray2rgb


def crf(mask_img, n_labels: int = 2):
    # Converting annotated image to RGB if it is Gray scale
    if len(mask_img.shape) < 3:
        mask_img = gray2rgb(mask_img)

    mask_img = np.asarray(mask_img, dtype=np.uint8)
    # Converting the annotations RGB color to single 32 bit integer
    annotated_label = \
        mask_img[:, :, 0] + \
        np.left_shift(mask_img[:, :, 1], 8) + \
        np.left_shift(mask_img[:, :, 2], 16)

    # Convert the 32bit integer color to 0, 1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)

    # Setting up the CRF model
    d = dcrf.DenseCRF2D(
        mask_img.shape[1], mask_img.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3),
                          compat=3,
                          kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run Inference for 10 steps
    Q = d.inference(10)

    # Find out the most probable class for each pixel.
    map_ = np.argmax(Q, axis=0)
    return map_.reshape((mask_img.shape[0], mask_img.shape[1]))
