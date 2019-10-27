from sklearn.metrics import confusion_matrix
import numpy as np


def compute_iou(y_pred: np.ndarray, y_true: np.ndarray):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    iou = intersection / union.astype(np.float32)
    return np.mean(iou)
