from keras import backend as K
from torch import tensor, flatten, Tensor

smooth = 0.0000001


def jacc_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth))


def jacc_coef_pt(y_true: Tensor, y_pred: Tensor):
    y_true = flatten(y_true)
    y_pred = flatten(y_pred)
    intersection = sum(y_true * y_pred).item()
    union = sum(y_true).item() + sum(y_pred).item() - intersection
    coef = - (
            (intersection + smooth) / (union + smooth)
    )
    return tensor(coef, requires_grad=True)
