import torch

smooth = 0.0000001

def jaccard_coef(y_true, y_pred):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1 - ((intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection + smooth))