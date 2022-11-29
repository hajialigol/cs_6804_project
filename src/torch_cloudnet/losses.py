import torch
import torch.nn as nn

class JaccardLoss(nn.Module):
    def __init__(self, smooth = 0.0000001):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_true, y_pred):  
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        
        intersection = torch.sum(y_true * y_pred)
        total = torch.sum(y_true) + torch.sum(y_pred)
        union = total - intersection 
        
        jacc = (intersection + self.smooth)/(union + self.smooth)
                
        return 1 - jacc