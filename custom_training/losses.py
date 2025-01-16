import torch
from torch import nn, optim
from config import Arguments as args
import numpy as np

def min_max_weight(X):
    '''
    각 slice의 input window 마다 max-min 으로 가중치 set
    '''
    # weights = np.mean(np.max(X, axis=1),axis=-1) - np.mean(np.min(X, axis=1), axis=-1)
    weights = np.max(X, axis=1)[:,0] - np.min(X, axis=1)[:,0]
    

    return weights

def max_weight(X):
    '''
    각 slice의 input window 마다 max-min 으로 가중치 set
    '''
    weights = np.max(X, axis=1)[:,0]

    return weights


class losses(torch.nn.Module):
    def __init__(self):
        super(losses,self).__init__()
        self.loss_fn = args.loss_fn
        
        
        
    def weight_MSE(self, pred, y, weights=1):
            # criterion = nn.MSELoss(reduce=False)
            # loss = criterion(pred, y)
            loss = (pred - y)**2
            
            if args.ow != 1:
                loss_time_mean = torch.mean(loss, axis=1)
            
            else:
                loss_time_mean = loss
            
            loss = torch.mean(loss_time_mean*weights)
            
            return loss
        
        
    def weight_L1(self, pred, y, weights=1):
            criterion = nn.L1Loss(reduce=False)
            loss = criterion(pred, y)
            
            if args.ow != 1:
                loss_time_mean = torch.mean(loss, axis=1)
                
            loss = torch.mean(loss_time_mean*weights)
            
            return loss
        
        
    def __call__(self, pred, y, weights):
        if self.loss_fn == 'MSE':
            return self.weight_MSE(pred, y)
        if self.loss_fn == 'L1MSE':
            return self.weight_MSE(pred, y)
        if self.loss_fn == 'weight_MSE':
            return self.weight_MSE(pred, y, weights)
        if self.loss_fn == 'weight_L1':
            return self.weight_L1(pred, y, weights)
        if self.loss_fn == 'custom':
            return self.weight_MSE(pred, y, weights) + self.max_loss(pred, y, weights, 0.3, 0.3) + self.mean_loss(pred, y, weights, 0.3)
            # return self.max_loss(pred, y, weights, 0.3, 0.3)
        
    def max_loss(self, pred, y, weights=1, alpha=1, beta=1):
        max_pred, max_pred_idx = torch.max(pred, dim=1)
        max_y, max_y_idx = torch.max(y, dim=1)
        
        max_val_loss = torch.mean((max_pred - max_y)**2, axis=-1)
        max_time_loss = torch.mean((max_pred_idx/24 - max_y/24)**2, axis=-1)
        max_loss = alpha*torch.mean(max_val_loss*weights) + beta*torch.mean(max_time_loss*weights)
        return max_loss
        
    def mean_loss(self, pred, y, weights=1, gamma=1):
        mean_pred = torch.mean(pred, dim=1)
        mean_y = torch.mean(y, dim=1)
        
        mean_val_loss = torch.mean((mean_pred - mean_y)**2, axis=-1)
        mean_loss = gamma*torch.mean(mean_val_loss*weights)
        return mean_loss