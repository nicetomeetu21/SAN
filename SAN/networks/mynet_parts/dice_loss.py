#Dice损失函数
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5
    
    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        # print(predict.shape, target.shape)
        num = predict.size(0)
        
        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)
        
        intersection = (pre * tar).sum(-1).sum()  #利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()
        
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        
        return score


class MaskDiceLoss(nn.Module):
    def __init__(self):
        super(MaskDiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target, mask):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        # print(predict.shape, target.shape)
        num = predict.size(0)

        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)
        mask = mask.view(num, -1)
        intersection = ((pre * tar)*mask).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        union = ((pre + tar)*mask).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score