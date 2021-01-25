import numpy as np
from torch import nn, tensor
import torch
from torch.autograd import Variable


class cross_center_loss(nn.Module):
    def __init__(self, dist_type='l2'):
        super(cross_center_loss, self).__init__()

        self.dist_type = dist_type
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if dist_type == 'l1':
            self.dist = nn.L1Loss()

    def adjust_margin(self, epoch):
        if epoch < 20:
            margin = 0.1
        elif epoch >= 20 and epoch < 50:
            margin = 0.1
        elif epoch >= 50:
            margin = 0.1
        
        return margin
    
    def forward(self, feat1, feat2, label1, label2, epoch):
        margin = self.adjust_margin(epoch)
        label_num = len(label1.unique())
        feat1 = feat1.chunk(label_num, 0)
        feat2 = feat2.chunk(label_num, 0)
        for i in range(label_num):
            center1 = torch.mean(feat1[i], dim=0)
            center2 = torch.mean(feat2[i], dim=0)
            feat_num = feat1[i].size()[0]
            feat1_single = feat1[i].chunk(feat_num, 0)
            feat2_single = feat2[i].chunk(feat_num, 0)
            for j in range(feat_num):
                if j == 0:
                    r1 = max(0, self.dist(torch.squeeze(feat1_single[j]), center2) - self.adjust_margin(epoch))
                    r2 = max(0, self.dist(torch.squeeze(feat2_single[j]), center1) - self.adjust_margin(epoch))
                else:
                    r1 += max(0, self.dist(torch.squeeze(feat1_single[j]), center2) - self.adjust_margin(epoch))
                    r2 += max(0, self.dist(torch.squeeze(feat2_single[j]), center1) - self.adjust_margin(epoch))
            if i == 0:
                dist = (r1 + r2) / (2 * feat_num)
            else:
                dist += (r1 + r2) / (2 * feat_num)
        dist = dist / label_num
        return dist
