import torch
import torch.nn as nn
import torch.nn.functional as F



class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        # self.weight = torch.tensor([0.11360945,  0.82000506, 16.37525253,  2.4017037 ,  6.97268817,
        #                             0.57981044,  2.57325397, 22.51597222,  0.90516471, 11.02823129,
        #                             2.84412281,  0.25257459,  1.08293253,  3.55515351,  2.02390762,
        #                             0.95643068,  2.59176659,  0.30299037,  0.87582388,  5.68824561,
        #                             2.08642214,  6.51064257, 27.01916667,  5.59982729,  7.77529976,
        #                             11.25798611, 13.1800813 ,  7.94681373,  1.35945493,  0.51489598]).to(device=device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )
    


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=30, smoothing=0.5, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    
## F1Loss 작성중 ##

###################
criterion = {'label_smoothing': LabelSmoothingLoss, 'focal_loss': FocalLoss}

## 원하는 criterion loss 사용!
def use_criterion(criterion_n, **kwargs):
    choose_criterion=criterion[criterion_n]
    return choose_criterion(**kwargs)