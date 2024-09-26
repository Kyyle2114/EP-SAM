import torch
import torch.nn.functional as F

# class FocalLoss(torch.nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0):
#         """
#         Focal Loss for Pytorch
#         Input tensor should be 3-dimensional (N(batch_size), H, W), contains values of either 0(False) or 1(True).
#         Alpha is the weighting factor for the class, and gamma is focusing parameter.
#         """
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma

#     def forward(self, pred, target):
#         # Ensure pred is a probability (i.e., apply sigmoid)
#         pred = torch.sigmoid(pred)
#         ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
#         p_t = pred * target + (1 - pred) * (1 - target)
#         fl = ce_loss * ((1 - p_t) ** self.gamma)
#         fl = self.alpha * fl.mean()

#         return fl

# class FocalLoss(torch.nn.Module):
#     def __init__(self, alpha=0.25, gamma=2):
#         super(FocalLoss, self).__init__()
#         self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
#         self.gamma = gamma

#     def forward(self, inputs, targets):
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         targets = targets.type(torch.long)
#         at = self.alpha.gather(0, targets.data.view(-1))
#         pt = torch.exp(-BCE_loss)
#         F_loss = at*(1-pt)**self.gamma * BCE_loss
#         return F_loss.mean()

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.1, gamma=2, device='cuda'):
        super(FocalLoss, self).__init__()
        # alpha 값을 cuda 텐서로 초기화
        self.alpha = torch.tensor([alpha, 1-alpha], device=device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Binary Cross Entropy Loss 계산
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # targets를 long 타입으로 변경
        targets = targets.type(torch.long)
        
        # alpha 값 맞춤 조정
        # targets의 각 요소에 대한 alpha 값을 가져와서 inputs와 같은 크기로 조정합니다.
        at = self.alpha.gather(0, targets.view(-1)).view_as(targets)
        
        # pt (확률 p_t) 계산
        pt = torch.exp(-BCE_loss)
        
        # Focal Loss 계산
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        
        # 배치에 대한 평균 Focal Loss 반환
        return F_loss.mean()