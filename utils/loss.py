import torch
# 整個 batch 中，基於樣本的prompt 來區分來自不同prompt的文章。
def supervised_contrastive_loss(pred, target, tau=0.1):

    batch_size = pred.size()[0]

    # Calculate the similarity.
    similarities = torch.nn.functional.cosine_similarity(
        pred.unsqueeze(1), pred.unsqueeze(0), dim=-1
    )
    # print(similarities)
    # Tau is the softmax temperature
    similarities = similarities / tau

    # Calculate the sum of exponential similarity
    similarities = torch.exp(similarities)
    # print(similarities)
    supervised_contrative_mask = (target.repeat(batch_size, 1) == target.unsqueeze(dim=1).repeat(1, batch_size))
    # print(supervised_contrative_mask)

    similarities_sum = similarities.sum(dim=-1).unsqueeze(dim=-1)

    # Calculate the loss of each index.
    similarities = similarities / similarities_sum

    # Only keep the similarity of positive instance.
    similarities = torch.masked_select(
        similarities, supervised_contrative_mask
    )

    # Calculate the Average loss
    loss = torch.mean(-torch.log(similarities))

    return loss

# same input with different representation
def SimCSE_loss(pred, positive_pair, tau=0.1):
    batch_size = pred.size()[0]

    # Calculate the similarity.
    similarities = torch.nn.functional.cosine_similarity(
        pred.unsqueeze(1), positive_pair.unsqueeze(0), dim=-1
    )

    # Tau is the softmax temperature
    similarities = similarities / tau

    # Calculate the sum of exponential similarity
    similarities = torch.exp(similarities)
    similarities_sum = similarities.sum(dim=-1).unsqueeze(dim=-1)

    # Calculate the loss of each index.
    similarities = similarities / similarities_sum

    # Only keep the similarity of positive instance.
    similarities = torch.masked_select(
        similarities, torch.eye(batch_size, device=pred.device) == 1
    )

    # Calculate the Average loss
    loss = torch.mean(-torch.log(similarities))

    return loss

import torch

import torch

def margin_loss_classification(pred, target):
    """
    計算分類任務的 Margin Loss
    :param pred: 模型輸出分數 (batch_size, num_classes)
    :param target: 樣本的正確類別 (batch_size)
    :return: L1 Loss
    """
    loss_fn = torch.nn.L1Loss()

    print("Target shape before one-hot encoding:", target.size())
    print("Pred shape:", pred.size())

    batch_size, num_classes = pred.size()
    target_onehot = torch.zeros(batch_size, num_classes, device=pred.device)
    target_onehot.scatter_(1, target.unsqueeze(1), 1)
    print("Target one-hot shape:", target_onehot.size())

    pred_expanded = pred.unsqueeze(2)  # [batch_size, num_classes, 1]
    print("pred expanded",pred_expanded)
    pred_diff = pred_expanded - pred_expanded.transpose(1, 2)  # [batch_size, num_classes, num_classes]
    print("pred diff",pred_diff)
    target_onehot_expanded = target_onehot.unsqueeze(2)  # [batch_size, num_classes, 1]
    target_diff = target_onehot_expanded - target_onehot_expanded.transpose(1, 2)  # [batch_size, num_classes, num_classes]

    loss = loss_fn(pred_diff, target_diff)

    print("Margin pred shape:", pred_diff.size())
    print("Margin target shape:", target_diff.size())
    print("Loss:", loss.item())

    return loss



# def margin_loss(pred, target):
#     loss_fn = torch.nn.MSELoss(reduction='sum')
#     pred = pred.unsqueeze(0)
#     target = target.unsqueeze(0)
#     scalar = pred.size()[-1] * (pred.size()[-1] - 1)
#     return loss_fn((pred - pred.T), (target - target.T)) / scalar
