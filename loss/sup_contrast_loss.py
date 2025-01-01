import torch
import torch.nn.functional as F

def domain_contrastive_loss(domains_features, domains_labels, temperature,device):
    anchor_feature = domains_features
    anchor_feature = F.normalize(anchor_feature, dim=1)
    labels = domains_labels
    labels= labels.contiguous().view(-1, 1)

    mask = torch.eq(labels, labels.T).float().to(device)
    anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, anchor_feature.T), temperature)

    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(anchor_feature.shape[0]).view(-1, 1).to(device), 0)
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mask_sum = mask.sum(1)
    zeros_idx = torch.where(mask_sum == 0)[0]
    mask_sum[zeros_idx] = 1
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

    loss = (- 1 * mean_log_prob_pos)
    loss = loss.mean()

    return loss