import torch
import torch.nn as nn
from .base_tta_algorithm import BaseTestTimeAlgorithm, softmax_entropy
import torch.nn.functional as F
from loss.sup_contrast_loss import domain_contrastive_loss

class ACCUP(BaseTestTimeAlgorithm):

    def __init__(self, configs, hparams, model, optimizer):
        super(ACCUP, self).__init__(configs, hparams, model, optimizer)
        self.featurizer = model.feature_extractor
        self.classifier = model.classifier
        self.filter_K = hparams['filter_K']
        self.tau = hparams['tau']
        self.temperature = hparams['temperature']
        self.num_classes =  configs.num_classes
        warmup_supports = self.classifier.logits.weight.data.detach()
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_labels = F.one_hot(warmup_prob.argmax(1), num_classes = self.num_classes).float()
        self.warmup_ent = softmax_entropy(warmup_prob, warmup_prob)
        self.warmup_cls_scores = F.softmax(warmup_prob, 1)

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ents = self.warmup_ent.data
        self.cls_scores = self.warmup_cls_scores

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        raw_data, aug_data = batch_data[0], batch_data[1]
        r_feat, r_seq_feat = model.feature_extractor(raw_data)
        r_output = model.classifier(r_feat)
        a_feat, a_seq_feat = model.feature_extractor(aug_data)
        a_output = model.classifier(a_feat)
        z = (r_feat + a_feat) / 2.0
        p = (r_output + a_output) / 2.0
        yhat = F.one_hot(p.argmax(1), num_classes=self.num_classes).float()
        ent = softmax_entropy(p, p)
        cls_scores = F.softmax(p, 1)

        with torch.no_grad():
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ents = self.ents.to(z.device)
            self.cls_scores = self.cls_scores.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ents = torch.cat([self.ents, ent])
            self.cls_scores = torch.cat([self.cls_scores, cls_scores])

        supports, labels, indices = self.select_supports()
        loss = 0.
        prt_scores = self.compute_logits(z, supports, labels)
        prt_ent = softmax_entropy(prt_scores, prt_scores)
        idx = prt_ent < ent
        idx_un = idx.unsqueeze(1).expand(-1, prt_scores.shape[1])
        select_pred = torch.where(idx_un, prt_scores, cls_scores)
        pseudo_labels = select_pred.max(1, keepdim=False)[1]

        cat_p = torch.cat([r_output, a_output, p], dim=0)
        cat_pseudo_labels = pseudo_labels.repeat(3)
        loss += domain_contrastive_loss(cat_p, cat_pseudo_labels, temperature=self.temperature, device=z.device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return  select_pred

    def get_topk_neighbor(self, feature, supports, cls_scores, k_neighbor):
        feature = F.normalize(feature, dim=1)
        supports = F.normalize(supports, dim=1)
        sim_matrix = feature @ supports.T
        _, idx_near = torch.topk(sim_matrix, k_neighbor, dim=1)
        cls_score_near = cls_scores[idx_near].detach().clone()

        return cls_score_near

    def compute_logits(self, z, supports, labels):
        B, dim = z.size()
        N, dim_ = supports.size()
        assert (dim == dim_)
        temp_centroids = (labels / (labels.sum(dim=0, keepdim=True) + 1e-12)).T @ supports
        temp_z = F.normalize(z, dim=1)
        temp_centroids = F.normalize(temp_centroids, dim=1)
        logits = self.tau * temp_z @ temp_centroids.T

        return logits

    def select_supports(self):
        ent_s = self.ents
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))
        else:
            indices = []
            indices1 = torch.LongTensor(list(range(len(ent_s))))
            for i in range(self.num_classes):
                _, indices2 = torch.sort(ent_s[y_hat == i])
                indices.append(indices1[y_hat == i][indices2][:filter_K])
            indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ents = self.ents[indices]
        self.cls_scores = self.cls_scores[indices]

        return self.supports, self.labels, indices

    def configure_model(self, model):
        model.train()
        model.requires_grad_(False)

        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

        for name, module in model.feature_extractor.named_children():
            if name == 'conv_block1' or name == 'conv_block2' or name == 'conv_block3':
                for sub_module in module.children():
                    if isinstance(sub_module, nn.Conv1d):
                        sub_module.requires_grad_(True)

        return model

def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]

    return ema_model

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    return kl_div