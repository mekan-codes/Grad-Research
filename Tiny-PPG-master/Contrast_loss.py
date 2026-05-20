import torch
import torch.nn as nn
from abc import ABC


class ContrastLoss(nn.Module, ABC):
    def __init__(self):
        super(ContrastLoss, self).__init__()

        self.temperature = 0.07
        self.ignore_label = -1
        self.max_samples = 1024
        self.max_views = 100

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [
                x
                for x in this_classes
                if (this_y == x).nonzero().shape[0] > self.max_views
            ]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        device = X.device
        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float, device=device)
        y_ = torch.zeros(total_classes, dtype=torch.float, device=device)

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep

                hard_perm = torch.randperm(num_hard, device=hard_indices.device)
                easy_perm = torch.randperm(num_easy, device=easy_indices.device)
                hard_indices = hard_indices[hard_perm[:num_hard_keep]]
                easy_indices = easy_indices[easy_perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
            self.temperature,
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(
            1,
            torch.arange(anchor_num * anchor_count, device=mask.device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits + neg_logits)

        positive_pairs = mask.sum(1).clamp_min(1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / positive_pairs

        loss = -1 * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None):
        labels = torch.nn.functional.interpolate(labels, (feats.shape[2]), mode="nearest")
        predict = torch.nn.functional.interpolate(predict, (feats.shape[2]), mode="nearest")

        labels = labels + 1
        predict = predict + 1

        feats = feats.permute(0, 2, 1)

        labels = labels.contiguous().view(labels.shape[0], -1)
        predict = predict.contiguous().view(predict.shape[0], -1)

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)
        if feats_ is None:
            return feats.sum() * 0.0

        loss = self._contrastive(feats_, labels_)
        return loss
