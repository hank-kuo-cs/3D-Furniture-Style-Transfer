import torch
from ....config import config


class TripletLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s_features, p_features, n_features):
        assert s_features.size() == p_features.size() == n_features.size()

        batch_size = s_features.size(0)
        margin = config.style_extractor.feature_margin

        dist_p = abs(s_features - p_features).mean(dim=1)
        dist_n = abs(s_features - n_features).mean(dim=1)

        triplet_loss_each_batches = margin + dist_p - dist_n
        assert triplet_loss_each_batches.size() == (batch_size,)

        compare_zeros = torch.zeros(size=(batch_size,), dtype=torch.float, device=config.cuda.device)

        triplet_loss = torch.max(triplet_loss_each_batches, compare_zeros).mean()
        return triplet_loss
