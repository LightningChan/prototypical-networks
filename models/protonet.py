import torch.nn as nn
import torch
from torch.nn import functional as F
from tools.utils import euclidean_dist


class ProtoNet(nn.Module):
    def __init__(self, x_dim, hid_dim, out_dim):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            self._conv_block(x_dim, hid_dim),
            self._conv_block(hid_dim, hid_dim),
            self._conv_block(hid_dim, hid_dim),
            self._conv_block(hid_dim, out_dim)
        )

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

    def loss(self, samples, labels, num_way, num_support, num_query):
        samples = samples.to('cpu')
        labels = labels.to('cpu')

        categories = torch.unique(labels)
        support_idxs = []
        query_idxs = []
        for category in categories:
            support_idxs.append(labels.eq(category).nonzero()[:num_support].squeeze())
            query_idxs.append(labels.eq(category).nonzero()[num_support:].squeeze())

        prototypes = torch.stack([samples[i].mean(0) for i in support_idxs])
        query_idxs = torch.stack(query_idxs).view(-1)

        distances = euclidean_dist(samples[query_idxs], prototypes)

        log_p_y = F.log_softmax(-distances, dim=1).view(num_way, num_query, -1)

        target_idxs = torch.arange(0, num_way).view(num_way, 1, 1).expand(num_way, num_query, 1).long()

        loss = -log_p_y.gather(2, target_idxs).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        accurary = y_hat.eq(target_idxs.squeeze()).float().mean()

        return loss, accurary
