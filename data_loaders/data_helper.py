import torch
import numpy as np


class EpisodicBatchSampler(object):
    def __init__(self, labels, way, episodes, num_sample):
        self.labels = labels
        self.way = way
        self.episodes = episodes
        self.num_sample = num_sample

        self.categories, self.counts = np.unique(self.labels, return_counts=True)
        self.categories = torch.LongTensor(self.categories)
        self.counts = torch.LongTensor(self.counts)

        self.matrix = np.empty((len(self.categories), max(self.counts)), dtype=int) * np.nan
        for idx, label in enumerate(self.labels):
            label_index = np.argwhere(self.categories == label).item()
            self.matrix[label_index, np.where(np.isnan(self.matrix[label_index]))[0][0]] = idx

    def __len__(self):
        return self.episodes

    def __iter__(self):
        for i in range(self.episodes):
            batch = []
            category_idxs = torch.randperm(len(self.categories))[:self.way]
            for category in self.categories[category_idxs]:
                label_index = np.argwhere(self.categories == category).item()
                sample_indices = torch.randperm(self.counts[label_index])[:self.num_sample]
                batch.append(torch.LongTensor(self.matrix[label_index][sample_indices]))

            batch = torch.cat(batch, 0)
            batch = batch[torch.randperm(len(batch))]
            yield batch
