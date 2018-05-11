from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from PIL import Image

from data_loaders.data_helper import EpisodicBatchSampler

METADATA = {db: {'train': 'datasets/' + db + '/splits/vinyals/train.txt',
                 'val': 'datasets/' + db + '/splits/vinyals/val.txt',
                 'trainval': 'datasets/' + db + '/splits/vinyals/trainval.txt',
                 'test': 'datasets/' + db + '/splits/vinyals/test.txt'}
            for db in ('omniglot', 'cupid')}


def load_class_partition(dataset):
    train = [line.strip() for line in open(METADATA[dataset]['train'])]
    val = [line.strip() for line in open(METADATA[dataset]['val'])]
    trainval = [line.strip() for line in open(METADATA[dataset]['trainval'])]
    test = [line.strip() for line in open(METADATA[dataset]['test'])]

    classes = {
        'train': train,
        'val': val,
        'trainval': trainval,
        'test': test
    }

    return classes


class CupidDataset(Dataset):
    def __init__(self, image_dir, transform, mode):
        self.image_dir = image_dir
        self.classes = load_class_partition('cupid')
        self.transform = transform
        self.mode = mode
        self.imgs = self._make_datasets()
        self.labels = self._get_labels()

    def __getitem__(self, index):
        path, label = self.imgs[index]
        image = Image.open(os.path.join(self.image_dir, path))
        image = image.convert('RGB')
        return self.transform(image), label

    def _make_datasets(self):
        images = []
        categories = ['jiang', 'xi', 'li']
        if self.mode == 'train':
            files = self.classes['train']
        elif self.mode == 'val':
            files = self.classes['val']
        elif self.mode == 'test':
            files = self.classes['test']
        elif self.mode == 'trainval':
            files = self.classes['trainval']

        for file in files:
            category = file.split('/')[0]
            item = (file, categories.index(category))
            images.append(item)
        return images

    def _get_labels(self):
        labels = []
        for img in self.imgs:
            labels.append(img[1])
        return labels

    def __len__(self):
        return len(self.imgs)


def get_cupid_loader(image_dir, num_way, num_episodes, num_sample, mode):
    transform = transforms.Compose([
        transforms.Resize(280),
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CupidDataset(image_dir, transform, mode)

    batch_sampler = EpisodicBatchSampler(labels=dataset.labels, way=num_way, episodes=num_episodes,
                                         num_sample=num_sample)

    data_loader = DataLoader(dataset=dataset, batch_sampler=batch_sampler)

    return data_loader
