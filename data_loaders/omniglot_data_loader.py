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
    train_classes = [line.strip() for line in open(METADATA[dataset]['train'])]
    train_idx = [i for i in range(len(train_classes))]
    val_classes = [line.strip() for line in open(METADATA[dataset]['val'])]
    val_idx = [i for i in range(len(val_classes))]
    trainval_classes = [line.strip() for line in open(METADATA[dataset]['trainval'])]
    trainval_idx = [i for i in range(len(trainval_classes))]
    test_classes = [line.strip() for line in open(METADATA[dataset]['test'])]
    test_idx = [i for i in range(len(test_classes))]

    classes = {
        'train': train_classes,
        'train_idx': train_idx,
        'val': val_classes,
        'val_idx': val_idx,
        'trainval': trainval_classes,
        'trainval_idx': trainval_idx,
        'test': test_classes,
        'test_idx': test_idx
    }

    return classes


class OmniglotDataset(Dataset):
    def __init__(self, image_dir, transform, mode):
        self.image_dir = image_dir
        self.classes = load_class_partition('omniglot')
        self.transform = transform
        self.mode = mode
        self.imgs = self._make_datasets()
        self.labels = self._get_labels()

    def __getitem__(self, index):
        path, label = self.imgs[index]
        image = Image.open(os.path.join(self.image_dir, path))
        image = image.convert('L')
        return self.transform(image), label

    def _make_datasets(self):
        images = []
        if self.mode == 'train':
            categories = self.classes['train']
            indexes = self.classes['train_idx']
        elif self.mode == 'val':
            categories = self.classes['val']
            indexes = self.classes['val_idx']
        elif self.mode == 'test':
            categories = self.classes['test']
            indexes = self.classes['test_idx']
        elif self.mode == 'trainval':
            categories = self.classes['trainval']
            indexes = self.classes['trainval_idx']
        for i, category in enumerate(categories):
            for fname in os.listdir(os.path.join(self.image_dir, category.split('/rot')[0])):
                path = os.path.join(category.split('/rot')[0], fname)
                item = (path, indexes[i])
                images.append(item)
        return images

    def _get_labels(self):
        labels = []
        for img in self.imgs:
            labels.append(img[1])
        return labels

    def __len__(self):
        return len(self.imgs)


def get_omniglot_loader(image_dir, num_way, num_episodes, num_sample, mode):
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.CenterCrop(28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = OmniglotDataset(image_dir, transform, mode)

    batch_sampler = EpisodicBatchSampler(labels=dataset.labels, way=num_way, episodes=num_episodes,
                                         num_sample=num_sample)

    data_loader = DataLoader(dataset=dataset, batch_sampler=batch_sampler)

    return data_loader
