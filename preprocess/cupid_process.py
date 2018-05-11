import os
import numpy as np
from PIL import Image


def remove_gray_image(data_dir):
    for (root, dirs, files) in os.walk(data_dir):
        for f in files:
            img = Image.open(os.path.join(root, f))
            if img.mode not in ['RGB', 'RGBA']:
                print('removing : {}'.format(os.path.join(root, f)))
                os.remove(os.path.join(root, f))

        for dir in dirs:
            print('{}:{}'.format(dir, len(os.listdir(os.path.join(root, dir)))))


def split_dataset(data_dir, n_test=150, n_val=100):
    store_dir = '/'.join(data_dir.split('/')[:-1])
    store_dir = os.path.join(store_dir, 'splits', 'vinyals')
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)

    test = []
    val = []
    train = []
    for dir in os.listdir(data_dir):
        images = []

        for file in os.listdir(os.path.join(data_dir, dir)):
            file_path = os.path.join(dir, file)
            images.append(file_path)

        indexes = np.random.permutation(len(images))

        for i, index in enumerate(indexes):
            if i < n_test:
                test.append(images[index])
            elif i < n_test + n_val:
                val.append(images[index])
            else:
                train.append(images[index])

    with open(os.path.join(store_dir, 'train.txt'), 'w') as f:
        f.writelines(line + '\n' for line in train)

    with open(os.path.join(store_dir, 'val.txt'), 'w') as f:
        f.writelines(line + '\n' for line in val)

    with open(os.path.join(store_dir, 'test.txt'), 'w') as f:
        f.writelines(line + '\n' for line in test)

    with open(os.path.join(store_dir, 'trainval.txt'), 'w') as f:
        f.writelines(line + '\n' for line in train)
        f.writelines(line + '\n' for line in val)

    print('train:{},val:{},test:{}'.format(len(train), len(val), len(test)))


def main():
    data_dir = '../datasets/cupid/data'
    remove_gray_image(data_dir)
    split_dataset(data_dir)


if __name__ == '__main__':
    main()
