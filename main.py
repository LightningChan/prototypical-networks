import argparse
import os
import torch
import numpy as np
from data_loaders.omniglot_data_loader import get_omniglot_loader
from data_loaders.cupid_data_loader import get_cupid_loader
from solver import Solver

from torch.backends import cudnn


def omniglot(config):
    if config.mode == 'train':
        omniglot_train_loader = get_omniglot_loader(config.image_dir, config.num_train_way, config.num_train_episodes,
                                                    config.num_train_support + config.num_train_query, mode='train')

        omniglot_val_loader = get_omniglot_loader(config.image_dir, config.num_train_way, config.num_train_episodes,
                                                  config.num_train_support + config.num_train_query, mode='val')

        solver = Solver(config, train_data_loader=omniglot_train_loader, test_data_loader=None,
                        val_data_loader=omniglot_val_loader)
        solver.train()
    elif config.mode == 'test':
        omniglot_test_loader = get_omniglot_loader(config.image_dir, config.num_test_way, config.num_test_episodes,
                                             config.num_test_support + config.num_test_query, mode='test')
        solver = Solver(config, train_data_loader=None, test_data_loader=omniglot_test_loader)
        solver.test()


def cupid(config):
    if config.mode == 'train':
        cupid_train_loader = get_cupid_loader(config.image_dir, config.num_train_way, config.num_train_episodes,
                                              config.num_train_support + config.num_train_query, mode='train')
        cupid_val_loader = get_cupid_loader(config.image_dir, config.num_train_way, config.num_train_episodes,
                                            config.num_train_support + config.num_train_query, mode='val')
        solver = Solver(config, train_data_loader=cupid_train_loader, test_data_loader=None,
                        val_data_loader=cupid_val_loader)
        solver.train()
    elif config.mode == 'test':
        cupid_test_loader = get_cupid_loader(config.image_dir, config.num_test_way, config.num_test_episodes,
                                             config.num_test_support + config.num_test_query, mode='test')
        solver = Solver(config, train_data_loader=None, test_data_loader=cupid_test_loader,
                        val_data_loader=None)
        solver.test()

def init_seed(config):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)
    torch.cuda.manual_seed(config.manual_seed)

def main(config):
    # For fast training
    cudnn.benchmark = True

    # Create directories if not exist
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)

    init_seed(config)

    if config.dataset == 'omniglot':
        omniglot(config)
    elif config.dataset == 'cupid':
        cupid(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train prototypical networks')

    # Model hyper-parameters
    parser.add_argument('--num_train_episodes', type=int, default=500)
    parser.add_argument('--num_train_way', type=int, default=3)
    parser.add_argument('--num_train_support', type=int, default=5)
    parser.add_argument('--num_train_query', type=int, default=5)
    parser.add_argument('--num_test_episodes', type=int, default=100)
    parser.add_argument('--num_test_way', type=int, default=3)
    parser.add_argument('--num_test_support', type=int, default=5)
    parser.add_argument('--num_test_query', type=int, default=5)

    # Training settings
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.5)
    parser.add_argument('--lr_scheduler_step', type=int, default=60)
    parser.add_argument('--pretrained_model', type=str, default=None)

    # path
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--model_save_dir', type=str, default='./weights')
    parser.add_argument('--image_dir', type=str, default='./datasets/cupid/data')

    # Misc
    parser.add_argument('--dataset', type=str, default='cupid', choices=['omniglot', 'cupid'])
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str, default='true')
    parser.add_argument('--manual_seed', type=int, default=7)

    config = parser.parse_args()
    main(config)
