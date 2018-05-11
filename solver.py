import torch
from torch.autograd import Variable
import numpy as np
from models.protonet import ProtoNet

import os

import time
import datetime


class Solver(object):
    def __init__(self, config, train_data_loader, test_data_loader, val_data_loader=None):
        # Data loader
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader

        self.config = config
        self.build_model()

        # Build tensorboard if use
        if self.config.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.config.pretrained_model is not None:
            self.load_pretrained_model()

    def load_pretrained_model(self):
        self.protonet.load_state_dict(
            torch.load(os.path.join(self.config.model_save_dir, self.config.pretrained_model)))

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def build_model(self):
        self.protonet = ProtoNet(x_dim=3, hid_dim=64, out_dim=64)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.protonet.parameters(), self.config.lr)

        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,
                                                            gamma=self.config.lr_scheduler_gamma,
                                                            step_size=self.config.lr_scheduler_step)
        if torch.cuda.is_available():
            self.protonet.cuda()

    def build_tensorboard(self):
        from tools.logger import Logger
        self.logger = Logger(self.config.log_dir)

    def train(self):

        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        best_acc = 0

        best_model_path = os.path.join(self.config.model_save_dir, 'best_model.pth')
        last_model_path = os.path.join(self.config.model_save_dir, 'last_model.pth')

        # Start training
        start_time = time.time()
        for e in range(self.config.num_epochs):
            print('=== Epoch: {} ==='.format(e))
            for i, (images, labels) in enumerate(self.train_data_loader):
                self.optimizer.zero_grad()
                images = self.to_var(images)
                labels = self.to_var(labels)
                features = self.protonet(images)
                loss, acc = self.protonet.loss(samples=features, labels=labels,
                                               num_way=self.config.num_train_way,
                                               num_support=self.config.num_train_support,
                                               num_query=self.config.num_train_query)
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
                train_acc.append(acc.item())
            train_avg_loss = np.mean(train_loss[-self.config.num_train_episodes:])
            train_avg_acc = np.mean(train_acc[-self.config.num_train_episodes:])
            print('Avg Train Loss: {}, Avg Train Acc: {}'.format(train_avg_loss, train_avg_acc))
            self.lr_scheduler.step()

            # Logging
            log = {}
            log['train_avg_loss'] = train_avg_loss
            log['train_avg_acc'] = train_avg_acc

            if self.val_data_loader is not None:
                # Start Validating
                for images, labels in self.val_data_loader:
                    images = self.to_var(images)
                    labels = self.to_var(labels)
                    features = self.protonet(images)
                    loss, acc = self.protonet.loss(samples=features, labels=labels,
                                                   num_way=self.config.num_train_way,
                                                   num_support=self.config.num_train_support,
                                                   num_query=self.config.num_train_query)
                    val_loss.append(loss.item())
                    val_acc.append(acc.item())
                val_avg_loss = np.mean(val_loss[-self.config.num_train_episodes:])
                val_avg_acc = np.mean(val_acc[-self.config.num_train_episodes:])
                postfix = ' (Best)' if val_avg_acc >= best_acc else ' (Best: {})'.format(
                    best_acc)
                print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
                    val_avg_loss, val_avg_acc, postfix))
                if val_avg_acc >= best_acc:
                    torch.save(self.protonet.state_dict(), best_model_path)
                    best_acc = val_avg_acc

                log['val_avg_loss'] = val_avg_loss
                log['val_avg_acc'] = val_avg_acc

            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            print('Time:{}'.format(elapsed))

            if self.config.use_tensorboard:
                for tag, value in log.items():
                    self.logger.scalar_summary(tag, value, e + 1)

        torch.save(self.protonet.state_dict(), last_model_path)

    def test(self):
        best_model_path = os.path.join(self.config.model_save_dir, 'best_model.pth')
        self.protonet.load_state_dict(torch.load(best_model_path))
        avg_acc = list()
        for e in range(self.config.num_test_episodes):
            for images, labels in self.test_data_loader:
                images = self.to_var(images)
                labels = self.to_var(labels)
                features = self.protonet(images)
                loss, acc = self.protonet.loss(samples=features, labels=labels,
                                               num_way=self.config.num_train_way,
                                               num_support=self.config.num_train_support,
                                               num_query=self.config.num_train_query)
                avg_acc.append(acc.item())
        avg_acc = np.mean(avg_acc)
        print('Test Acc: {}'.format(avg_acc))
        return avg_acc
