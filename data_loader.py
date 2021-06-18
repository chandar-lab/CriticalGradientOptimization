"""
This dataloader comes courtesy of https://github.com/chandar-lab/PatchUp/
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


def per_image_standardization(x):
    y = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
    mean = y.mean(dim=1, keepdim=True).expand_as(y)
    std = y.std(dim=1, keepdim=True).expand_as(y)
    adjusted_std = torch.max(std, 1.0 / torch.sqrt(
        torch.cuda.FloatTensor([x.shape[1] * x.shape[2] * x.shape[3]])))
    y = (y - mean) / adjusted_std
    standarized_input = y.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    return standarized_input


def load_data(data_aug, batch_size, workers, dataset, data_target_dir):
    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]

    else:
        assert False, "Unknown dataset : {}".format(dataset)

    if data_aug == 1:
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset == 'cifar10':
        train_data = datasets.CIFAR10(data_target_dir, train=True,
                                      transform=train_transform, download=True)
        test_data = datasets.CIFAR10(data_target_dir, train=False,
                                     transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'cifar100':
        train_data = datasets.CIFAR100(data_target_dir, train=True,
                                       transform=train_transform, download=True)
        test_data = datasets.CIFAR100(data_target_dir, train=False,
                                      transform=test_transform, download=True)
        num_classes = 100
    else:
        assert False, 'Do not support dataset : {}'.format(dataset)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=workers, pin_memory=True)

    return train_loader, test_loader, num_classes


def load_data_subset(data_aug, batch_size, workers, dataset, data_target_dir,
                     labels_per_class=100,
                     valid_labels_per_class=500):
    import numpy as np
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler

    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif dataset == 'mnist':
        pass
    else:
        assert False, "Unknown dataset : {}".format(dataset)

    if data_aug == 1:
        print('data aug')
        if dataset == 'mnist':
            hw_size = 28
            train_transform = transforms.Compose([
                transforms.RandomCrop(hw_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            test_transform = transforms.Compose([
                transforms.CenterCrop(hw_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            train_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, padding=2),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        print('no data aug')
        if dataset == 'mnist':
            hw_size = 28
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        else:
            train_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset == 'cifar10':
        train_data = datasets.CIFAR10(data_target_dir, train=True,
                                      transform=train_transform, download=True)
        test_data = datasets.CIFAR10(data_target_dir, train=False,
                                     transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'cifar100':
        train_data = datasets.CIFAR100(data_target_dir, train=True,
                                       transform=train_transform, download=True)
        test_data = datasets.CIFAR100(data_target_dir, train=False,
                                      transform=test_transform, download=True)
        num_classes = 100
    elif dataset == 'mnist':
        train_data = datasets.MNIST(data_target_dir, train=True,
                                    transform=train_transform, download=True)
        test_data = datasets.MNIST(data_target_dir, train=False,
                                   transform=test_transform, download=True)
        num_classes = 10
    else:
        assert False, 'Do not support dataset : {}'.format(dataset)

    n_labels = num_classes

    def get_sampler(labels, n=None, n_valid=None):
        # Only choose digits in n_labels
        # n = number of labels per class for training
        # n_val = number of lables per class for validation
        (indices,) = np.where(
            reduce(__or__, [labels == i for i in np.arange(n_labels)]))
        # Ensure uniform distribution of labels
        np.random.shuffle(indices)

        indices_valid = np.hstack(
            [list(filter(lambda idx: labels[idx] == i, indices))[:n_valid] for i in
             range(n_labels)])
        indices_train = np.hstack(
            [list(filter(lambda idx: labels[idx] == i, indices))[n_valid:n_valid + n]
             for i in range(n_labels)])
        indices_unlabelled = np.hstack(
            [list(filter(lambda idx: labels[idx] == i, indices))[:] for i in
             range(n_labels)])

        indices_train = torch.from_numpy(indices_train)
        indices_valid = torch.from_numpy(indices_valid)
        indices_unlabelled = torch.from_numpy(indices_unlabelled)
        sampler_train = SubsetRandomSampler(indices_train)
        sampler_valid = SubsetRandomSampler(indices_valid)
        sampler_unlabelled = SubsetRandomSampler(indices_unlabelled)
        return sampler_train, sampler_valid, sampler_unlabelled

    # Dataloaders for MNIST
    if dataset == 'mnist':
        train_sampler, valid_sampler, unlabelled_sampler = get_sampler(
            train_data.targets.numpy(), labels_per_class,
            valid_labels_per_class)
    else:
        train_sampler, valid_sampler, unlabelled_sampler = get_sampler(
            train_data.targets, labels_per_class,
            valid_labels_per_class)

    labelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           sampler=train_sampler, shuffle=False,
                                           num_workers=workers, pin_memory=True)
    validation = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                             sampler=valid_sampler,
                                             shuffle=False, num_workers=workers,
                                             pin_memory=True)
    unlabelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                             sampler=unlabelled_sampler,
                                             shuffle=False, num_workers=workers,
                                             pin_memory=True)
    test = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                       num_workers=workers,
                                       pin_memory=True)

    return labelled, validation, unlabelled, test, num_classes


def imshow(img, title):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()
