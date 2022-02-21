import numpy as np
import cv2
import random
import os
import pickle
import shutil
import matplotlib.pyplot as plt

from torch.utils import data
import torch.optim as optim
import torch.nn as nn
import torch

from ppg.models import ResFCN, Classifier, Regressor
from ppg.utils import utils


class HeightmapDataset(data.Dataset):
    def __init__(self, dataset_dir, dir_ids):
        super(HeightmapDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.dir_ids = dir_ids

        # self.widths = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1]) # ToDo: Hardcoded
        self.widths = np.array([0.6])

    def __getitem__(self, idx):
        # print(os.path.join(self.dataset_dir, self.dir_ids[idx]))
        heightmap = cv2.imread(os.path.join(self.dataset_dir, self.dir_ids[idx], 'heightmap.exr'), -1)
        action = pickle.load(open(os.path.join(self.dataset_dir, self.dir_ids[idx], 'action'), 'rb'))

        diagonal_length = float(heightmap.shape[0]) * np.sqrt(2)
        diagonal_length = np.ceil(diagonal_length / 16) * 16
        padding_width = int((diagonal_length - heightmap.shape[0]) / 2)
        padded_heightmap = np.pad(heightmap, padding_width, 'constant', constant_values=-0.01)

        # Normalize heightmap.
        image_mean = 0.01
        image_std = 0.03
        padded_heightmap = (padded_heightmap - image_mean) / image_std

        # Add extra channel.
        padded_heightmap = np.expand_dims(padded_heightmap, axis=0)

        # Convert theta to range 0-360 and then compute the rot_id
        angle = (action[2] + (2 * np.pi)) % (2 * np.pi)
        rot_id = round(angle / (2 * np.pi / 16))

        action_area = np.zeros((heightmap.shape[0], heightmap.shape[1]))
        action_area[int(action[1]), int(action[0])] = 1.0
        label = np.zeros((1, padded_heightmap.shape[1], padded_heightmap.shape[2]))
        label[0, padding_width:padded_heightmap.shape[1] - padding_width,
                 padding_width:padded_heightmap.shape[2] - padding_width] = action_area

        return padded_heightmap, rot_id, label

    def __len__(self):
        return len(self.dir_ids)


class ApertureDataset(data.Dataset):
    def __init__(self, dataset_dir, dir_ids):
        super(ApertureDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.dir_ids = dir_ids
        self.widths = np.array([0.6, 0.8, 1.1]) # ToDo: Hardcoded
        self.crop_size = 32
        self.plot = False

    def __getitem__(self, idx):
        # print(os.path.join(self.dataset_dir, self.dir_ids[idx], 'action'))
        heightmap = cv2.imread(os.path.join(self.dataset_dir, self.dir_ids[idx], 'heightmap.exr'), -1)
        action = pickle.load(open(os.path.join(self.dataset_dir, self.dir_ids[idx], 'action'), 'rb'))

        # Add extra padding (to handle rotations inside network)
        diag_length = float(heightmap.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 16) * 16
        padding_width = int((diag_length - heightmap.shape[0]) / 2)
        depth_heightmap = np.pad(heightmap, padding_width, 'constant')
        padded_shape = depth_heightmap.shape

        # Rotate image (push always on the right)
        p1 = np.array([action[0], action[1]]) + padding_width
        theta = -((action[2] + (2 * np.pi)) % (2 * np.pi))
        rot = cv2.getRotationMatrix2D((int(padded_shape[0] / 2), int(padded_shape[1] / 2)),
                                      theta * 180 / np.pi, 1.0)
        rotated_heightmap = cv2.warpAffine(depth_heightmap, rot, (padded_shape[0], padded_shape[1]))

        # Compute the position of p1 on rotated heightmap
        rotated_pt = np.dot(rot, (p1[0] , p1[1], 1.0))
        rotated_pt = (int(rotated_pt[0]), int(rotated_pt[1]))

        # Crop heightmap
        cropped_map = np.zeros((2 * self.crop_size, 2 * self.crop_size), dtype=np.float32)
        y_start = max(0, rotated_pt[1] - self.crop_size)
        y_end = min(padded_shape[0], rotated_pt[1] + self.crop_size)
        x_start = rotated_pt[0]
        x_end = min(padded_shape[0], rotated_pt[0] + 2 * self.crop_size)
        cropped_map[0:y_end - y_start, 0:x_end - x_start] = rotated_heightmap[y_start: y_end, x_start: x_end]

        if self.plot:
            print(action[3])

            p2 = np.array([0, 0])
            p2[0] = p1[0] + 20 * np.cos(action[2])
            p2[1] = p1[1] - 20 * np.sin(action[2])

            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(depth_heightmap)
            ax[0].plot(p1[0], p1[1], 'o', 2)
            ax[0].plot(p2[0], p2[1], 'x', 2)
            ax[0].arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], width=1)

            rotated_p2 = np.array([0, 0])
            rotated_p2[0] = rotated_pt[0] + 20 * np.cos(0)
            rotated_p2[1] = rotated_pt[1] - 20 * np.sin(0)
            ax[1].imshow(rotated_heightmap)
            ax[1].plot(rotated_pt[0], rotated_pt[1], 'o', 2)
            ax[1].plot(rotated_p2[0], rotated_p2[1], 'x', 2)
            ax[1].arrow(rotated_pt[0], rotated_pt[1], rotated_p2[0] - rotated_pt[0], rotated_p2[1] - rotated_pt[1], width=1)

            ax[2].imshow(cropped_map)
            plt.show()

        # Normalize maps ( ToDo: find mean and std) # Todo
        image_mean = 0.01
        image_std = 0.03
        cropped_map = (cropped_map - image_mean) / image_std

        # Add extra channel
        # cropped_map = np.expand_dims(cropped_map, axis=0)
        three_channel_map = np.zeros((3, 2*self.crop_size, 2*self.crop_size))
        three_channel_map[0] = cropped_map
        three_channel_map[1] = cropped_map
        three_channel_map[2] = cropped_map

        aperture = action[3]
        # opening_id = np.argwhere(self.widths == action['opening']['min_width'])[0, 0]
        # one_hot = np.zeros((len(self.widths),))
        # one_hot[opening_id] = 1.0
        normalized_aperture = utils.min_max_scale(action[3], range=[0.6, 1.1], target_range=[0, 1])

        return three_channel_map, np.array([normalized_aperture])

    def __len__(self):
        return len(self.dir_ids)


def train_fcn_net(params):

    # Create dir for model weights
    if os.path.exists(params['log_path']):
        print('Directory ', params['log_path'], 'exists, do you want to remove it? (y/n)')
        answer = input('')
        if answer == 'y':
            shutil.rmtree(params['log_path'])
            os.mkdir(params['log_path'])
        else:
            exit()
    else:
        os.mkdir(params['log_path'])

    transition_dirs = next(os.walk(params['dataset_dir']))[1]

    # Split data to training/validation
    random.seed(0)
    random.shuffle(transition_dirs)
    train_ids = transition_dirs[:int(params['split_ratio'] * len(transition_dirs))]
    val_ids = transition_dirs[int(params['split_ratio'] * len(transition_dirs)):]
    # train_ids = train_ids[::4]
    # val_ids = val_ids[::4]

    train_dataset = HeightmapDataset(params['dataset_dir'], train_ids)
    val_dataset = HeightmapDataset(params['dataset_dir'], val_ids)

    data_loader_train = data.DataLoader(train_dataset,
                                        batch_size=params['batch_size'],
                                        shuffle=True)
    data_loader_val = data.DataLoader(val_dataset, batch_size=params['batch_size'])
    data_loaders = {'train': data_loader_train, 'val': data_loader_val}
    print('{} training data, {} validation data'.format(len(train_ids), len(val_ids)))

    model = ResFCN().to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    # criterion = nn.SmoothL1Loss(reduction='none')
    criterion = nn.BCELoss(reduction='none')

    for epoch in range(params['epochs']):
        model.train()
        for batch in data_loader_train:
            x = batch[0].to('cuda')
            rotations = batch[1]
            y = batch[2].to('cuda', dtype=torch.float)

            pred = model(x, specific_rotation=rotations)

            # Compute loss in the whole scene
            loss = criterion(pred, y)
            loss = torch.sum(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        epoch_loss = {'train': 0.0, 'val': 0.0}
        for phase in ['train', 'val']:
            for batch in data_loaders[phase]:
                x = batch[0].to('cuda')
                rotations = batch[1]
                y = batch[2].to('cuda', dtype=torch.float)

                pred = model(x, specific_rotation=rotations)

                loss = criterion(pred, y)
                loss = torch.sum(loss)
                epoch_loss[phase] += loss.detach().cpu().numpy()

        # Save model
        if epoch % 1 == 0:
            torch.save(model.state_dict(), os.path.join(params['log_path'], 'model_' + str(epoch) + '.pt'))

        print('Epoch {}: training loss = {:.4f} '
              ', validation loss = {:.4f}'.format(epoch, epoch_loss['train'] / len(data_loaders['train']),
                                                  epoch_loss['val'] / len(data_loaders['val'])))


def train_classifier(params):
    # Create dir for model weights
    if os.path.exists(params['log_path']):
        print('Directory ', params['log_path'], 'exists, do you want to remove it? (y/n)')
        answer = input('')
        if answer == 'y':
            shutil.rmtree(params['log_path'])
            os.mkdir(params['log_path'])
        else:
            exit()
    else:
        os.mkdir(params['log_path'])

    transition_dirs = next(os.walk(params['dataset_dir']))[1]
    transition_dirs = transition_dirs[:int(0.95 * len(transition_dirs))]

    # Split data to training/validation
    random.seed(0)
    random.shuffle(transition_dirs)
    train_ids = transition_dirs[:int(params['split_ratio'] * len(transition_dirs))]
    val_ids = transition_dirs[int(params['split_ratio'] * len(transition_dirs)):]

    train_dataset = ApertureDataset(params['dataset_dir'], train_ids)
    val_dataset = ApertureDataset(params['dataset_dir'], val_ids)

    data_loader_train = data.DataLoader(train_dataset,
                                        batch_size=params['batch_size'],
                                        shuffle=True)
    data_loader_val = data.DataLoader(val_dataset,
                                      batch_size=params['batch_size'])
    data_loaders = {'train': data_loader_train, 'val': data_loader_val}
    print('{} training data, {} validation data'.format(len(train_ids), len(val_ids)))

    # model = Classifier(n_classes=3).to('cuda')
    model = Regressor().to('cuda')

    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.SmoothL1Loss()

    for epoch in range(params['epochs']):
        model.train()
        for batch in data_loader_train:
            x = batch[0].to('cuda', dtype=torch.float32)
            # y = batch[1].to('cuda', dtype=torch.long)
            y = batch[1].to('cuda', dtype=torch.float32)

            pred = model(x)

            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        epoch_loss = {'train': 0.0, 'val': 0.0}
        # accuraccy = {'train': 0.0, 'val': 0.0}
        for phase in ['train', 'val']:
            for batch in data_loaders[phase]:
                x = batch[0].to('cuda', dtype=torch.float32)
                # y = batch[1].to('cuda', dtype=torch.long)
                y = batch[1].to('cuda', dtype=torch.float32)

                pred = model(x)

                loss = criterion(pred, y)
                loss = torch.sum(loss)
                epoch_loss[phase] += loss.detach().cpu().numpy()

                # Compute classification accuracy
                # max_id = torch.argmax(pred, axis=1).detach().cpu().numpy()
                # for i in range(len(max_id)):
                #     if max_id[i] == y[i].detach().cpu().numpy():
                #         accuraccy[phase] += 1

        # Save model
        if epoch % 1 == 0:
            torch.save(model.state_dict(), os.path.join(params['log_path'], 'model_' + str(epoch) + '.pt'))

        print('Epoch:', epoch)
        print('loss: train/val {:.4f}/{:.4f}'.format(epoch_loss['train'] / len(data_loaders['train']),
                                                      epoch_loss['val'] / len(data_loaders['val'])))

        # print('accuracy: train/val {:.4f}/{:.4f}'.format(accuraccy['train'] / len(train_dataset),
        #                                                  accuraccy['val'] / len(val_dataset)))
        print('-----------------------')


def eval_aperture_net(params):
    transition_dirs = next(os.walk(params['dataset_dir']))[1]
    test_transitions = transition_dirs[int(0.95 * len(transition_dirs)):]

    test_dataset = ApertureDataset(params['dataset_dir'], test_transitions)
    data_loader_test = data.DataLoader(test_dataset, batch_size=1)

    print('{} test data'.format(len(test_dataset)))

    # model = Classifier(n_classes=3).to('cuda')
    model = Regressor().to('cuda')
    model.load_state_dict(torch.load('../logs/regressor/model_10.pt'))
    model.eval()

    for batch in data_loader_test:
        x = batch[0].to('cuda', dtype=torch.float32)
        y = batch[1].to('cuda', dtype=torch.float32)

        pred = model(x)

        pred = utils.min_max_scale(pred.detach().cpu().numpy(), range=[0, 1], target_range=[0.6, 1.1])
        y = utils.min_max_scale(y.detach().cpu().numpy(), range=[0, 1], target_range=[0.6, 1.1])

        print(pred, y)
        input('')