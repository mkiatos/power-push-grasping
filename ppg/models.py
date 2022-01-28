import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.downsample = downsample

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = nn.functional.relu(self.bn1(out))
        out = nn.functional.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(identity)

        out += identity
        out = nn.functional.relu(out)

        return out


class ResFCN(nn.Module):
    def __init__(self):
        super(ResFCN, self).__init__()
        self.nr_rotations = 16
        self.device = 'cuda'

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.rb1 = self.make_layer(64, 128)
        self.rb2 = self.make_layer(128, 256)
        self.rb3 = self.make_layer(256, 512)
        self.rb4 = self.make_layer(512, 256)
        self.rb5 = self.make_layer(256, 128)
        self.rb6 = self.make_layer(128, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def make_layer(self, in_channels, out_channels, blocks=1, stride=1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            # downsample = nn.Sequential(conv3x3(in_channels, out_channels, stride=stride),
            #                            nn.BatchNorm2d(out_channels))
            downsample = nn.Sequential(conv3x3(in_channels, out_channels, stride=stride))

        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def predict(self, depth):
        x = nn.functional.relu(self.conv1(depth)) # ReLu is needed?
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.rb1(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)

        x = self.rb5(x)
        x = nn.functional.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.rb6(x)
        x = nn.functional.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.final_conv(x) # activation function?
        return out

    def forward(self, depth_heightmap, specific_rotation=-1, is_volatile=[]):
        if is_volatile:
            # rotations x channel x h x w
            batch_rot_depth = torch.zeros((self.nr_rotations,
                                           1,
                                           depth_heightmap.shape[3],
                                           depth_heightmap.shape[3])).to('cuda')

            for rot_id in range(self.nr_rotations):

                # Compute sample grid for rotation before neural network
                theta = np.radians(rot_id * (360 / self.nr_rotations))
                affine_mat_before = np.array([[np.cos(theta), np.sin(theta), 0.0],
                                              [-np.sin(theta), np.cos(theta), 0.0]])
                affine_mat_before.shape = (2, 3, 1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()

                flow_grid_before = F.affine_grid(
                    Variable(affine_mat_before, requires_grad=False).to(self.device),
                    depth_heightmap.size(),
                    align_corners=True)

                # Rotate images clockwise
                rotate_depth = F.grid_sample(
                    Variable(depth_heightmap, requires_grad=False).to(self.device),
                    flow_grid_before,
                    mode='nearest',
                    align_corners=True,
                    padding_mode="border")

                batch_rot_depth[rot_id] = rotate_depth[0]

            # Compute rotated feature maps
            prob = self.predict(batch_rot_depth)

            # Undo rotation
            affine_after = torch.zeros((self.nr_rotations, 2, 3))
            for rot_id in range(self.nr_rotations):
                # Compute sample grid for rotation before neural network
                theta = np.radians(rot_id * (360 / self.nr_rotations))
                affine_mat_after = np.array([[np.cos(-theta), np.sin(-theta), 0.0],
                                             [-np.sin(-theta), np.cos(-theta), 0.0]])
                affine_mat_after.shape = (2, 3, 1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                affine_after[rot_id] = affine_mat_after

            flow_grid_after = F.affine_grid(Variable(affine_after, requires_grad=False).to(self.device),
                                            prob.size(), align_corners=True)
            out_prob = F.grid_sample(prob, flow_grid_after, mode='nearest', align_corners=True)

            # # Image-wide softmax
            # output_shape = out_prob.shape
            # out_prob = out_prob.view(output_shape[0], -1)
            # out_prob = torch.softmax(out_prob, dim=1)
            # out_prob = out_prob.view(output_shape).to(dtype=torch.float)

            return out_prob

        else:
            thetas = np.radians(specific_rotation * (360 / self.nr_rotations))

            affine_before = torch.zeros((depth_heightmap.shape[0], 2, 3))
            for i in range(len(thetas)):
                # Compute sample grid for rotation before neural network
                theta = thetas[i]
                affine_mat_before = np.array([[np.cos(theta), np.sin(theta), 0.0],
                                              [-np.sin(theta), np.cos(theta), 0.0]])
                affine_mat_before.shape = (2, 3, 1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
                affine_before[i] = affine_mat_before

            flow_grid_before = F.affine_grid(Variable(affine_before, requires_grad=False).to(self.device),
                                             depth_heightmap.size(), align_corners=True)

            # Rotate image clockwise_
            rotate_depth = F.grid_sample(Variable(depth_heightmap, requires_grad=False).to(self.device),
                                         flow_grid_before, mode='nearest', align_corners=True, padding_mode="border")

            # Compute intermediate features
            prob = self.predict(rotate_depth)

            # Compute sample grid for rotation after branches
            affine_after = torch.zeros((depth_heightmap.shape[0], 2, 3))
            for i in range(len(thetas)):
                theta = thetas[i]
                affine_mat_after = np.array([[np.cos(-theta), np.sin(-theta), 0.0],
                                             [-np.sin(-theta), np.cos(-theta), 0.0]])
                affine_mat_after.shape = (2, 3, 1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                affine_after[i] = affine_mat_after

            flow_grid_after = F.affine_grid(Variable(affine_after, requires_grad=False).to(self.device),
                                            prob.size(), align_corners=True)

            # Forward pass through branches, undo rotation on output predictions, upsample results
            out_prob = F.grid_sample(prob, flow_grid_after, mode='nearest', align_corners=True)

            # # Image-wide softmax
            # output_shape = out_prob.shape
            # out_prob = out_prob.view(output_shape[0], -1)
            # out_prob = torch.softmax(out_prob, dim=1)
            # out_prob = out_prob.view(output_shape).to(dtype=torch.float)

            return out_prob


class Classifier(nn.Module):
    def __init__(self, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.fc1 = nn.Linear(4096, 128)
        # self.fc2 = nn.Linear(4096, 256)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        # x = nn.functional.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        # x = nn.functional.relu(self.fc3(x))

        # x = torch.softmax(x, dim=1)
        return x


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.fc1 = nn.Linear(4096, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

