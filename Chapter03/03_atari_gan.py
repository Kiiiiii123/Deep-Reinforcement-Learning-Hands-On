import random
import argparse
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from tensorboardX import SummaryWriter

import gym
import gym.spaces

import numpy as np

log = gym.logger
log.setLevel(gym.logger.INFO)

LATENT_VECTOR_SIZE = 100
DISCR_FILTERS = 64
GENER_FILTERS = 64
BATCH_SIZE = 16

IMAGE_SIZE = 64

LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000


class InputWrapper(gym.ObservationWrapper):
    """
    Preprocessing of input numpy array:
    1. resize image into predefined size
    2. move color channel axis to a first place
    """
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            self.observation(old_space.low),
            self.observation(old_space.high),
            dtype = np.float32
        )

    def observation(self, observation):
        new_obs = cv2.resize(observation, (IMAGE_SIZE, IMAGE_SIZE))
        # transform (210, 160,3) -> (3, 210, 160)
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32)


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        # this pipe converges image into the single number
        self. conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=DISCR_FILTERS, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS, out_channels=DISCR_FILTERS * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS*2, out_channels=DISCR_FILTERS * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS*4, out_channels=DISCR_FILTERS * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 8, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self,x):
        conv_out = self.conv_pipe(x)
        return conv_out.view(-1, 1).squeeze(dim=1)


class Generator(nn.Module):
    def __init__(self, output_shape):
        super(Generator, self).__init__()
        # this pipe deconvolves input vector into (3, 64, 64) image
        self.deconv_pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE, out_channels=GENER_FILTERS * 8, kernel_size=4, stride=2,
                               padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 8, out_channels=GENER_FILTERS * 4, kernel_size=4, stride=2,
                               padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 4, out_channels=GENER_FILTERS * 2, kernel_size=4, stride=2,
                               padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 2, out_channels=GENER_FILTERS, kernel_size=4, stride=2,
                               padding=1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS, out_channels=output_shape[0], kernel_size=4, stride=2,
                               padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.deconv_pipe(x)






















