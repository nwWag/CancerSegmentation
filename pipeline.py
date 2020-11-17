# Pytorch
import torchvision.datasets as datasets
from torchvision import transforms
from torch import nn
import torch
from torch.utils.data import random_split

# Models
from models.unet import UNet
from models.super_unet import SuperUNet
from models.losses import FocalLoss, BinaryIOU

# Data
from datasets.interface import HAM10000

"""
# Data
from super_selfish.data import AugmentationDataset, MomentumContrastAugmentations
"""

# Trainer
from trainer import BaseTrainer

# Test
from utils import test, draw, load


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Configuration
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Global Config
device = 'cuda'
what = 'train'

# Superviser Config
name = 'std_u'  # Storage name
lr = 1e-3
epochs = 50
batch_size = 32
pretrained_path = 'store/' + name

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Data and Model
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
train_dataset, test_dataset = random_split(HAM10000('datasets/archive/'),
                                           [8015, 2000],
                                           generator=torch.Generator().manual_seed(42))


model = UNet().to(device)
if pretrained_path is not None:
    load(model=model, name=pretrained_path)

if what == 'train':
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Trainer and Training
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    trainer = BaseTrainer(model, train_dataset, loss=FocalLoss()).to(device)

    trainer.supervise(lr=lr, epochs=epochs,
                      batch_size=batch_size, name='store/' + name)

elif what in ['train', 'validate']:
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Validate
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    test(model, test_dataset, BinaryIOU())

elif what in ['train', 'validate', 'draw']:
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Draw Example
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    draw(model, test_dataset[0])
