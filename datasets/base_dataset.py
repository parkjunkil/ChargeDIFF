import numpy as np
from imageio import imread
from PIL import Image

from termcolor import colored, cprint

import torch.nn as nn
import torch.utils.data as data

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

# Create Dataset for ChargeDIFF
def CreateDataset(opt):

    cprint('[*] Data Processing Start', 'yellow')
    cprint("[*] Could take a while if this is the first run", 'blue')
     
    from datasets.chargediff_dataset import CHARGEDIFF_Dataset
    train_dataset = CHARGEDIFF_Dataset()
    val_dataset = CHARGEDIFF_Dataset()
    test_dataset = CHARGEDIFF_Dataset()
    
    train_dataset.initialize(opt, phase = 'train')
    val_dataset.initialize(opt, phase = 'val')
    test_dataset.initialize(opt, phase = 'test')

    cprint('[*] Data Processing End', 'yellow')
    cprint("[*] Dataset has been created: %s" % (train_dataset.name()), 'blue')
    
    return train_dataset, val_dataset, test_dataset
