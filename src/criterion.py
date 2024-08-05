import torch.nn as nn
import torch as t
import numpy as np
import json 
import warnings
import sys
sys.path.append('..')

from constants import *
from sklearn import metrics
from pathlib import Path

from utils.logger import get_logger
logger = get_logger()

warnings.filterwarnings("ignore", category=UserWarning)

class Loss():
    """
    Class to manage the loss functions.
    """

    def __init__(self, device):

        self.device = device        
        self.loss = nn.MSELoss()

    def evaluate(self, input: t.Tensor, target: t.Tensor, mask = None) -> [t.Tensor]:
        if mask is not None:
            raise NotImplementedError("Masking is not implemented for this loss function")

        target.to(self.device)
        input.to(self.device)

        l = self.loss(target, input)

        return l