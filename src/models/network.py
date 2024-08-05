import torch
import torch.nn as nn
import sys
import copy
sys.path.extend([".", ".."])

from torchvision import transforms as t
from itertools import chain

from models.unet import UNet, UNet_conditional
from utils.logger import get_logger


logger = get_logger()

class ANetwork(nn.Module):
    # The order for the outputs is: upper_color, lower_color, gender, bag, hat
    FINAL_RES = (64,64) 
    
    def __init__(self, conditional = True, num_classes = 10, device = 'cpu'):
        """
        Initializes the model
        """
        if not isinstance(conditional, bool):
            raise ValueError("The conditional parameter must be a boolean")
        
        if not isinstance(num_classes, int) and conditional:
            raise ValueError("The num_classes parameter must be an integer")
        
        if num_classes is not None and not conditional:
            raise ValueError("The num_classes parameter must be set only if conditional is True")
        
        super(ANetwork, self).__init__()
        self.conditional = conditional
        self.num_classes = num_classes
        self.model = UNet_conditional(num_classes = num_classes, device=device) if conditional else UNet(device=device)

        self.device = device
        self.model.to(device)
        self.eval()
        logger.info("Network initialized with {}".format("UNet_conditional" if conditional else "UNet"))

    def clone(self):
        clone = copy.deepcopy(self).eval()
        clone.model.requires_grad_(False)

        return clone

    def forward(self, Xt, t, y = None):
        """
        Forward pass of the model

        Parameters
        ----------
        Xt : torch.Tensor
            The input tensor
        t : torch.Tensor
            The time tensor
        y : torch.Tensor
            The conditional tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        if self.conditional:
            return self.model(Xt, t, y)
        else:
            return self.model(Xt, t)
        
    
    def to(self, device):
        """
        Overrides the to method in order to set the device for the backbone and the heads
        
        Parameters
        ----------
        device : str
            The device to use, can be one of ['cuda', 'cpu', 'mps']

        Raises
        ------
        ValueError
            If the device is not one of ['cuda', 'cpu', 'mps']        
        """
        if device == 'cuda':
            self.cuda()
        elif device == 'cpu':
            self.cpu()
        elif device == 'mps':
            self.mps()
        else:
            raise ValueError("The device must be one of ['cuda', 'cpu', 'mps']")

    def cuda(self):
        """
        Overrides the cuda method in order to set the device for the backbone and the heads
        """
        self.device = 'cuda'
        self.model.cuda()

    def cpu(self):
        """
        Overrides the cpu method in order to set the device for the backbone and the heads
        """
        self.device = 'cpu'
        self.model.cpu()

    def mps(self):
        """
        Overrides the mps method in order to set the device for the backbone and the heads
        """
        self.device = 'mps'
        self.model.to('mps')

    def all_parameters(self):
        """
        Returns an iterator of all the net parameters

        Returns
        -------
        param : iterator
            An iterator of all the net parameters
        """
        param = self.model.parameters()

        return iter(param)
    
    def parameters(self):
        """
        Returns an iterator of all the net parameters that require optimization.

        Returns
        -------
        param : iterator
            An iterator of all the net parameters that require optimization
        """
        param = iter([])

        # In this way we can dynamically manage the backbone being trainable or not
        par = self.model.parameters()
        param=chain(param,par)

        return iter(param)

    def eval(self):
        """
        Overrides the eval method in order to set the backbone and the heads in evaluation mode
        """
        self.model.eval()

        return self

    def train(self):
        """
        Overrides the train method in order to set the backbone and the heads in training mode
        """
        self.model.train()

        return self
    
    def get_preprocessing(self):
        """
        Returns the preprocessing pipeline for the PedNet model
        """
        return t.Compose([
        t.Resize(self.FINAL_RES),
        t.ToTensor(),
        t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __str__(self):
        """
        Returns a string representation of the model

        Returns
        -------
        str
            A string representation of the model
        """
        return str(self.model)

    def num_parameters(self):
        """
        Returns the number of parameters of the PedNet model

        Returns
        -------
        int
            The number of parameters of the PedNet model
        """
        return sum(p.numel() for p in self.all_parameters())

if __name__ == '__main__':
    pass