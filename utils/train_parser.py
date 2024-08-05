import os
import sys
import torch

from argparse import ArgumentParser

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from train_hp import WORKERS, BATCH_SIZE, LEARNING_RATE, EPOCHS, OPTIMIZER

strToOptim = {
    'adamw': torch.optim.AdamW,
    'sgd': torch.optim.SGD
}

def parse_args():
    """
    Parse the arguments for the training script.
    The dataset organization requested is the following:
        args.data:
            |---> face1
            |       |->image1.jpg
            |       L->image2.jpg
            |       L->...
            |
            L---> face2
                    |->image1.jpg
                    L->image2.jpg
                    L->...
    Returns
    -------
    args : argparse.Namespace
        The parsed arguments.
    """
    parser = ArgumentParser()

    # checkpoint
    # The checkpoint name must comprise the string '_epoch_N' in order to start from epoch N
    parser.add_argument('--data', dest='data',
                        help="Path to the dataset folder", default=None)
    parser.add_argument("--checkpoint", dest="checkpoint", help="path to checkpoint to restore",
                        default=None, type=str)

    # training hyper-parameters
    parser.add_argument('--nw', dest='nw', help="number of workers for dataloader",
                        default=WORKERS, type=int)
    parser.add_argument('--bs', dest='bs', help="batch size",
                        default=BATCH_SIZE, type=int)
    parser.add_argument('--lr', dest='lr', help="learning rate",
                        default=LEARNING_RATE, type=float)
    parser.add_argument('--device', dest='device', help="device to use",
                        default='cpu', type=str)
    parser.add_argument('--sched', dest='sched', action="store_true", help="Use or not use the learning rate scheduler",
                        default=False)
    parser.add_argument('--epochs', dest='epochs', help="Number of epochs to train",
                        default=EPOCHS, type=int)
    parser.add_argument('--severity', dest='severity', help='Logging severity level',
                        default='INFO', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    parser.add_argument('--profile', dest='profile', action="store_true", 
                        help="Profile the training by plotting mean time and memory consumed during validation",)
    parser.add_argument('--optim', dest='optim', help="Optimizer to use", 
                        type=str, choices=['adamw', 'sgd']) # TODO: enhance
    parser.add_argument('--cond', dest='conditional', action="store_true", help="If using a conditional model",
                        default=True)
    parser.add_argument('--classes', dest='classes', help="Number of classes for the conditional model",
                        default=10, type=int)

    args = parser.parse_args()

    return args
