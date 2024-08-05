import os
import torch

def checkpoint_save(experiment_name, model, ema_model, optimizer, epoch):
    """
    Save the model's state_dict to a file.

    Parameters:
    ----------
    experiment_name : str
        The name of the experiment.

    model : torch.nn.Module
        The model to save.

    ema_model : torch.nn.Module
        The EMA model to save.

    optimizer : torch.optim.Optimizer
        The optimizer to save.

    epoch : int
        The epoch number.
    """
    save_path = "../{}/epoch_{}/checkpoints".format(experiment_name, epoch)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, f"ckpt.pth"))
        torch.save(ema_model.state_dict(), os.path.join(save_path, f"ema_ckpt.pth"))
        torch.save(optimizer.state_dict(), os.path.join(save_path, f"optim.pth"))



def checkpoint_load(model, path):
    """
    Load the model's state_dict from a file.

    Parameters:
    ----------
    model : torch.nn.Module
        The model to load the state_dict into.

    path : str
        The path to the file.
    """
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.backbone.load_state_dict(checkpoint['backbone_state_dict'])
    model.head.load_state_dict(checkpoint['head_state_dict'])
