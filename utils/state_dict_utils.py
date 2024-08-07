import os
import torch

def checkpoint_save(experiment_name, model, ema_model, optimizer, epoch, scheduler=None):
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
        # save scheduler state_dict if needed
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_path, f"scheduler.pth"))



def checkpoint_load(ckp_folder, model, ema_model, optimizer, scheduler=None):
    """
    Load the model's state_dict from a file.

    Parameters:
    ----------
    model : torch.nn.Module
        The model to load the state_dict into.

    ema_model : torch.nn.Module
        The EMA model to load the state_dict into.

    optimizer : torch.optim.Optimizer
        The optimizer to load the state_dict into.

    ckp_folder : str
        The path to the folder containing the checkpoint files.
    """
    load_path = os.path.join(ckp_folder, "checkpoints")
    model.load_state_dict(torch.load(os.path.join(load_path, "ckpt.pth")))
    ema_model.load_state_dict(torch.load(os.path.join(load_path, "ema_ckpt.pth")))
    optimizer.load_state_dict(torch.load(os.path.join(load_path, "optim.pth")))
    # load scheduler state_dict if needed
    if scheduler is not None and os.path.exists(os.path.join(load_path, "scheduler.pth")):
        scheduler.load_state_dict(torch.load(os.path.join(load_path, "scheduler.pth")))

    return

