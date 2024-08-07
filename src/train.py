import sys
sys.path.append("..")
import torch
import re
import traceback
import logging
import numpy as np
import os
import torchvision


from time import localtime, strftime
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.datasets import ImageFolder

from datasets import ImageDataset
from models.network import ANetwork
from models.ema import EMA
from diffuser import Diffuser
from criterion import Loss
from train_hp import *
from constants import METRICS, BOT_TOKEN
from utils.state_dict_utils import checkpoint_save, checkpoint_load
from utils.train_plots import *
from utils.train_parser import parse_args, strToOptim
from utils.telegram_bot import TelegramBot, update_telegram
from utils.logger import get_logger, set_level
from utils.profiler import Profile

global bot 
bot = TelegramBot(token = BOT_TOKEN)

global profiler
profiler = None

logger = None 
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

torch.autograd.set_detect_anomaly(False)

def mean(l):
    return sum(l) / len(l)

def one_epoch(model, diffuser, criterion, optimizer, train_loader, 
              device, ema, ema_model):

    model.train()
    batches_done = 0
    
    pbar = tqdm(train_loader, desc='Train')
    for X, y in pbar:

        X = X.to(device).float()
        t = diffuser.sample_timesteps(X.shape[0]).to(device)
        Xt, noise = diffuser.noise_images(X, t)

        if model.conditional:
            y = y.to(device).long()

            # TODO: motivate
            if np.random.random() < 0.1:
                y = None

            # print type(Xt), type(t), type(y)
            # print(Xt.shape, t.shape, y.shape)
            # print(Xt.dtype, t.dtype, y.dtype)
            # print(Xt.device, t.device, y.device)
            predicted_noise = model(Xt, t, y)
        else:
            predicted_noise = model(Xt, t)

        loss = criterion.evaluate(predicted_noise, noise)

        loss.backward()

        optimizer.step()

        ema.step_ema(ema_model, model)

        optimizer.zero_grad()

        batches_done += 1
        pbar.set_postfix(MSE=loss.item())

    # if batches_done != train_loader.batch_sampler.sampler.num_batches:
    #     warnings.warn(
    #         "Number of batches done during training is not sufficient to cover the whole training set. "
    #         "{} out of {} done.".format(batches_done, train_loader.batch_sampler.sampler.num_batches),
    #         RuntimeWarning
    #     )

    
    return

def train(model, optimizer, diffuser, ema, start_epoch, epochs, dataloader, 
          criterion, device, experiment_name, scheduler=False, ema_model=None):
    model.train()

    if ema_model is None:
        ema_model = model.clone()

    # training
    training_steps = 0

    last_epoch = start_epoch

    for epoch in range(start_epoch, epochs):
        logger.info(f'EPOCH {epoch+1} out of {epochs}')

        one_epoch(model, diffuser, criterion, optimizer, dataloader, device, ema, ema_model)

        if (epoch+1) % SAVE_PERIOD == 0 or epoch == epochs-1:
            labels = torch.arange(5).long().to(device)
            sampled_images = diffuser.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffuser.sample(ema_model, n=len(labels), labels=labels)
            save_path = "../{}/epoch_{}/images".format(experiment_name, epoch+1)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            plot_images(sampled_images)
            save_images(sampled_images, os.path.join(save_path, f"{epoch+1}.jpg"))
            save_images(ema_sampled_images, os.path.join(save_path, f"{epoch+1}_ema.jpg"))

            checkpoint_save(experiment_name, model, ema_model, optimizer, epoch+1)

        if scheduler is not None:
            scheduler.step()

        last_epoch = epoch
        training_steps += 1

    if training_steps == 0:
        # No training done
        logger.error("No training performed")
        update_telegram(bot, "No training performed")
        
    return

def main():
    args = parse_args()

    global logger
    logger = get_logger()
    set_level(args.severity)

    train_img_root = args.data
    
    experiment_name = EXP_BASE_NAME + " " + strftime("%d %b %H %M", localtime())

    logger.info("Using main device: " + args.device)
    logger.info("Training on: " + train_img_root)
    logger.info("Conditional model" if args.conditional else "Not conditional model")
    if args.conditional:
        logger.info("Using {} classes".format(args.classes))
    logger.info("Checkpoints will be saved in: " + experiment_name)
    logger.info("Using scheduler" if args.sched else "Not using scheduler")
    logger.info("Received the following hyperparameters:")
    logger.info("\tNumber of workers: " + str(args.nw))
    logger.info("\tBatch size: " + str(args.bs))
    logger.info("\tLearning rate: " + str(args.lr))
    logger.info("\tNumber of epochs: " + str(args.epochs))
    logger.info("\tOptimizer: " + str(OPTIMIZER) if not args.optim else strToOptim[args.optim])
    

    model = ANetwork(num_classes = args.classes, conditional=args.conditional, device=args.device)
    model.train()
    model.to(args.device)
    logger.info("Model created")

    preprocessing = model.get_preprocessing()

    dataset = ImageFolder(train_img_root,
                          transform=preprocessing)
    dataloader = DataLoader(dataset, 
                            batch_size=args.bs, 
                            shuffle=True)
    
    # Show what we have loaded
    logger.info("Training set:\t{} samples".format(len(dataset)))


    criterion = Loss(device = args.device)
    diffuser = Diffuser(img_size=model.FINAL_RES, device=args.device)
    ema = EMA(0.995)
    optimizer = OPTIMIZER(
        model.parameters(),
        lr = args.lr,
        # eps = 1e-6,
        weight_decay=WEIGHT_DECAY,
    )
    ema_model = model.clone()

    if args.sched:
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max = T_MAX, 
            eta_min = ETA_MIN, 
            verbose = True
        )
    else:
        scheduler = None


    start_epoch = 0

    if args.checkpoint is not None:
        logger.info("Loading checkpoint {}...".format(args.checkpoint))
        checkpoint_load(args.checkpoint, model, ema_model, optimizer, scheduler)

        # Gatherng the starting epoch from the weights
        try:
            epoch_str = re.findall(r'epoch_\d+', args.checkpoint)[0]
            start_epoch = int(re.findall(r'\d+', epoch_str)[0])
        except:
            start_epoch = 0
            # raise ValueError("Unable to find starting epoch...\n \
                #   Checkpoint file name must comprise the string 'epoch_N' in order to start from epoch N")


    if args.profile:
        global profiler
        profiler = Profile()


    train(model,
        optimizer,
        diffuser,
        ema,
        start_epoch,
        args.epochs,
        dataloader,
        criterion,
        args.device,
        experiment_name,
        scheduler=scheduler)

    logger.info("Training completed")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error("An error occurred during training")
        logger.error(traceback.format_exc())
        logger.error(e)
        update_telegram(bot, "An error occurred during training:\n" + str(e) + "\n" + traceback.format_exc())
        sys.exit(1)
