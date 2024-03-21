"""Script for training KWT model"""
from argparse import ArgumentParser
from torch import nn, optim
from torch.utils.data import DataLoader
import torch
from torch import nn
import wandb
from utils.trainer import train, evaluate
from utils.dataset import get_loader
from src.data.dataset import SpectrogramDataset
#from utils.spectrogram_dataset import SpectrogramDataset
from utils.config_parser import parse_config
from KWT import KWT
from utils.misc import seed_everything, count_params, get_model, calc_step, log
#from torch.nn.parallel import DataParallel #probably too much overhead, is too slow with even 2 GPUs
import matplotlib.pyplot as plt
import os

""""
Run script with configuration file as argument

command: srun --time=4:00:00 --gres=gpu:1 singularity exec --nv <your_singularity_container_here> python train.py --conf KWT_configs/KWT_config.cfg

"""

def training_pipeline(config):
    """
    Initiates and executes all the steps involved with KWT model training
    :param config: KWT configuration
    """
    #####################################
    # Training initialzation
    #####################################
 # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

       # Initialize wandb run
    wandb.init(project=config["exp"]["proj_name"], name=config["exp"]["exp_name"], config=config["hparams"])

    # Initialize KWT
    model = KWT(**config['hparams']['KWT'])
    #model = DataParallel(model) # Wrapping model in DataParallel class
    model.to(device); # Sending device to GPU

    if args.ckpt:
        checkpoint = torch.load(args.ckpt)
        #print(checkpoint.keys())  # Print out the keys
        # Look for the correct key for the model state di
        model.load_state_dict(checkpoint['state_dict'])

    # Make dataset
    train_set = SpectrogramDataset(config['tr_manifest_path'], config['labels_map'], config['audio_config'])
    val_set = SpectrogramDataset(config['val_manifest_path'], config['labels_map'], config['audio_config'])
    test_set = SpectrogramDataset(config['eval_manifest_path'],config['labels_map'],config['audio_config'])

    train_set.files = train_set.files[:1000]
    train_set.labels = train_set.labels[:1000]
    train_set.len = len(train_set.files)
    val_set.files = val_set.files[:1000]
    val_set.labels = val_set.labels[:1000]
    val_set.len = len(val_set.files)

    # Make dataloaders
    train_loader = DataLoader(train_set, batch_size=config['hparams']['batch_size'])
    val_loader = DataLoader(val_set, batch_size=config['hparams']['batch_size'])
    test_loader = DataLoader(test_set, batch_size=config['hparams']['batch_size'])

    #classes = ('COVID-19', 'healthy', 'symptomatic')
    classes = {"COVID-19": 0, "healthy": 1, "symptomatic": 2}
    
    # Cross Entropy loss
    criterion = nn.CrossEntropyLoss() # multi class 
    #criterion = nn.BCEWithLogitsLoss() #multi label classification (FDS50K)

    # optimizer
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=config["hparams"]["optimizer"]["lr"],
                           betas=config["hparams"]["optimizer"]["betas"],
                           eps=config["hparams"]["optimizer"]["eps"],
                           weight_decay=config["hparams"]["optimizer"]["weight_decay"])
    
    #optimizer = get_optimizer(model, config["hparams"]["optimizer"])
    
    # # Make a scheduler
    schedulers = {'scheduler': None,
                  'warmup': None}
    

    #####################################
    # Training Run
    #####################################

      # Training loop
    print("Initiating training.")
   # for epoch in range(config["hparams"]["n_epochs"]):
    #    print("Current epoch:", epoch)
     #   train_loss = train(model, optimizer, criterion, train_loader, val_loader, schedulers, config)
       # val_loss, val_acc = evaluate(model, criterion, val_loader, config)

        # Log metrics to wandb
        #wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})
     #   wandb.log({"train_loss:": train_loss})
    # Save model
    #torch.save(model.state_dict(), "final_model.pth")

    # Log final model to wandb
    #wandb.save("final_model.pth")
    #wandb.finish()  # Finish wandb run after training

    train(model, optimizer, criterion, train_loader, val_loader, schedulers, config) 
    
    #####################################
    # Final Test
    #####################################

    final_step = calc_step(config["hparams"]["n_epochs"] + 1, len(train_loader), len(train_loader) - 1)

    # evaluating the final state (last.pth)
    test_acc, test_loss = evaluate(model, criterion, test_loader, config["hparams"])
    log_dict = {
        "test_loss_last": test_loss,
        "test_acc_last": test_acc
    }
    log(log_dict, final_step, config)

    # evaluating the best validation state (best.pth)
    ckpt = torch.load(os.path.join(config["exp"]["save_dir"], "best.pth"))
    model.load_state_dict(ckpt["model_state_dict"])
    print("Best ckpt loaded.")

    test_acc, test_loss = evaluate(model, criterion, test_loader, config["hparams"])
    log_dict = {
        "test_loss_best": test_loss,
        "test_acc_best": test_acc
    }
    log(log_dict, final_step, config)


def main(args):
    """
    Calls training pipeline and sets up wandb logging if used
    :param args: input arguments
    
    """

    config = parse_config(args.conf)

    

    # below code is for logging everything in wandb
    if args.id:
        config["exp"]["exp_name"] = config["exp"]["exp_name"] + args.id

    if config["exp"]["wandb"]:
        if config["exp"]["wandb_api_key"] is not None:
            with open(config["exp"]["wandb_api_key"], "r") as f:
                os.environ["WANDB_API_KEY"] = f.read()
        
        elif os.environ.get("WANDB_API_KEY", False):
            print("Found API key from env variable.")
        
        else:
            wandb.login()

        with wandb.init(project=config["exp"]["proj_name"], name=config["exp"]["exp_name"], config=config["hparams"],entity=config["exp"]["entity"]):
            training_pipeline(config)
    else:

        training_pipeline(config)

if __name__ == "__main__":
    parser = ArgumentParser("Driver code.")
    parser.add_argument("--conf", type=str, required=True, help="Path to config.yaml file.")
    parser.add_argument("--ckpt", type=str, required=False, help="Path to checkpoint file.", default=None)
    parser.add_argument("--id", type=str, required=False, help="Obtional experiment identifier.", default=None)
    args = parser.parse_args()

    main(args)

    
