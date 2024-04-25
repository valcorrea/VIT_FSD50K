'''
Hyperparameter Optimization
'''
import pprint

from utils.config_parser import parse_config


def main(args):
    config = merge_config(args.model_config, args.sweep_config)
    pprint.pprint(config)


def merge_config(model_config:str, sweep_config:str) -> dict:
    '''
    Reads config files and merges the model and sweep configurations into one that contains the essentials.

    Args:
        model_config (str): Path to cfg. Contains the model architecture, audio config and experiment configs
        sweep_config (str): Path to cfg. Contains the hyperparameters to sweep through and their possible values. 
    
    Returns:
        model_config (dict): Contains the necessary combination of configs from model/sweep_config.
    '''
    model_config = parse_config(model_config)
    sweep_config = parse_config(sweep_config)
    model_config.update(sweep_config)
    return model_config


if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--model_config', type=str, help='Path to the model configuration file.')
    ap.add_argument('--sweep_config', type=str, help='Path to the hyperparameter sweep configuration file.')
    args = ap.parse_args()

    main(args)