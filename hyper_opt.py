'''
Hyperparameter Optimization
'''
import pprint

from utils.config_parser import parse_config


def main(args):
    model_config = parse_config(args.model_config)
    sweep_config = parse_config(args.sweep_config)
    pprint.pprint(sweep_config)


if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--model_config', type=str, help='Path to the model configuration file.')
    ap.add_argument('--sweep_config', type=str, help='Path to the hyperparameter sweep configuration file.')
    args = ap.parse_args()

    main(args)