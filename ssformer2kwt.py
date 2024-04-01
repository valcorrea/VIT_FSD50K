import torch
import os

'''
Script to change an SSformer saved model to match KWT.
'''

def load_model(model_path:str) -> dict:
    return torch.load(model_path, map_location=torch.device('cpu'))

def change_keys(model:dict, lightning:bool) -> dict:
    '''
    Change a Self-Supervised ViT Model to KWT
    model:
    In practice: 
    Renames state_dict key to model_state_dict
    Renames all model.parameter keys to parameter
    '''

    # Remove all keys that are not for KWT
    for item in [key for key in model['state_dict'].keys() if 'model.encoder' not in key]:
        model['state_dict'].pop(item)

    # Remove 'model.encoder' from the start of all KWT keys
    if lightning:
        for item in model['state_dict'].copy().keys():
            model['state_dict'][item.replace('model.encoder.', 'model.')] = model['state_dict'].pop(item)
    else:
        for item in model['state_dict'].copy().keys():
            model['state_dict'][item.replace('model.encoder.', '')] = model['state_dict'].pop(item)
    return model

def change_keys_KWTL2KWT(model:dict) -> dict:
    '''
    Change a KWT-Lightning Model to KWT-classic
    In practice: 
    Renames state_dict key to model_state_dict
    Renames all model.parameter keys to parameter
    '''
    model['model_state_dict'] = model.pop('state_dict')
    for item in model['model_state_dict'].copy().keys():
        model['model_state_dict'][item.replace('model.', '')] = model['model_state_dict'].pop(item)
    return model

def save_model(model:dict, out_dir:str, filename:str):
    print(f'Saving model to: {os.path.join(out_dir, filename)}')
    torch.save(model, os.path.join(out_dir, filename))

if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--model_path', type=str, required=True, help='Path to the ssformer model checkpoint')
    ap.add_argument('--out_dir', type=str, required=False, help='Directory to save the new checkpoint.')
    ap.add_argument('--filename', type=str, required=False, help='Name of the new checkpoint.')
    ap.add_argument('--task', type=str, choices=['ssformer2kwt', 'ssformer2kwtlight', 'kwtlight2kwt'])
    args = ap.parse_args()

    model = load_model(args.model_path)
    if args.task == 'ssformer2kwt':
        model = change_keys(model, lightning=False)
    elif args.task == 'ssformer2kwtlight':
        model = change_keys(model, lightning=True)
    else:
        model = change_keys_KWTL2KWT(model)

    out_dir = args.out_dir if args.out_dir else os.path.dirname(args.model_path)
    filename = args.filename if args.filename else f"{os.path.basename(args.model_path).split('.')[0]}_KWT.ckpt"
    save_model(model, out_dir, filename)