from mindspore.train.serialization import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore import Tensor
import torch
import os
import argparse

from pipeline.scoring.aesthetic.inference import AestheticScorer

def show_params(params, value):
    if value:
        for k, v in params.items():
            print('{}: {}'.format(k, v))
    else:
        for k in params.keys():
            print(k)

def show_params_from_model(model, value = True):
    """ Show contents of an initialized model, support mindspore model only """
    params = model.parameters_dict()
    show_params(params, value)

def show_params_from_path(file_path, framework, value = True):
    """ Show contents of a checkpoint file. """
    if not os.path.isfile(file_path):
        raise FileExistsError("The file `{}` does not exist! ".format(file_path))

    if framework == "torch":
        params = torch.load(file_path, map_location=torch.device('cpu'))
    elif framework == "mindspore":
        params = load_checkpoint(file_path)
    else:
        raise ValueError("Attribute `params` must be in [`torch`, `mindspore`]! ")
    show_params(params, value)


def aesthetic_pth_to_ckpt(pth_path='pretrained_models/aesthetic.pth',
                            save_path='pretrained_models/aesthetic.ckpt'):
    """
    Transform a torch checkpoint file into mindspore checkpoint.
    Modify the param's name first, then change tensor type.
    """

    if not os.path.isfile(pth_path):
        raise FileExistsError("The file `{}` does not exist! ".format(pth_path))
    if ".ckpt" not in save_path:
        raise ValueError("Attribute `save_path` should be a checkpoint file with the end of `.ckpt`!")

    params = torch.load(pth_path, map_location=torch.device('cpu'))

    torch_params = list(params.items())
    num_params = len(torch_params)
    params_list = []
    for i in range(num_params):
        key, value = torch_params[i]
        if 'layer' in key:
            key = 'mlp.' + key
        params_list.append({"name": key, "data": Tensor(value.numpy())})
    save_checkpoint(params_list, save_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help = 'The model to convert from pth to ckpt.')
    parser.add_argument('--pth_path', type=str, required=True, help='pth checkpoint location')
    parser.add_argument('--save_path', type=str, required=True, help='the location to save ckpt checkpoint')

    parser.add_argument('--show_pth', action='store_true', help = 'Show torch parameters.')
    parser.add_argument('--show_ckpt', action='store_true', help = 'Show mindspore parameters.')
    parser.add_argument('--convert', action='store_true', help = 'Convert pth to ckpt')
    parser.add_argument('--value', action='store_true', help='whether to print the values of the parameters.')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    assert args.model in ['aesthetic'], f"The model must be 'aesthetic', got {args.model}."
    if args.show_pth:
        print("=========Showing pth parameters==========")
        show_params_from_path(args.pth_path, 'torch', args.value)

    if args.show_ckpt:
        print("=========Showing ckpt parameters==========")
        if args.model == 'aesthetic':
            model = AestheticScorer()
            show_params_from_model(model, value = False)

    if args.convert:
        print("=========Converting pth to ckpt==========")
        if args.model == 'aesthetic':
            aesthetic_pth_to_ckpt(args.pth_path, args.save_path)
        print("Done!")

if __name__ == '__main__':
    main()