import os
from collections import OrderedDict

import torch


# os.environ['TORCH_HOME'] = '/userhome/cjh/models/'

def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + eps)
    return x / (norm_factor + eps)


def get_state_dict(opt, net_type: str = 'alex', version: str = '0.1'):
    # build url
    url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' \
          + f'master/models/weights/v{version}/{net_type}.pth'

    # download
    # old_state_dict = torch.hub.load_state_dict_from_url(
    #     url, progress=True,
    #     map_location=None if torch.cuda.is_available() else torch.device('cpu')
    # )
    path = os.path.join(opt.checkpoints_dir, opt.name)
    old_state_dict = torch.load(os.path.join(path, "alex.pth"))
    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val

    return new_state_dict
