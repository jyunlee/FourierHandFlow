import os
import yaml
import datasets
import trainers
import sys
import modules
from lib.leap import LEAP

def load_config(path):
    """ Loads config file.

    Args:
        path (str): path to config file
    """
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    dataset_type = cfg['data']['dataset']

    if dataset_type == 'interhand_seq':
        # add additional attributes
        bm_path = cfg['data']['bm_path']
        model_type, num_joints = LEAP.get_num_joints(bm_path)

        cfg['model']['num_joints'] = num_joints
        cfg['model']['model_type'] = model_type
        cfg['model']['parent_mapping'] = LEAP.get_parent_mapping(model_type)
        
        if cfg['method'] == '4d_model':
            for key in ['inv_lbs_model_config', 'fwd_lbs_model_config']:
                for attr in ['num_joints', 'model_type', 'parent_mapping']:
                    cfg['model'][key][attr] = cfg['model'][attr]

    return cfg


def get_model(cfg):
    """ Returns the model instance.

    Args:
        cfg (dict): config dictionary

    Returns:
        model (torch.nn.Module)
    """
    method = cfg['method']

    assert method in ['4d_model'], \
        'Not supported method type'

    model = {
        '4d_model': modules.ShapeFlowNet
    }[method].from_cfg(cfg['model'])

    return model.to(device=cfg['device'])


def get_trainer(model, optimizer, cfg):
    """ Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary

    Returns:
        trainer instance (BaseTrainer)
    """
    method = cfg['method']

    assert method in ['4d_model'], \
        'Not supported method type'

    trainer = {
        '4d_model': trainers.ShapeFlowTrainer,
    }[method](model, optimizer, cfg)

    return trainer


def get_dataset(mode, cfg):
    """ Returns the dataset.

    Args:
        mode (str): `train`, `val`, or 'test' dataset mode
        cfg (dict): config dictionary

    Returns:
        dataset (torch.data.utils.data.Dataset)
    """
    method = cfg['method']
    dataset_type = cfg['data']['dataset']

    assert method in ['4d_model']
    assert dataset_type in ['interhand_seq']
    assert mode in ['train', 'val', 'test']

    # Create dataset
    if dataset_type == 'interhand_seq':
        dataset = {
            '4d_model': datasets.InterHandSeqDataset,
        }[method]

    else:
        raise NotImplementedError(f'Not supported dataset type ({dataset_type})')

    dataset = dataset(cfg['data'], mode)

    return dataset
