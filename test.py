import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as standard_transforms
from crowd_datasets.SHHA.SHHA import SHHA
from models import build_model
from engine import evaluate_crowd_no_overlap
import util.misc as utils
from pathlib import Path
from engine import *
import os
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for testing P2PNet', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for testing')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers for data loader')
    parser.add_argument('--data_root', default='./DATASET/part_A', help='Path to the dataset')
    parser.add_argument('--dataset_file', default='SHHA', help='Dataset file')
    parser.add_argument('--resume', default='./ckpt/best_mae.pth', help='Path to the model checkpoint')
    parser.add_argument('--gpu_id', default=0, type=int, help='The GPU used for testing')
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                    help="Name of the convolutional backbone to use")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="L1 point coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)

    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")
    return parser

def main(args):
    # Set GPU
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
    ])
    # Build the dataset
    test_set = SHHA(args.data_root, train=False, transform=transform, test = True)
    print(len(test_set))
    # Create data loader
    sampler_val = torch.utils.data.SequentialSampler(test_set)
    data_loader_val = DataLoader(test_set, batch_size=args.batch_size, sampler=sampler_val,
                                 collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    # Build the model
    model = build_model(args, training=False)
    model.to(device)

    # Load the checkpoint
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    else:
        raise ValueError("No checkpoint path provided for testing.")

    # Evaluate the model
    print("Start testing")
    result = evaluate_crowd_no_overlap(model, data_loader_val, device)

    print('=======================================Test Results=======================================')
    print(f"MAE: {result[0]}, MSE: {result[1]}")
    print('==========================================================================================')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet testing script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
