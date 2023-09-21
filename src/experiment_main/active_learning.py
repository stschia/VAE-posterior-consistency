import sys
from numpy import loadtxt
sys.path.append("/Users/timursudak/Documents/GitHub/Timur-Study/")
import torch
from src.utils.loaders import data_loader, ConcatDataset
from src.utils.utils import setup_parser
import argparse
from src.experiment_main.train import train
from src.experiment_main.evaluate import active_learning_func
from sklearn.model_selection import train_test_split
from src.utils.AIS import linear_schedule, eval_ais, sigmoidial_schedule
import json
from torch.utils.data import DataLoader
import os
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Device: ", device)

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    with open("Data/imputation_args.json") as data_files:
        for data_file in data_files:
            for missing in [30]:
                for alpha in [1.0]:
                    data = json.loads(data_file)
                    parser = setup_parser(data, 'impute_eval')
                    args = parser.parse_args()

                    """
                    Train and evaluate model for imputation
                    """
                    data = torch.load(os.path.join(args.data_path, args.data_type, 'data.pt'))
                    index = [i for i in args.vae_type if i.isdigit()][0]
                    train_indices = loadtxt(
                        os.path.join(args.data_path, args.data_type, 'train_index' + index + '.csv'),
                        delimiter=',')
                    test_indices = loadtxt(
                        os.path.join(args.data_path, args.data_type, 'test_index' + index + '.csv'),
                        delimiter=',')
                    mask = torch.load(
                        os.path.join(args.data_path, args.data_type,
                                     'mask_' + str(args.missing_rate) + '_missing' + index + '.pt'))

                    obs_dim = data.shape[1]
                    ### data preprocess
                    max_Data = 1  #
                    min_Data = 0  #
                    Data_std = (data - data.min(axis=0).values) / (
                            data.max(axis=0).values - data.min(axis=0).values)
                    data_norm = Data_std * (max_Data - min_Data) + min_Data
                    data_train = data_norm[train_indices]
                    data_test = data_norm[test_indices]
                    mask_train = mask[train_indices]
                    mask_test = mask[test_indices]
                    data_loader_train = DataLoader(
                        ConcatDataset(data_train.to(device), mask_train.to(device)),
                        batch_size=args.batch_size,
                        shuffle=True,
                        drop_last=False, num_workers=0)

                    active_learning_func(data_loader_train, data_test, mask_test, args.missing_rate, obs_dim,
                                         args.hid_dim,
                                         args.K,
                                         args.M, args.latent_dim,
                                         args.data_type,
                                         {'batch_size': args.batch_size, 'patience': args.patience},
                                         args.experiment_type,
                                         args.vae_type, args.epoch, args.valid_k,
                                         10, device=device, alpha=alpha, p_missingness=missing,
                                         reg_type=args.reg_type,
                                         Repeat=1)
