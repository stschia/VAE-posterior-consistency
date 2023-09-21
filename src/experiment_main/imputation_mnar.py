import sys
from numpy import loadtxt

sys.path.append("/Users/timursudak/Documents/GitHub/Timur-Study/")
import torch
from src.utils.loaders import data_loader, data_loader_mnar
from src.utils.utils import setup_parser
from src.experiment_main.train import train
from src.experiment_main.evaluate import eval_vae, eval_vae_mnar
from src.utils.AIS import linear_schedule, eval_ais, sigmoidial_schedule
import json
from torch.utils.data import DataLoader
import os
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Device: ", device)

torch.autograd.set_detect_anomaly(True)
# [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2]
# [0.1, 0.5, 1.0, 1.3, 1.5, 2]
# [5, 10, 15, 20, 25, 30, 35, 40, 50]

# 0.1, 0.5, 1.0, 1.5

if __name__ == "__main__":
    with open("Data/imputation_args_mnar.json") as data_files:
        for data_file in data_files:
            for missing in [50]:
                for alpha in [1.0]:
                    data = json.loads(data_file)
                    parser = setup_parser(data, 'impute_eval')
                    args = parser.parse_args()
                    """
                        Train and evaluate model for imputation
                    """
                    data_transform = 'minmax'  # or 'stand' for transformation of the data as in the implementation of authors
                    not_miwae_type = 'changed'  # or 'author' for the notMIWAE models as in the implementation of authors

                    data_loader_train, obs_dim = data_loader_mnar(args.data_path, args.vae_type,
                                                                args.missing_rate,
                                                                args.batch_size,
                                                                args.data_type,
                                                                device=device, data_transform=data_transform)

                    # TODO should train really be a function of the model?
                    data = torch.load(os.path.join(args.data_path, args.data_type, 'data.pt'))
                    data = data[:, :-1]
                    index = [i for i in args.vae_type if i.isdigit()][0]

                    temp_perm = torch.load('Data/' + args.data_type + '/rand_perm' + index + '.pt').numpy()
                    data = data[temp_perm, :]
                    mask = torch.load(os.path.join(args.data_path, args.data_type, 'mnar_mask_missing' + index + '.pt'))
                    mask = mask[:, :-1]
                    if data_transform == 'minmax':
                        ### data preprocess for my version
                        max_Data = 1  #
                        min_Data = 0  #
                        Data_std = (data - data.min(axis=0).values) / (
                                data.max(axis=0).values - data.min(axis=0).values)
                        data = Data_std * (max_Data - min_Data) + min_Data
                    else:
                        ### data preprocess for notMIWAE version
                        data = data - data.mean(0)
                        data = data / data.std(0)

                    train(data_loader_train, args.missing_rate, obs_dim, args.hid_dim,
                          args.K,
                          args.M,
                          args.latent_dim, args.data_type,
                          {'batch_size': args.batch_size, 'patience': args.patience}, args.experiment_type,
                          args.vae_type,
                          args.train_k, 10,
                          args.epoch, device=device, alpha=alpha, p_missingness=missing, reg_type=args.reg_type, not_miwae_type=not_miwae_type)

                    eval_vae_mnar(data, mask, args.missing_rate, obs_dim,
                                  args.hid_dim,
                                  args.K,
                                  args.M, args.latent_dim,
                                  args.data_type,
                                  {'batch_size': args.batch_size, 'patience': args.patience}, args.experiment_type,
                                  args.vae_type, args.epoch,
                                  args.valid_k, 10, device=device, alpha=alpha, p_missingness=missing,
                                  reg_type=args.reg_type, not_miwae_type=not_miwae_type)
