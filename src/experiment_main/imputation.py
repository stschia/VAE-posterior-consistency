import sys

sys.path.append("/Users/timursudak/Documents/GitHub/Timur-Study/")
import torch
from src.utils.loaders import data_loader, data_loader_mnar, data_loader_mnist
from src.utils.utils import setup_parser
# from src.utils.stuff import mut_inf_v2
import argparse
from src.experiment_main.train import train
from src.experiment_main.evaluate import eval_vae, eval_miwae
from src.utils.AIS import linear_schedule, eval_ais, sigmoidial_schedule
import json
from torch.utils.data import DataLoader
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
                    data_loader_train, data_loader_test, obs_dim = data_loader(args.data_path, args.vae_type,
                                                                               args.missing_rate,
                                                                               args.batch_size,
                                                                               args.data_type,
                                                                               device=device)
                    train(data_loader_train, args.missing_rate, obs_dim, args.hid_dim, args.K,
                          args.M,
                          args.latent_dim, args.data_type,
                          {'batch_size': args.batch_size, 'patience': args.patience}, args.experiment_type,
                          args.vae_type,
                          args.train_k, 10,
                          args.epoch, device=device, alpha=alpha, p_missingness=missing, reg_type=args.reg_type)
                    if 'MIWAE' in args.vae_type:
                        eval_miwae([data_loader_train, data_loader_test], args.missing_rate, obs_dim,
                                   args.hid_dim,
                                   args.K,
                                   args.M, args.latent_dim,
                                   args.data_type,
                                   {'batch_size': args.batch_size, 'patience': args.patience}, args.experiment_type,
                                   args.vae_type, args.epoch,
                                   args.valid_k, 10, device=device, alpha=alpha, p_missingness=missing,
                                   reg_type=args.reg_type)
                    else:
                        eval_vae([data_loader_train, data_loader_test], args.missing_rate, obs_dim,
                                 args.hid_dim,
                                 args.K,
                                 args.M, args.latent_dim,
                                 args.data_type,
                                 {'batch_size': args.batch_size, 'patience': args.patience}, args.experiment_type,
                                 args.vae_type, args.epoch,
                                 args.valid_k, 10, device=device, alpha=alpha, p_missingness=missing,
                                 reg_type=args.reg_type)
