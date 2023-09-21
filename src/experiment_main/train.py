import torch
from torch import optim
from src.utils.loaders import model_loader
from src.utils.pytorchtools import EarlyStopping
from tqdm import tqdm
from src.utils.utils import create_missing_uci, create_missing_toy, create_missing_uci_drop_eddi, introduce_mising, \
    introduce_mising_mnar_based_on_mean_half_features, \
    introduce_mising_mnar_based_on_mean_all_features, introduce_mising_mnar_based_on_variance_all_features, \
    introduce_mising_mnar_based_on_variance_half_features
import numpy as np


def train(data_loader_train, missing_rate, obs_dim, hid_dim, K, M, latent_dim, data_type,
          training_parameters, experiment_type, vae_type, train_k, num_estimates, max_epochs=1000,
          device=torch.device('cpu'), alpha=1.0, stage='train', p_missingness=30, reg_type='ml_reg', beta=1.0,
          beta_annealing=False, alpha_annealing=True, not_miwae_type='changed'):
    model = model_loader('train', obs_dim, hid_dim, K, latent_dim, missing_rate, data_type,
                         training_parameters, max_epochs, train_k, num_estimates, experiment_type, reg_type, vae_type,
                         alpha=alpha, p_missingness=p_missingness)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if 'notMIWAE' not in vae_type:
        data_loader_train, _ = data_loader_train
    else:
        data_loader_train = data_loader_train
    for i in tqdm(range(max_epochs)):
        total_loss = 0
        for batch_idx, sample in enumerate(data_loader_train):
            data_sample, mask = sample
            data_sample = data_sample.to(device)
            if data_type == 'mnist':
                if 'with_drop' in vae_type:
                    mask_drop = []
                    for j in range(len(data_sample)):
                        temp_mask_drop = create_missing_uci_drop_eddi(data_sample[j].shape)
                        mask_drop.append(temp_mask_drop)
                    mask_drop = torch.stack(mask_drop, 0)
                else:
                    if 'reg' in vae_type:
                        mask_p = []
                        for j in range(len(data_sample)):
                            temp_mask = create_missing_uci(data_sample[j].shape, p_missingness)
                            mask_p.append(mask[j] * temp_mask)
                        mask_p = torch.stack(mask_p, 0)
                        mask_drop = torch.ones(mask_p.shape)
                    else:
                        mask_drop = torch.ones(data_sample.shape)

            else:
                if 'with_drop' in vae_type:
                    mask_drop = create_missing_uci_drop_eddi(data_sample.shape)
                else:
                    if 'reg' in vae_type:
                        temp_mask = create_missing_uci(data_sample.shape, p_missingness)
                        mask_p = temp_mask * mask
                        mask_drop = torch.ones(data_sample.shape)
                    else:
                        mask_drop = torch.ones(data_sample.shape)
            '''
            if p_missingness == 'half_features_mnar_mean':
                temp_mask = introduce_mising_mnar_based_on_mean_half_features(data_sample)
                mask_p = temp_mask * mask
            elif p_missingness == 'all_features_mnar_mean':
                temp_mask = introduce_mising_mnar_based_on_mean_all_features(data_sample)
                mask_p = temp_mask * mask
            elif p_missingness == 'half_features_mnar_var':
                temp_mask = introduce_mising_mnar_based_on_variance_half_features(data_sample)
                mask_p = temp_mask  * mask
            #elif p_missingness == 'all_features_mnar_var':
            else:
                temp_mask = introduce_mising_mnar_based_on_variance_all_features(data_sample)
                mask_p = temp_mask * mask
            '''
            data_sample = data_sample.to(device)
            mask = mask.to(device)
            # mask_p = mask_p.to(device)
            if 'flow' in vae_type:
                if 'reg_flow' in vae_type:
                    z_p, z_log_prob_p, x_mean_p, x_logvar_p, z_q, z_log_prob_q, x_mean_q, x_logvar_q = model.forward(
                        data_sample, mask, mask_p)
                    print_loss, train_loss = model.loss(data_sample, x_mean_q, x_logvar_q, z_q, z_log_prob_q, x_mean_p,
                                                        x_logvar_p, z_p, z_log_prob_p, mask, mask_p, alpha, stage=stage)
                else:
                    z, z_log_prob, x_mean, x_logvar = model.forward(
                        data_sample, mask)
                    print_loss, train_loss = model.loss(data_sample, x_mean, x_logvar, z, z_log_prob, mask)
            elif 'reg_vae' in vae_type or 'reg_EDDI' in vae_type or 'reg_mnist' in vae_type or 'reg_notMIWAE' in vae_type:
                mean_p, logvar_p, x_mean_p, x_logvar_p, mean_q, logvar_q, x_mean_q, x_logvar_q = model.forward(
                    data_sample, mask, mask_p, stage=stage)
                print_loss, train_loss = model.loss(
                    data_sample, x_mean_p, x_logvar_p, mean_p, logvar_p,
                    x_mean_q, x_logvar_q,
                    mean_q, logvar_q, mask, mask_p, i + 1, beta_annealing=beta_annealing, beta=beta, alpha=alpha,
                    alpha_annealing=alpha_annealing, stage=stage)
            elif 'vanilla_vae' in vae_type or 'vanilla_EDDI' in vae_type or 'vanilla_notMIWAE' in vae_type:
                mean_q, logvar_q, x_mean_q, x_logvar_q = model.forward(
                    data_sample, mask * mask_drop)
                print_loss, train_loss = model.loss(
                    data_sample,
                    x_mean_q, x_logvar_q,
                    mean_q, logvar_q, i + 1, mask * mask_drop, beta_annealing=beta_annealing, beta=beta, stage=stage)
            elif 'reg_MIWAE' in vae_type:
                mean_p, scale_p, x_mean_p, x_scale_p, deg_free_p, mean_q, scale_q, x_mean_q, x_scale_q, deg_free_q = model.forward(
                    data_sample, mask, mask_p)
                print_loss, train_loss = model.loss(
                    data_sample, x_mean_p, x_scale_p, deg_free_p, mean_p, scale_p,
                    x_mean_q, x_scale_q, deg_free_q,
                    mean_q, scale_q, mask, mask_p, i + 1, beta_annealing=beta_annealing, beta=beta, alpha=alpha)
            else:
                # or 'reg_notMIWAE' in vae_type
                mean, scale, x_mean, x_scale, deg_free = model.forward(data_sample, mask)
                print_loss, train_loss = model.loss(
                    data_sample, x_mean, x_scale, deg_free, mean, scale, mask, i + 1)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            total_loss += train_loss.item()
        tqdm.write('Epoch: [{}/{}], Total Loss: {}'.format(i, max_epochs, total_loss))

    if 'vanilla' in vae_type:
        torch.save(model.state_dict(),
                   'experiments/' + experiment_type + '/' + data_type + '/checkpoints/' + ''.join(
                       [i for i in '_'.join(vae_type.split('_')[:2]) if
                        not i.isdigit()]) + '/' + 'checkpoint_' + vae_type + '_' + str(
                       missing_rate) + '_missing_rate_test.pt')
    else:
        torch.save(model.state_dict(), 'experiments/' + experiment_type + '/' + data_type + '/checkpoints/' + ''.join(
            [i for i in '_'.join(vae_type.split('_')[:2]) if
             not i.isdigit()]) + '/' + 'checkpoint_' + vae_type + '_' + str(
            alpha) + '_' + str(p_missingness) + '_' + reg_type + '_' + str(
            missing_rate) + '_missing_rate_full_reg_test.pt')

    print('Training is over!')
