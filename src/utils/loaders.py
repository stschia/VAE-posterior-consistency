import torch
from src.models.VAE import Reg_VAE, Flow, Reg_EDDI, MIWAE, Reg_MIWAE, vanilla_VAE, vanilla_EDDI, vanilla_VAE_mask, \
    Reg_VAE_mask, notMIWAE, \
    REG_notMIWAE, notMIWAE_myversion, REG_notMIWAE_new_version, REG_notMIWAE_v2, REG_VAEFlow, VAEFlow, \
    vanilla_EDDI_mnist, Reg_EDDI_mnist
from torch.utils.data import DataLoader
from numpy import loadtxt
from sklearn.model_selection import train_test_split
import os
from torchvision import datasets, transforms


def model_loader(stage, obs_dim, hid_dim, K, latent_dim, missing_rate, data_type, training_parameters,
                 max_epochs,
                 num_samples, num_estimates, experiment_type, reg_type,
                 vae_type='vae', alpha=1.0, p_missingness=30, beta=0.5, beta_annealing=True,
                 alpha_annealing=True,
                 not_miwae_type='changed'):
    if 'flow' in vae_type:
        if 'reg_flow' in vae_type:
            if stage == 'train':
                print("Initializing fresh model")
                model = REG_VAEFlow(obs_dim, hid_dim, K, latent_dim, training_parameters,
                                    num_samples=num_samples, num_estimates=num_estimates)
            else:

                print("Loading saved model")
                model = REG_VAEFlow(obs_dim, hid_dim, K, latent_dim, training_parameters,
                                    num_samples=num_samples, num_estimates=num_estimates)
                state_dict = torch.load(
                    'experiments/' + experiment_type + '/' + data_type + '/checkpoints/' + ''.join(
                        [i for i in vae_type if not i.isdigit()]) + '/' + 'checkpoint_' + vae_type + '_' + str(
                        alpha) + '_' + str(
                        p_missingness) + '_' + reg_type + '_' + str(
                        missing_rate) + '_missing_rate_full_reg_test.pt')
                model.load_state_dict(state_dict)
        else:
            if stage == 'train':
                print("Initializing fresh model")
                model = VAEFlow(obs_dim, hid_dim, K, latent_dim, training_parameters,
                                num_samples=num_samples, num_estimates=num_estimates)
            else:
                print("Loading saved model")
                model = VAEFlow(obs_dim, hid_dim, K, latent_dim, training_parameters,
                                num_samples=num_samples, num_estimates=num_estimates)
                state_dict = torch.load(
                    'experiments/' + experiment_type + '/' + data_type + '/checkpoints/' + ''.join(
                        [i for i in vae_type if not i.isdigit()]) + '/' + 'checkpoint_' + vae_type + '_' + str(
                        missing_rate) + '_missing_rate_test.pt')
                model.load_state_dict(state_dict)
    elif 'reg_vae' in vae_type:
        if 'mask_augm' in vae_type:
            if stage == 'train':
                print("Initializing fresh model")
                # print(hid_dim)
                # print(latent_dim)
                model = Reg_VAE_mask(obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type, reg_type,
                                     num_samples, num_estimates)
            else:
                print("Loading saved model")
                model = Reg_VAE_mask(obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type, reg_type,
                                     num_samples, num_estimates)
                state_dict = torch.load(
                    'experiments/' + experiment_type + '/' + data_type + '/checkpoints/' + ''.join(
                        [i for i in 'reg_vae' if
                         not i.isdigit()]) + '/' + 'checkpoint_' + vae_type + '_' + str(
                        alpha) + '_' +
                    str(p_missingness) + '_' + reg_type + '_' + str(
                        missing_rate) + '_missing_rate_full_reg_test.pt',
                    map_location=torch.device('cpu'))
                model.load_state_dict(state_dict)
        else:
            if stage == 'train':
                print("Initializing fresh model")
                model = Reg_VAE(obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type, reg_type,
                                num_samples, num_estimates)
            else:
                print("Loading saved model")
                model = Reg_VAE(obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type, reg_type,
                                num_samples, num_estimates)
                state_dict = torch.load(
                    'experiments/' + experiment_type + '/' + data_type + '/checkpoints/' + ''.join(
                        [i for i in vae_type if not i.isdigit()]) + '/' + 'checkpoint_' + vae_type + '_' + str(
                        alpha) + '_' +
                    str(p_missingness) + '_' + reg_type + '_' + str(
                        missing_rate) + '_missing_rate_full_reg_test.pt',
                    map_location=torch.device('cpu'))
                model.load_state_dict(state_dict)
    elif 'reg_notMIWAE' in vae_type:
        if stage == 'train':
            print("Initializing fresh model")
            model = REG_notMIWAE_v2(obs_dim, hid_dim, K, latent_dim, training_parameters,
                                    num_samples, num_estimates)
        else:
            print("Loading saved model")
            model = REG_notMIWAE_v2(obs_dim, hid_dim, K, latent_dim, training_parameters,
                                    num_samples, num_estimates)
            state_dict = torch.load(
                'experiments/' + experiment_type + '/' + data_type + '/checkpoints/' + ''.join(
                    [i for i in vae_type if not i.isdigit()]) + '/' + 'checkpoint_' + vae_type + '_' + str(
                    alpha) + '_' + str(p_missingness) + '_' + reg_type + '_' + str(
                    missing_rate) + '_missing_rate_full_reg_test.pt')
            model.load_state_dict(state_dict)
    elif 'reg_EDDI' in vae_type:
        if 'mnist' == data_type:
            if stage == 'train':
                print("Initializing fresh model")
                model = Reg_EDDI_mnist(obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type, reg_type,
                                       num_samples, num_estimates)
            else:
                print("Loading saved model")
                model = Reg_EDDI_mnist(obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type, reg_type,
                                       num_samples, num_estimates)
                state_dict = torch.load(
                    'experiments/' + experiment_type + '/' + data_type + '/checkpoints/' + ''.join(
                        [i for i in vae_type if not i.isdigit()]) + '/' + 'checkpoint_' + vae_type + '_' + str(
                        alpha) + '_' + str(p_missingness) + '_' + reg_type + '_' + str(
                        missing_rate) + '_missing_rate_full_reg_test.pt')
                model.load_state_dict(state_dict)
        else:
            if stage == 'train':
                print("Initializing fresh model")
                model = Reg_EDDI(obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type, reg_type,
                                 num_samples, num_estimates)
            else:
                print("Loading saved model")
                model = Reg_EDDI(obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type, reg_type,
                                 num_samples, num_estimates)
                state_dict = torch.load(
                    'experiments/' + experiment_type + '/' + data_type + '/checkpoints/' + ''.join(
                        [i for i in vae_type if not i.isdigit()]) + '/' + 'checkpoint_' + vae_type + '_' + str(
                        alpha) + '_' + str(p_missingness) + '_' + reg_type + '_' + str(
                        missing_rate) + '_missing_rate_full_reg_test.pt')
                model.load_state_dict(state_dict)
    elif 'reg_MIWAE' in vae_type:
        if stage == 'train':
            print("Initializing fresh model")
            model = Reg_MIWAE(obs_dim, hid_dim, K, latent_dim, training_parameters, num_samples, num_estimates)
        else:
            print("Loading saved model")
            model = Reg_MIWAE(obs_dim, hid_dim, K, latent_dim, training_parameters, num_samples, num_estimates)
            state_dict = torch.load(
                'experiments/' + experiment_type + '/' + data_type + '/checkpoints/' + ''.join(
                    [i for i in vae_type if not i.isdigit()]) + '/' + 'checkpoint_' + vae_type + '_' + str(
                    alpha) + '_' + str(
                    p_missingness) + '_' + reg_type + '_' + str(
                    missing_rate) + '_missing_rate_full_reg_test.pt')
            model.load_state_dict(state_dict)
    elif 'vanilla_vae' in vae_type:
        if 'mask_augm' in vae_type:
            if stage == 'train':
                print("Initializing fresh model")
                model = vanilla_VAE_mask(obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type,
                                         num_samples,
                                         num_estimates)
            else:
                print("Loading saved model")
                model = vanilla_VAE_mask(obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type,
                                         num_samples,
                                         num_estimates)
                state_dict = torch.load(
                    'experiments/' + experiment_type + '/' + data_type + '/checkpoints/' + ''.join(
                        [i for i in 'vanilla_vae' if
                         not i.isdigit()]) + '/' + 'checkpoint_' + vae_type + '_' + str(
                        missing_rate) + '_missing_rate_test.pt')
                model.load_state_dict(state_dict)
        else:
            if stage == 'train':
                print("Initializing fresh model")
                model = vanilla_VAE(obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type,
                                    num_samples,
                                    num_estimates)
            else:
                print("Loading saved model")
                model = vanilla_VAE(obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type,
                                    num_samples,
                                    num_estimates)
                state_dict = torch.load(
                    'experiments/' + experiment_type + '/' + data_type + '/checkpoints/' + ''.join(
                        [i for i in 'vanilla_vae' if
                         not i.isdigit()]) + '/' + 'checkpoint_' + vae_type + '_' + str(
                        missing_rate) + '_missing_rate_test.pt')
                model.load_state_dict(state_dict)
    elif 'vanilla_EDDI' in vae_type:
        if 'mnist' == data_type:
            if stage == 'train':
                print("Initializing fresh model")
                model = vanilla_EDDI_mnist(obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type,
                                           num_samples,
                                           num_estimates)
            else:
                print("Loading saved model")
                model = vanilla_EDDI_mnist(obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type,
                                           num_samples,
                                           num_estimates)
                state_dict = torch.load(
                    'experiments/' + experiment_type + '/' + data_type + '/checkpoints/' + ''.join(
                        [i for i in 'vanilla_EDDI' if
                         not i.isdigit()]) + '/' + 'checkpoint_' + vae_type + '_' + str(
                        missing_rate) + '_missing_rate_test.pt')
                model.load_state_dict(state_dict)
        else:
            if stage == 'train':
                print("Initializing fresh model")
                model = vanilla_EDDI(obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type,
                                     num_samples,
                                     num_estimates)
            else:
                print("Loading saved model")
                model = vanilla_EDDI(obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type,
                                     num_samples,
                                     num_estimates)
                state_dict = torch.load(
                    'experiments/' + experiment_type + '/' + data_type + '/checkpoints/' + ''.join(
                        [i for i in 'vanilla_EDDI' if
                         not i.isdigit()]) + '/' + 'checkpoint_' + vae_type + '_' + str(
                        missing_rate) + '_missing_rate_test.pt')
                model.load_state_dict(state_dict)
    elif 'vanilla_notMIWAE' in vae_type:
        if stage == 'train':
            print("Initializing fresh model")
            model = notMIWAE_myversion(obs_dim, hid_dim, K, latent_dim, training_parameters, num_samples,
                                       num_estimates)
        else:
            print("Loading saved model")
            model = notMIWAE_myversion(obs_dim, hid_dim, K, latent_dim, training_parameters, num_samples,
                                       num_estimates)
            state_dict = torch.load(
                'experiments/' + experiment_type + '/' + data_type + '/checkpoints/' + ''.join(
                    [i for i in vae_type if
                     not i.isdigit()]) + '/' + 'checkpoint_' + vae_type + '_' + str(
                    missing_rate) + '_missing_rate_test.pt')
            model.load_state_dict(state_dict)
    else:
        if stage == 'train':
            print("Initializing fresh model")
            model = MIWAE(obs_dim, hid_dim, K, latent_dim, training_parameters, num_samples, num_estimates)
        else:
            print("Loading saved model")
            model = MIWAE(obs_dim, hid_dim, K, latent_dim, training_parameters, num_samples, num_estimates)
            state_dict = torch.load(
                'experiments/' + experiment_type + '/' + data_type + '/checkpoints/' + ''.join(
                    [i for i in vae_type if not i.isdigit()]) + '/' + 'checkpoint_' + vae_type  + '_' + str(
                       missing_rate) + '_missing_rate_test.pt')
            model.load_state_dict(state_dict)
    return model


def data_loader_mnist(data_path, vae_type, missing_rate, batch_size, data_type, device=torch.device('cpu'),
                      shuffle=True,
                      data_transform='minmax'):
    # import os
    # index = [i for i in vae_type if i.isdigit()][0]
    # train_indices = loadtxt(os.path.join(data_path, data_type, 'train_index' + index + '.csv'), delimiter=',')
    # test_indices = loadtxt(os.path.join(data_path, data_type, 'test_index' + index + '.csv'), delimiter=',')
    # valid_indices = loadtxt(os.path.join(data_path, data_type, 'valid_index' + index + '_es.csv'), delimiter=',')

    # valid_indices = loadtxt(os.path.join(data_path, data_type, 'valid_index.csv'), delimiter=',')
    # data = torch.load('Data/experiment_data/imputed_data_30_missing.pt')
    # data = torch.load(os.path.join(data_path, data_type, 'data.pt'))
    # temp_perm = torch.load('Data/' + data_type + '/rand_perm' + index + '.pt').numpy()
    # data = data[temp_perm, :]
    # temp = [i for i in range(data.shape[0])]
    # train_indices, test_indices = train_test_split(temp, test_size=0.1, random_state=5)
    # p_samples = torch.randperm(data_q.shape[1])
    # mask = create_missing_toy(1000, missing_rate)
    # mask = create_missing_uci(data.shape, 30)
    # mask = torch.load(os.path.join(data_path, data_type, 'mask_' + str(missing_rate) + '_missing' + index + '.pt'))
    # print(mask)
    # mask_p = torch.load(os.path.join(data_path, data_type, 'reg_mask_' + str(missing_rate) + '_missing.pt'))
    # mask_p = torch.load('Data/toy_data/reg_mask_30_missing.pt')
    # mask_p = torch.stack([torch.ones(1000, dtype=torch.bool), torch.zeros(1000, dtype=torch.bool)], 1)
    # index = torch.Tensor([i + 1 for i in range(data.shape[0])])
    # if data_type == 'bos_housing' or data_type == 'concrete' or data_type == 'wine' or data_type == 'yacht' or data_type == 'enb' or data_type == 'kin8nm':
    # if data_transform == 'minmax':
    #    ### data preprocess for my version
    #    max_Data = 1  #
    #    min_Data = 0  #
    #    Data_std = (data - data.min(axis=0).values) / (data.max(axis=0).values - data.min(axis=0).values)
    #    data = Data_std * (max_Data - min_Data) + min_Data
    # else:
    ### data preprocess for notMIWAE version
    #    data = data - data.mean(0)
    #    data = data / data.std(0)
    data_train = torch.load(os.path.join(data_path, data_type, 'experiment_train_data.pt'))
    # data_valid = data[valid_indices]
    data_test = torch.load(os.path.join(data_path, data_type, 'experiment_test_data.pt'))
    mask_train = torch.load(os.path.join(data_path, data_type, 'experiment_train_mask.pt'))
    mask_test = torch.load(os.path.join(data_path, data_type, 'experiment_test_mask.pt'))
    # mask_valid = mask[valid_indices]
    # mask_train_p = mask_p[train_indices]
    # mask_test_p = mask_p[test_indices]
    # mask_valid_p = mask_p[valid_indices]
    # index_train = index[train_indices]
    # index_valid = index[valid_indices]
    # index_test = index[test_indices]
    # setup data loaders
    data_loader_train = DataLoader(
        ConcatDataset(data_train.to(device), mask_train.to(device)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False, num_workers=0)

    # data_loader_valid = DataLoader(
    #    ConcatDataset(data_valid.to(device), mask_valid.to(device)),
    #    batch_size=batch_size,
    #    shuffle=shuffle,
    #    drop_last=False, num_workers=0)

    data_loader_test = DataLoader(
        ConcatDataset(data_test.to(device), mask_test.to(device)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False, num_workers=0)
    obs_dim = 28 * 28
    return [data_loader_train, 'train'], [data_loader_test, 'test'], obs_dim


def data_loader(data_path, vae_type, missing_rate, batch_size, data_type, device=torch.device('cpu'), shuffle=True,
                data_transform='minmax'):
    import os
    index = [i for i in vae_type if i.isdigit()][0]
    train_indices = loadtxt(os.path.join(data_path, data_type, 'train_index' + index + '.csv'), delimiter=',')
    test_indices = loadtxt(os.path.join(data_path, data_type, 'test_index' + index + '.csv'), delimiter=',')
    data = torch.load(os.path.join(data_path, data_type, 'data.pt'))
    mask = torch.load(os.path.join(data_path, data_type, 'mask_' + str(missing_rate) + '_missing' + index + '.pt'))
    if data_transform == 'minmax':
        ### data preprocess for my version
        max_Data = 1  #
        min_Data = 0  #
        Data_std = (data - data.min(axis=0).values) / (data.max(axis=0).values - data.min(axis=0).values)
        data = Data_std * (max_Data - min_Data) + min_Data
    else:
        ### data preprocess for notMIWAE version
        data = data - data.mean(0)
        data = data / data.std(0)
    data_train = data[train_indices]
    data_test = data[test_indices]
    mask_train = mask[train_indices]
    mask_test = mask[test_indices]
    # setup data loaders
    data_loader_train = DataLoader(
        ConcatDataset(data_train.to(device), mask_train.to(device)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False, num_workers=0)

    data_loader_test = DataLoader(
        ConcatDataset(data_test.to(device), mask_test.to(device)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False, num_workers=0)
    obs_dim = data.shape[1]
    return [data_loader_train, 'train'], [data_loader_test, 'test'], obs_dim


def data_loader_mnar(data_path, vae_type, missing_rate, batch_size, data_type, device=torch.device('cpu'), shuffle=True,
                     data_transform='minmax'):
    import os
    index = [i for i in vae_type if i.isdigit()][0]
    data = torch.load(os.path.join(data_path, data_type, 'data.pt'))
    temp_perm = torch.load('Data/' + data_type + '/rand_perm' + index + '.pt').numpy()
    data = data[temp_perm, :]
    data = data[:, :-1]
    mask = torch.load(os.path.join(data_path, data_type, 'mnar_mask_missing' + index + '.pt'))
    mask = mask[:, :-1]
    if data_transform == 'minmax':
        ### data preprocess for my version
        max_Data = 1  #
        min_Data = 0  #
        Data_std = (data - data.min(axis=0).values) / (data.max(axis=0).values - data.min(axis=0).values)
        data = Data_std * (max_Data - min_Data) + min_Data
    else:
        ### data preprocess for notMIWAE version
        data = data - data.mean(0)
        data = data / data.std(0)
    # setup data loaders
    data_loader_train = DataLoader(
        ConcatDataset(data.to(device), mask.to(device)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False, num_workers=0)
    obs_dim = data.shape[1]
    return data_loader_train, obs_dim


# taken from https://stackoverflow.com/questions/58367385/does-a-dataloader-created-from-concatdataset-create-a-batch-from-a-different

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
