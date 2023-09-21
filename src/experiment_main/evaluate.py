import torch
from src.utils.loaders import model_loader
import numpy as np
import torch.nn.functional as F
from src.utils.utils import create_missing_uci, \
    create_missing_toy, introduce_mising, \
    introduce_mising_mnar_based_on_mean_half_features, \
    introduce_mising_mnar_based_on_mean_all_features, introduce_mising_mnar_based_on_variance_all_features, \
    introduce_mising_mnar_based_on_variance_half_features
import matplotlib.pyplot as plt


def eval_vae_mnar(data_test, mask_test, missing_rate, obs_dim, hid_dim, K, M, latent_dim,
                  data_type,
                  training_parameters, experiment_type, vae_type, max_epochs, valid_k, num_estimates,
                  device=torch.device('cpu'), alpha=0.5, stage='evaluate', p_missingness=30, reg_type='ml_reg',
                  beta=1.0,
                  beta_annealing=False, alpha_annealing=True, not_miwae_type='changed'):
    with torch.no_grad():
        model = model_loader('test', obs_dim, hid_dim, K, latent_dim, missing_rate, data_type,
                             training_parameters, max_epochs, valid_k, num_estimates, experiment_type, reg_type,
                             vae_type,
                             alpha=alpha, p_missingness=p_missingness, not_miwae_type=not_miwae_type)
        model.to(device)
        opt_epoch = max_epochs
        N = data_test.shape[0]
        temp_recon = []
        for _ in range(M):
            XM = torch.zeros_like(data_test)
            for i in range(N):
                temp_mask = create_missing_uci(data_test.shape, p_missingness)
                mask_p = mask_test * temp_mask
                if 'reg_notMIWAE' in vae_type:
                    mean_p, logvar_p, x_mean_p, x_logvar_p, mean_q, logvar_q, x_mean_q, x_logvar_q = model.forward(
                        data_test[i, :][None, :], mask_test[i, :][None, :], mask_p[i, :][None, :], stage=stage)
                    xm, train_loss, negl = model.loss(
                        data_test[i, :][None, :], x_mean_p, x_logvar_p, mean_p, logvar_p,
                        x_mean_q, x_logvar_q,
                        mean_q, logvar_q, mask_test[i, :][None, :], mask_p[i, :][None, :], opt_epoch, llh_eval=True,
                        beta_annealing=beta_annealing,
                        beta=beta, alpha=alpha, alpha_annealing=alpha_annealing, stage=stage)
                    XM[i, :] = xm
                else:
                    mean_q, logvar_q, x_mean, x_logvar_q = model.forward(
                        data_test[i, :][None, :], mask_test[i, :][None, :])
                    xm, train_loss, negl = model.loss(
                        data_test[i, :][None, :],
                        x_mean, x_logvar_q,
                        mean_q, logvar_q, opt_epoch, mask_test[i, :][None, :], llh_eval=True,
                        beta_annealing=beta_annealing, beta=beta,
                        stage=stage)
                    XM[i, :] = xm
            temp_recon.append(
                torch.sqrt(torch.sum(torch.square(
                    torch.squeeze(XM) * (1 - mask_test) - data_test.view(-1, obs_dim) * (
                            1 - mask_test))) / torch.sum((1 - mask_test))))
        recon = torch.stack(temp_recon).mean()
        if 'vanilla' in vae_type:
            torch.save(recon,
                       'experiments/' + experiment_type + '/' + data_type + '/rest/' + ''.join(
                           [i for i in vae_type if
                            not i.isdigit()]) + '/' + vae_type + '_rmse' + '_' + not_miwae_type + '_large_batch_test.pt')

        else:
            torch.save(recon,
                       'experiments/' + experiment_type + '/' + data_type + '/rest/' + ''.join(
                           [i for i in vae_type if
                            not i.isdigit()]) + '/' + vae_type + '_rmse' + '_' + str(
                           alpha) + '_' + str(p_missingness) + '_' + reg_type + '_full_reg_large_batch_v2_test.pt')


def eval_miwae(list_loaders, missing_rate, obs_dim, hid_dim, K, M, latent_dim,
               data_type,
               training_parameters, experiment_type, vae_type, max_epochs, valid_k, num_estimates,
               device=torch.device('cpu'), alpha=0.5, stage='evaluate', p_missingness=30, reg_type='ml_reg',
               beta=1.0,
               beta_annealing=False, alpha_annealing=True):
    with torch.no_grad():
        model = model_loader('test', obs_dim, hid_dim, K, latent_dim, missing_rate, data_type,
                             training_parameters, max_epochs, valid_k, num_estimates, experiment_type, reg_type,
                             vae_type,
                             alpha=alpha, p_missingness=p_missingness)
        model.to(device)
        opt_epoch = max_epochs
        for loader in list_loaders:
            loader, loader_stage = loader
            recon = []
            for _ in range(M):
                XM = []
                for batch_idx, sample in enumerate(loader):
                    data_sample, mask = sample
                    temp_XM = torch.zeros_like(data_sample)
                    temp_mask = create_missing_uci(data_sample.shape, p_missingness)
                    mask_p = mask * temp_mask
                    for i in range(data_sample.shape[0]):
                        if 'reg_MIWAE' in vae_type:
                            mean_p, scale_p, x_mean_p, x_scale_p, deg_free_p, mean_q, scale_q, x_mean_q, x_scale_q, deg_free_q = model.forward(
                                data_sample[i, :][None, :], mask[i, :][None, :], mask_p[i, :][None, :])
                            xm, train_loss, _ = model.loss(
                                data_sample[i, :][None, :], x_mean_p, x_scale_p, deg_free_p, mean_p, scale_p,
                                x_mean_q, x_scale_q, deg_free_q,
                                mean_q, scale_q, mask[i, :][None, :], mask_p[i, :][None, :], i + 1, llh_eval=True,
                                beta_annealing=beta_annealing, beta=beta, alpha=alpha)
                            temp_XM[i, :] = xm
                        else:
                            mean, scale, x_mean, x_scale, deg_free = model.forward(
                                data_sample[i, :][None, :], mask[i, :][None, :])
                            xm, train_loss, _ = model.loss(
                                data_sample[i, :][None, :],
                                x_mean, x_scale, deg_free,
                                mean, scale, mask[i, :][None, :], opt_epoch, llh_eval=True,
                                beta_annealing=beta_annealing, beta=beta)
                            temp_XM[i, :] = xm
                    XM.append(
                        torch.sqrt(torch.sum(torch.square(
                            torch.squeeze(temp_XM) * ~mask - data_sample.view(-1, obs_dim) * ~mask)) / torch.sum(
                            ~mask)))
                recon.append(torch.stack(XM).mean())
            recon = torch.stack(recon).mean()
            if 'vanilla' in vae_type:
                torch.save(recon,
                           'experiments/' + experiment_type + '/' + data_type + '/rest/' + ''.join(
                               [i for i in '_'.join(vae_type.split('_')[:2]) if
                                not i.isdigit()]) + '/' + loader_stage + '_' + vae_type + '_rmse_50_missing_rate_test.pt')


            else:
                torch.save(recon,
                           'experiments/' + experiment_type + '/' + data_type + '/rest/' + ''.join(
                               [i for i in '_'.join(vae_type.split('_')[:2]) if
                                not i.isdigit()]) + '/' + loader_stage + '_' + vae_type + '_rmse' + '_' + str(
                               alpha) + '_' + str(
                               p_missingness) + '_' + reg_type + '_full_reg_50_missing_rate_test.pt')


def eval_vae(list_loaders, missing_rate, obs_dim, hid_dim, K, M, latent_dim,
             data_type,
             training_parameters, experiment_type, vae_type, max_epochs, valid_k, num_estimates,
             device=torch.device('cpu'), alpha=0.5, stage='evaluate', p_missingness=30, reg_type='ml_reg', beta=1.0,
             beta_annealing=False, alpha_annealing=True):
    with torch.no_grad():
        model = model_loader('test', obs_dim, hid_dim, K, latent_dim, missing_rate, data_type,
                             training_parameters, max_epochs, valid_k, num_estimates, experiment_type, reg_type,
                             vae_type,
                             alpha=alpha, p_missingness=p_missingness)
        model.to(device)
        opt_epoch = max_epochs
        # opt_epoch = 300
        for loader in list_loaders:
            loader, loader_stage = loader
            res = []
            res_negll = []
            res_negll_imp = []
            recon = []
            recon_miss = []
            # mask_list = []
            for _ in range(M):
                elbos = []
                negls = []
                negls_imp = []
                temp_recon = []
                temp_recon_miss = []
                # temp_mask = []
                for batch_idx, sample in enumerate(loader):
                    data_sample, mask = sample
                    if data_type == 'mnist':
                        mask_p = []
                        for j in range(len(data_sample)):
                            temp_mask = create_missing_uci(data_sample[j].shape, p_missingness)
                            mask_p.append(mask[j] * temp_mask)
                        mask_p = torch.stack(mask_p, 0)
                    else:
                        temp_mask = create_missing_uci(data_sample.shape, p_missingness)
                        mask_p = temp_mask * mask
                    '''
                    if p_missingness == 'half_features_mnar_mean':
                        temp_mask = introduce_mising_mnar_based_on_mean_half_features(data_sample)
                        mask_p = temp_mask * mask
                    elif p_missingness == 'all_features_mnar_mean':
                        temp_mask = introduce_mising_mnar_based_on_mean_all_features(data_sample)
                        mask_p = temp_mask * mask
                    elif p_missingness == 'half_features_mnar_var':
                        temp_mask = introduce_mising_mnar_based_on_variance_half_features(data_sample)
                        mask_p = temp_mask * mask
                    elif p_missingness == 'all_features_mnar_var':
                        temp_mask = introduce_mising_mnar_based_on_variance_all_features(data_sample)
                        mask_p = temp_mask * mask
                    '''
                    if 'flow' in vae_type:
                        if 'reg_flow' in vae_type:
                            z_p, z_log_prob_p, x_mean_p, x_logvar_p, z_q, z_log_prob_q, x_mean_q, x_logvar_q = model.forward(
                                data_sample, mask, mask_p)
                            print_loss, train_loss, negl, negl_imp = model.loss(data_sample, x_mean_q, x_logvar_q,
                                                                                z_q,
                                                                                z_log_prob_q,
                                                                                x_mean_p, x_logvar_p, z_p,
                                                                                z_log_prob_p, mask,
                                                                                mask_p,
                                                                                alpha, llh_eval=True, stage=stage)
                            x_mean = x_mean_q

                        else:
                            z, z_log_prob, x_mean, x_logvar = model.forward(
                                data_sample, mask)
                            print_loss, train_loss, negl, negl_imp = model.loss(data_sample, x_mean, x_logvar, z,
                                                                                z_log_prob,
                                                                                mask, llh_eval=True)

                    elif 'reg_vae' in vae_type or 'reg_EDDI' in vae_type or 'reg_mnist' in vae_type:
                        mean_p, logvar_p, x_mean_p, x_logvar_p, mean_q, logvar_q, x_mean_q, x_logvar_q = model.forward(
                            data_sample, mask, mask_p, stage=stage)
                        print_loss, train_loss, negl, negl_imp = model.loss(
                            data_sample, x_mean_p, x_logvar_p, mean_p, logvar_p,
                            x_mean_q, x_logvar_q,
                            mean_q, logvar_q, mask, mask_p, opt_epoch, llh_eval=True, beta_annealing=beta_annealing,
                            beta=beta, alpha=alpha, alpha_annealing=alpha_annealing, stage=stage)
                        x_mean = x_mean_q
                    else:
                        # elif 'vanilla_vae' in vae_type or 'vanilla_EDDI' in vae_type or 'vanilla_notMIWAE' in vae_type:
                        mean_q, logvar_q, x_mean, x_logvar_q = model.forward(
                            data_sample, mask)
                        print_loss, train_loss, negl, negl_imp = model.loss(
                            data_sample,
                            x_mean, x_logvar_q,
                            mean_q, logvar_q, opt_epoch, mask, llh_eval=True, beta_annealing=beta_annealing, beta=beta,
                            stage=stage)
                        if 'MIWAE' in vae_type:
                            x_mean = x_mean[:, 0, :]
                        else:
                            x_mean = x_mean
                    mask = mask.reshape(-1, obs_dim)
                    temp_recon.append(
                        torch.sqrt(torch.sum(torch.square(
                            torch.squeeze(x_mean) * ~mask - data_sample.view(-1, obs_dim) * ~mask)) / torch.sum(~mask)))
                    elbos.append(train_loss)
                    negls.append(negl)
                    negls_imp.append(negl_imp)
                recon.append(torch.stack(temp_recon).mean())
                res.append(torch.mean(torch.stack(elbos)))
                res_negll.append(torch.mean(torch.stack(negls)))
                res_negll_imp.append(torch.mean(torch.stack(negls_imp)))
            recon = torch.stack(recon).mean()
            res = torch.stack(res).mean()
            res_negll = torch.stack(res_negll).mean()
            res_negll_imp = torch.stack(res_negll_imp).mean()

            if 'vanilla' in vae_type:
                torch.save(recon,
                           'experiments/' + experiment_type + '/' + data_type + '/rest/' + ''.join(
                               [i for i in '_'.join(vae_type.split('_')[:2]) if
                                not i.isdigit()]) + '/' + loader_stage + '_' + vae_type + '_rmse' + '_' + str(
                               missing_rate) + '_missing_rate_test.pt')

                torch.save(res,
                           'experiments/' + experiment_type + '/' + data_type + '/elbos/' + ''.join(
                               [i for i in '_'.join(vae_type.split('_')[:2]) if
                                not i.isdigit()]) + '/' + loader_stage + '_' + vae_type + '_vae_elbo' + '_' + str(
                               missing_rate) + '_missing_rate_test.pt')
                torch.save(res_negll,
                           'experiments/' + experiment_type + '/' + data_type + '/rest/' + ''.join(
                               [i for i in '_'.join(vae_type.split('_')[:2]) if
                                not i.isdigit()]) + '/' + loader_stage + '_' + vae_type + '_negative_llh' + '_' + str(
                               missing_rate) + '_missing_rate_test.pt')
                torch.save(res_negll_imp,
                           'experiments/' + experiment_type + '/' + data_type + '/rest/' + ''.join(
                               [i for i in '_'.join(vae_type.split('_')[:2]) if
                                not i.isdigit()]) + '/' + loader_stage + '_' + vae_type + '_negative_llh_imputed' + '_' + str(
                               missing_rate) + '_missing_rate_test.pt')
            else:
                torch.save(recon,
                           'experiments/' + experiment_type + '/' + data_type + '/rest/' + ''.join(
                               [i for i in '_'.join(vae_type.split('_')[:2]) if
                                not i.isdigit()]) + '/' + loader_stage + '_' + vae_type + '_rmse' + '_' + str(
                               alpha) + '_' +
                           str(p_missingness) + '_' + reg_type + '_' + str(
                               missing_rate) + '_missing_rate_full_reg_test.pt')

                torch.save(res,
                           'experiments/' + experiment_type + '/' + data_type + '/elbos/' + ''.join(
                               [i for i in '_'.join(vae_type.split('_')[:2]) if
                                not i.isdigit()]) + '/' + loader_stage + '_' + vae_type + '_vae_elbo' + '_' + str(
                               alpha) + '_' + str(p_missingness) + '_' + reg_type + '_' + str(
                               missing_rate) + '_missing_rate_full_reg_test.pt')
                torch.save(res_negll,
                           'experiments/' + experiment_type + '/' + data_type + '/rest/' + ''.join(
                               [i for i in '_'.join(vae_type.split('_')[:2]) if
                                not i.isdigit()]) + '/' + loader_stage + '_' + vae_type + '_negative_llh_q' + '_' + str(
                               alpha) + '_' +
                           str(p_missingness) + '_' + reg_type + '_' + str(
                               missing_rate) + '_missing_rate_full_reg_test.pt')
                torch.save(res_negll_imp,
                           'experiments/' + experiment_type + '/' + data_type + '/rest/' + ''.join(
                               [i for i in '_'.join(vae_type.split('_')[:2]) if
                                not i.isdigit()]) + '/' + loader_stage + '_' + vae_type + '_negative_llh_q_imputed' + '_' + str(
                               alpha) + '_' +
                           str(p_missingness) + '_' + reg_type + '_' + str(
                               missing_rate) + '_missing_rate_full_reg_test.pt')


def active_learning_func(data_loader_train, test_data, test_mask, missing_rate, obs_dim, hid_dim, K, M,
                         latent_dim,
                         data_type,
                         training_parameters, experiment_type, vae_type, max_epochs, valid_k, num_estimates,
                         device=torch.device('cpu'), alpha=1.0, stage='evaluate', p_missingness=30, reg_type='ml_reg',
                         beta=1.0,
                         beta_annealing=False, alpha_annealing=True, Repeat=5):
    for r in range(Repeat):
        n_test = test_data.shape[0]
        # train_active(drop_type, data_loader_train, missing_rate, obs_dim, hid_dim, K, M, latent_dim, data_type,
        #             training_parameters, experiment_type, vae_type, valid_k, num_estimates, max_epochs=max_epochs,
        #             device=device, alpha=alpha, stage='train', p_missingness=p_missingness, reg_type=reg_type,
        #             beta=beta,
        #             beta_annealing=beta_annealing)
        with torch.no_grad():
            model = model_loader('test', obs_dim, hid_dim, K, latent_dim, missing_rate, data_type,
                                 training_parameters, max_epochs, valid_k, num_estimates, experiment_type,
                                 reg_type,
                                 vae_type,
                                 alpha=alpha, p_missingness=p_missingness,
                                 alpha_annealing=alpha_annealing)
            model.to(device)
            if r == 0:
                # information curves
                information_curve_RAND = torch.zeros(
                    Repeat, n_test, obs_dim - 1 + 1)
                information_curve_SING = torch.zeros(
                    Repeat, n_test, obs_dim - 1 + 1)
                information_curve_CHAI = torch.zeros(
                    Repeat, n_test, obs_dim - 1 + 1).float()

                # history of optimal actions
                action_SING = torch.zeros(Repeat, n_test,
                                          obs_dim - 1)
                action_CHAI = torch.zeros(Repeat, n_test,
                                          obs_dim - 1).float()

                # history of information reward values
                R_hist_SING = torch.zeros(
                    Repeat, obs_dim - 1, n_test,
                            obs_dim - 1)
                R_hist_CHAI = torch.zeros(
                    Repeat, obs_dim - 1, n_test,
                            obs_dim - 1).float()

                # history of posterior samples of partial inference
                im_SING = np.zeros((Repeat, obs_dim - 1, M,
                                    n_test, obs_dim))
                im_CHAI = torch.zeros(Repeat, obs_dim - 1, M,
                                      n_test, obs_dim).float()

            temp_mask = create_missing_uci(test_data.shape, p_missingness)
            mask_p = test_mask * temp_mask

            x = test_data[:, :].float()  #
            # x = np.reshape(x, [n_test, OBS_DIM])
            mask = torch.zeros(n_test,
                               obs_dim)  # this stores the mask of missingness (stems from both test data missingness and unselected features during active learing)
            mask2 = torch.zeros(n_test,
                                obs_dim)  # this stores the mask indicating that which features has been selected of each data
            mask[:,
            -1] = 0  # Note that no matter how you initialize mask, we always keep the target variable (last column) unobserved.
            # temp_mask = torch.from_numpy(create_missing_uci(data_sample.shape, 30))
            # mask = temp_mask
            temp_recon = []
            for _ in range(M):
                if 'flow' in vae_type:
                    if 'reg_flow' in vae_type:
                        z_p, z_log_prob_p, x_mean_p, x_logvar_p, z_q, z_log_prob_q, x_mean_q, x_logvar_q = model.forward(
                            x, mask, mask_p)
                        x_mean = x_mean_q
                    else:
                        z, z_log_prob, x_mean, x_logvar = model.forward(
                            x, mask)
                elif 'reg_vae' in vae_type or 'reg_EDDI' in vae_type or 'reg_MIWAE' in vae_type or 'reg_mnist' in vae_type:
                    mean_p, logvar_p, x_mean_p, x_logvar_p, mean_q, logvar_q, x_mean_q, x_logvar_q = model.forward(
                        x, mask, mask_p, stage=stage)
                    x_mean = x_mean_q
                elif 'vanilla_vae' in vae_type or 'vanilla_EDDI' in vae_type:
                    mean_q, logvar_q, x_mean_q, x_logvar_q = model.forward(
                        x, mask)
                    x_mean = x_mean_q
                else:
                    mean, logvar, x_mean, x_logvar = model.forward(x, mask)
                temp_recon.append(F.mse_loss(torch.squeeze(x_mean)[:, -1], test_data.view(-1, obs_dim)[:, -1]))
            negative_predictive_llh = torch.stack(temp_recon).mean()
            information_curve_CHAI[r, :, 0] = negative_predictive_llh
            for t in range(obs_dim - 1):  # t is a indicator of step
                print("Repeat = {:.1f}".format(r))
                print("Strategy = {:.1f}".format(2))
                print("Step = {:.1f}".format(t))
                R = -1e4 * torch.ones(n_test, obs_dim - 1)
                # im = completion(x, mask, M, vae)
                temp_im = []
                for _ in range(M):
                    if 'flow' in vae_type:
                        if 'reg_flow' in vae_type:
                            z_p, z_log_prob_p, x_mean_p, x_logvar_p, z_q, z_log_prob_q, x_mean_q, x_logvar_q = model.forward(
                                x, mask, mask_p)
                            x_mean = x_mean_q
                        else:
                            z, z_log_prob, x_mean, x_logvar = model.forward(
                                x, mask)
                    elif 'reg_vae' in vae_type or 'reg_EDDI' in vae_type or 'reg_MIWAE' in vae_type or 'reg_mnist' in vae_type:
                        mean_p, logvar_p, x_mean_p, x_logvar_p, mean_q, logvar_q, x_mean_q, x_logvar_q = model.forward(
                            x, mask, mask_p, stage=stage)
                        x_mean = x_mean_q
                    elif 'vanilla_vae' in vae_type or 'vanilla_EDDI' in vae_type:
                        mean_q, logvar_q, x_mean_q, x_logvar_q = model.forward(
                            x, mask)
                        x_mean = x_mean_q
                    else:
                        mean, logvar, x_mean, x_logvar = model.forward(x, mask)
                    temp_im.append(x_mean)
                im = torch.stack(temp_im, 0)
                im_CHAI[r, t, :, :, :] = im
                for u in range(
                        obs_dim - 1):  # u is the indicator for features. calculate reward function for each feature candidates
                    loc = np.where(mask[:, u] == 0)[0]
                    # print('loc: ', loc)
                    if 'flow' in vae_type:
                        R[loc, u] = R_lindley_chain_ratio_version(u, x, mask, M, model, im,
                                                                  loc).float()
                    else:
                        R[loc, u] = R_lindley_chain(u, x, mask, M, model, im,
                                                    loc).float()
                R_hist_CHAI[r, t, :, :] = R
                i_optimal = R.argmax(axis=1)

                # i_optimal = optimal_matrix[r, :, t].long()
                # print(i_optimal)
                io = torch.eye(obs_dim)[i_optimal]
                action_CHAI[r, :, t] = i_optimal
                mask = mask + io
                temp_recon = []
                for _ in range(M):
                    if 'flow' in vae_type:
                        if 'reg_flow' in vae_type:
                            z_p, z_log_prob_p, x_mean_p, x_logvar_p, z_q, z_log_prob_q, x_mean_q, x_logvar_q = model.forward(
                                x, mask, mask_p)
                            x_mean = x_mean_q
                        else:
                            z, z_log_prob, x_mean, x_logvar = model.forward(
                                x, mask)
                    elif 'reg_vae' in vae_type or 'reg_EDDI' in vae_type or 'reg_MIWAE' in vae_type or 'reg_mnist' in vae_type:
                        mean_p, logvar_p, x_mean_p, x_logvar_p, mean_q, logvar_q, x_mean_q, x_logvar_q = model.forward(
                            x, mask, mask_p, stage=stage)
                        x_mean = x_mean_q
                    elif 'vanilla_vae' in vae_type or 'vanilla_EDDI' in vae_type:
                        mean_q, logvar_q, x_mean_q, x_logvar_q = model.forward(
                            x, mask)
                        x_mean = x_mean_q
                    else:
                        mean, logvar, x_mean, x_logvar = model.forward(x, mask)
                    temp_recon.append(F.mse_loss(torch.squeeze(x_mean)[:, -1], test_data.view(-1, obs_dim)[:, -1]))

                # negative_predictive_llh = torch.stack(temp_recon, 0).sum(0) / M
                negative_predictive_llh = torch.stack(temp_recon).mean()
                mask2 = mask2 + io  # this mask only stores missingess of unselected features, i.e., which features has been selected of each data
                information_curve_CHAI[r, :, t + 1] = negative_predictive_llh
    if 'vanilla' in vae_type:
        torch.save(information_curve_CHAI,
                   'experiments/' + experiment_type + '/' + data_type + '/rest/' + ''.join(
                       [i for i in '_'.join(vae_type.split('_')[:2]) if
                        not i.isdigit()]) + '/' + vae_type + '_' + str(
                       missing_rate) + '_missing_rate_UCI_information_curve_CHAI_default_test.pt')

        torch.save(action_CHAI,
                   'experiments/' + experiment_type + '/' + data_type + '/rest/' + ''.join(
                       [i for i in '_'.join(vae_type.split('_')[:2]) if
                        not i.isdigit()]) + '/' + vae_type + '_' + str(
                       missing_rate) + '_missing_rate__UCI_action_CHAI_default_test.pt')
        torch.save(R_hist_CHAI,
                   'experiments/' + experiment_type + '/' + data_type + '/rest/' + ''.join(
                       [i for i in '_'.join(vae_type.split('_')[:2]) if
                        not i.isdigit()]) + '/' + vae_type + '_' + str(
                       missing_rate) + '_missing_rate__UCI_R_hist_CHAI_default_test.pt')
        torch.save(im_CHAI,
                   'experiments/' + experiment_type + '/' + data_type + '/rest/' + ''.join(
                       [i for i in '_'.join(vae_type.split('_')[:2]) if
                        not i.isdigit()]) + '/' + vae_type + '_' + str(
                       missing_rate) + '_missing_rate__UCI_im_CHAI_default_test.pt')

    else:
        torch.save(information_curve_CHAI,
                   'experiments/' + experiment_type + '/' + data_type + '/rest/' + ''.join(
                       [i for i in '_'.join(vae_type.split('_')[:2]) if
                        not i.isdigit()]) + '/' + vae_type + '_UCI_information_curve_CHAI' + '_' + str(
                       alpha) + '_' + str(
                       p_missingness) + '_' + reg_type + '_' + str(
                       missing_rate) + '_missing_rate_default_full_reg_test.pt')
        torch.save(action_CHAI,
                   'experiments/' + experiment_type + '/' + data_type + '/rest/' + ''.join(
                       [i for i in '_'.join(vae_type.split('_')[:2]) if
                        not i.isdigit()]) + '/' + vae_type + '_UCI_action_CHAI' + '_' + str(
                       alpha) + '_' + str(
                       p_missingness) + '_' + reg_type + '_' + str(
                       missing_rate) + '_missing_rate_default_full_reg_test.pt')
        torch.save(R_hist_CHAI,
                   'experiments/' + experiment_type + '/' + data_type + '/rest/' + ''.join(
                       [i for i in '_'.join(vae_type.split('_')[:2]) if
                        not i.isdigit()]) + '/' + vae_type + '_UCI_R_hist_CHAI' + '_' + str(
                       alpha) + '_' + str(
                       p_missingness) + '_' + reg_type + '_' + str(
                       missing_rate) + '_missing_rate_default_full_reg_test.pt')
        torch.save(im_CHAI,
                   'experiments/' + experiment_type + '/' + data_type + '/rest/' + ''.join(
                       [i for i in '_'.join(vae_type.split('_')[:2]) if
                        not i.isdigit()]) + '/' + vae_type + '_UCI_im_CHAI' + '_' + str(
                       alpha) + '_' + str(
                       p_missingness) + '_' + reg_type + '_' + str(
                       missing_rate) + '_missing_rate_default_full_reg_test.pt')


def R_lindley_chain(i, x, mask, M, vae, im, loc):
    '''
    function for computing reward function approximation
    :param i: indicates the index of x_i
    :param x: data matrix
    :param mask: mask of missingness
    :param M: number of MC samples
    :param vae: a pre-trained vae
    :param im: sampled missing data, a M by N by D matrix, where M is the number of samples.
    :return:
    '''

    im_i = im[:, :, i]
    # print(im_i)
    approx_KL = 0
    im_target = im[:, :, -1]
    temp_x = x.clone()
    for m in range(M):
        temp_x[loc, i] = im_i[m, loc].float()
        KL_I = chaini_I(temp_x[loc, :], mask[loc, :], i, vae)
        temp_x[loc, -1] = im_target[m, loc].float()
        KL_II = chaini_II(temp_x[loc, :], mask[loc, :], i, vae)

        approx_KL += KL_I
        approx_KL -= KL_II

    R = approx_KL / M

    return R


## calculate the first term of information reward approximation
def chaini_I(x, mask, i, vae):
    '''
    calculate the first term of information reward approximation
    used only in active learning phase
    :param x: data
    :param mask: mask of missingness
    :param i: indicates the index of x_i
    :return:  the first term of information reward approximation
    '''

    temp_mask = mask.clone()
    # m, v = self._sesh.run([self.mean, self.stddev],
    #                      feed_dict={
    #                          self.x: x,
    #                          self.mask: temp_mask
    #                      })
    _, mean, logvar = vae.encoder(x, temp_mask)
    var = torch.exp(logvar)
    v = torch.exp(logvar / 2)
    # log_var = 2 * np.log(v)

    temp_mask[:, i] = 1

    _, mean_i, logvar_i = vae.encoder(x, temp_mask)
    var_i = torch.exp(logvar_i)
    v_i = torch.exp(logvar_i / 2)

    # m_i, v_i = self._sesh.run([self.mean, self.stddev],
    #                          feed_dict={
    #                              self.x: x,
    #                              self.mask: temp_mask
    #                          })

    # var_i = v_i**2
    # log_var_i = 2 * np.log(v_i)

    kl_i = 0.5 * torch.sum(
        torch.square(mean_i - mean) / v + var_i / var - 1. - logvar_i + logvar, 1)

    return kl_i


## calculate the second term of information reward approximation
def chaini_II(x, mask, i, vae):
    '''
    calculate the second term of information reward approximation
    used only in active learning phase
    Note that we assume that the last column of x is the target variable of interest
    :param x: data
    :param mask: mask of missingness
    :param i: indicates the index of x_i
    :return:  the second term of information reward approximation
    '''
    # mask: represents x_o
    # targets: 0 by M vector, contains M samples from p(\phi|x_o)
    # x : 1 by obs_dim vector, contains 1 instance of data
    # i: indicates the index of x_i
    temp_mask = mask.clone()
    temp_mask[:, -1] = 1
    # m, v = self._sesh.run([self.mean, self.stddev],
    #                      feed_dict={
    #                          self.x: x,
    #                          self.mask: temp_mask
    #                      })

    _, mean, logvar = vae.encoder(x, temp_mask)

    var = torch.exp(logvar)
    v = torch.exp(logvar / 2)

    # log_var = 2 * np.log(v)

    temp_mask[:, i] = 1

    # m_i, v_i = self._sesh.run([self.mean, self.stddev],
    #                          feed_dict={
    #                              self.x: x,
    #                              self.mask: temp_mask
    #                          })

    _, mean_i, logvar_i = vae.encoder(x, temp_mask)

    var_i = torch.exp(logvar_i)
    # log_var_i = 2 * np.log(v_i)

    kl_i = 0.5 * torch.sum(
        torch.square(mean_i - mean) / v + var_i / var - 1. - logvar_i + logvar, 1)

    return kl_i


def R_lindley_chain_ratio_version(i, x, mask, M, vae, im, loc):
    '''
    function for computing reward function approximation
    :param i: indicates the index of x_i
    :param x: data matrix
    :param mask: mask of missingness
    :param M: number of MC samples
    :param vae: a pre-trained vae
    :param im: sampled missing data, a M by N by D matrix, where M is the number of samples.
    :return:
    '''

    im_i = im[:, :, i]
    # print(im_i)
    approx_KL = 0
    im_target = im[:, :, -1]
    temp_x = x.clone().float()
    for m in range(M):
        temp_x[loc, i] = im_i[m, loc].float()
        KL_I = chaini_I_ratio_version(temp_x[loc, :], mask[loc, :], i, vae)
        temp_x[loc, -1] = im_target[m, loc].float()
        KL_II = chaini_II_ratio_version(temp_x[loc, :], mask[loc, :], i, vae)

        approx_KL += KL_I
        approx_KL -= KL_II

    R = approx_KL / M

    return R


## calculate the first term of information reward approximation
def chaini_I_ratio_version(x, mask, i, vae):
    '''
    calculate the first term of information reward approximation
    used only in active learning phase
    :param x: data
    :param mask: mask of missingness
    :param i: indicates the index of x_i
    :return:  the first term of information reward approximation
    '''

    temp_mask = mask.clone()
    _, log_prob = vae.encoder(x, temp_mask)
    temp_mask[:, i] = 1
    _, log_prob_i = vae.encoder(x, temp_mask)
    # print(log_prob.shape)
    return torch.abs(log_prob - log_prob_i).sum(1)


## calculate the second term of information reward approximation
def chaini_II_ratio_version(x, mask, i, vae):
    '''
    calculate the second term of information reward approximation
    used only in active learning phase
    Note that we assume that the last column of x is the target variable of interest
    :param x: data
    :param mask: mask of missingness
    :param i: indicates the index of x_i
    :return:  the second term of information reward approximation
    '''
    # mask: represents x_o
    # targets: 0 by M vector, contains M samples from p(\phi|x_o)
    # x : 1 by obs_dim vector, contains 1 instance of data
    # i: indicates the index of x_i
    temp_mask = mask.clone()
    temp_mask[:, -1] = 1
    _, log_prob = vae.encoder(x, temp_mask)
    temp_mask[:, i] = 1
    _, log_prob_i = vae.encoder(x, temp_mask)
    # print(log_prob.shape)
    return torch.abs(log_prob - log_prob_i).sum(1)
