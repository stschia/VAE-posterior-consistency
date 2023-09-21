import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from src.utils.utils import log_mean_exp
import math
from torch.nn import Sigmoid, Softplus


class Reg_EDDI_mnist(nn.Module):
    """ Reg VAE """

    def __init__(self, obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type, reg_type, num_samples=1,
                 num_estimates=1):
        super(Reg_EDDI_mnist, self).__init__()
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.emb_dim = K
        self.latent_dim = latent_dim
        self.batch_size = training_parameters['batch_size']
        self.training_parameters = training_parameters
        self.experiment_type = experiment_type
        self.num_samples = num_samples
        self.num_estimates = num_estimates
        # encoder: q(z2 | x)
        self.reg_type = reg_type
        self.pnp_encoder1 = nn.Sequential(
            nn.Linear(2 + self.emb_dim, self.emb_dim),
            nn.ReLU()
        )

        self.pnp_encoder2 = nn.Sequential(
            nn.Linear(self.emb_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 2 * latent_dim)
        )
        self.seq_decoder = nn.Sequential(nn.Linear(latent_dim, 200), nn.ReLU(), nn.Linear(200, 500), nn.ReLU(),
                                         nn.Linear(500, 500), nn.ReLU(),
                                         nn.Linear(500, obs_dim),
                                         nn.Sigmoid())
        # self.x_logvar = nn.Sequential(nn.Linear(latent_dim, 50), nn.ReLU(), nn.Linear(50, 100), nn.ReLU(), nn.Linear(100, obs_dim),
        #                              nn.Hardtanh(min_val=-4.5, max_val=0))
        self.x_logvar = torch.log(torch.square(torch.Tensor([0.1 * np.sqrt(2)])))
        # self.type_pars1 = nn.parameter.Parameter(1e-3 * torch.randn(self.obs_dim, self.emb_dim))
        self.type_pars1 = nn.parameter.Parameter(torch.zeros(self.obs_dim, self.emb_dim), requires_grad=True)
        torch.nn.init.xavier_uniform(self.type_pars1)
        self.type_bias1 = nn.parameter.Parameter(torch.zeros(self.obs_dim, 1), requires_grad=True)
        torch.nn.init.xavier_uniform(self.type_bias1)

        # assume factorized prior of normal RVs
        self.prior_mean = torch.nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False)
        self.prior_std = torch.nn.Parameter(torch.ones(self.latent_dim), requires_grad=False)
        self.prior = torch.distributions.Normal(self.prior_mean, self.prior_std)
        self.max_epoch = 2800
        # self.sample = torch.FloatTensor(64, 14).uniform_(-10, -5)

    # TODO add mask again!
    def encoder(self, x, mask, sample=True):
        """
        Perform a point-net like transformation of the data
        """
        if mask.shape[0] == 0:
            return torch.empty(0, 10), torch.empty(0, 10), torch.empty(0, 10)
        else:
            x_flat = x.reshape(-1, 1)
            x_flat = torch.cat(
                [x_flat, x_flat * (self.type_pars1.repeat(x.shape[0], 1, 1)).reshape([x.shape[0] * self.obs_dim, -1]),
                 (self.type_bias1.repeat(x.shape[0], 1, 1)).reshape([x.shape[0] * self.obs_dim, 1])],
                dim=1)
            mask_exp = (mask.reshape(mask.shape[0], mask.shape[1], -1)).repeat(1, 1, self.emb_dim)
            agg = (mask_exp * (self.pnp_encoder1(x_flat.float())).reshape(
                [x.shape[0], self.obs_dim, self.emb_dim])).sum(1)
            mean, logvar = self.pnp_encoder2(agg).chunk(2, dim=1)
            if sample:
                log_std = logvar / 2
                dist = torch.distributions.Normal(mean, torch.exp(log_std))
                z = dist.rsample()
            else:
                z = mean
            return z, mean, logvar

    def decoder(self, z_int):
        decoded_int = self.seq_decoder(z_int)
        x_logvar = self.x_logvar
        # x_logvar = self.x_logvar(z_int)
        return decoded_int, x_logvar

    def loss(self, x, x_recon_p, x_logvar_p, mean_p, logvar_p, x_recon_q, x_logvar_q, mean_q, logvar_q, mask, mask_p,
             epoch,
             vae_elbo=False, llh_eval=False, MI=False, beta_annealing=False,
             beta=1.0, alpha=0.5, stage='train', alpha_annealing=False):

        x = x.reshape(-1, self.obs_dim)
        mask = mask.reshape(-1, self.obs_dim)
        mask_p = mask_p.reshape(-1, self.obs_dim)
        x_logvar_q = torch.ones_like(x_recon_q) * x_logvar_q
        x_logvar_p = torch.ones_like(x_recon_p) * x_logvar_p

        if stage == 'evaluate':
            RE_q = self.neg_gaussian_log_likelihood(x * mask, x_recon_q * mask,
                                                    x_logvar_q * mask)  # TODO add new_mask again
            RE_q_imputed = self.neg_gaussian_log_likelihood(x * ~mask, x_recon_q * ~mask,
                                                            x_logvar_q * ~mask)

            KL_q = self.kl_diagnormal_stdnormal(mean_q, logvar_q)

            if beta_annealing:
                loss_q = RE_q + (epoch / self.max_epoch) * beta * KL_q
            else:
                loss_q = RE_q + beta * KL_q
            loss = loss_q
        else:
            RE_q = self.neg_gaussian_log_likelihood(x * mask, x_recon_q * mask,
                                                    x_logvar_q * mask)  # TODO add new_mask again
            RE_p = self.neg_gaussian_log_likelihood(x * mask_p,
                                                    x_recon_p * mask_p,
                                                    x_logvar_p * mask_p)

            KL_q = self.kl_diagnormal_stdnormal(mean_q, logvar_q)
            KL_p = self.kl_diagnormal_stdnormal(mean_p, logvar_p)

            if beta_annealing:
                loss_q = RE_q + (epoch / self.max_epoch) * beta * KL_q
                loss_p = RE_p + (epoch / self.max_epoch) * beta * KL_p
            else:
                loss_q = RE_q + beta * KL_q
                loss_p = RE_p + beta * KL_p

            if self.reg_type == 'ml_reg':
                log_std_q = logvar_q / 2
                dist = torch.distributions.Normal(mean_q, torch.exp(log_std_q))
                z_q = dist.rsample()
                z_loglike = self.gaussian_log_likelihood(z_q, mean_p, logvar_p)
                loss = loss_q - (epoch / self.max_epoch) * alpha * z_loglike
            elif self.reg_type == 'kl_reg':
                KL_reg = self.kl_diagnormal_diagnormal(mean_q, logvar_q, mean_p, logvar_p)
                loss = loss_q + alpha * (KL_reg - loss_q + loss_p + self.neg_gaussian_log_likelihood(x * mask * ~mask_p,
                                                                                                     x_recon_q * mask * ~mask_p,
                                                                                                     x_logvar_q * mask * ~mask_p))
            else:
                loss = 0
                # print_loss = loss
                print('Not implemented!')
            RE_q_imputed = 0
        train_loss = loss / x.shape[0]
        print_loss = train_loss
        if llh_eval:
            return print_loss, train_loss, RE_q / x.shape[0], RE_q_imputed / x.shape[0]
        elif MI:
            aggregated_mean = torch.mean(mean_q, 0)
            aggregated_logvar = torch.mean(logvar_q, 0)
            KL_agg = self.kl_diagnormal_stdnormal2(aggregated_mean, aggregated_logvar)
            mut_inf = KL_q / x.shape[0] - KL_agg
            return print_loss, train_loss, mut_inf, KL_q / x.shape[0]
        elif vae_elbo:
            return print_loss, train_loss
        else:
            return print_loss, train_loss

    def kl_diagnormal_diagnormal(self, mean1, log_var1, mean2, log_var2):
        # let dist1 be from reconstructed data
        # let dist2 be from the original data
        dist1 = torch.distributions.Normal(mean1, torch.exp(log_var1 / 2))
        dist2 = torch.distributions.Normal(mean2, torch.exp(log_var2 / 2))
        return torch.sum(torch.distributions.kl_divergence(dist1, dist2))

    def kl_diagnormal_stdnormal(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior))

    def kl_diagnormal_stdnormal2(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior))

    def neg_gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(-dist.log_prob(targets))

    def gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(dist.log_prob(targets))

    def forward(self, data, mask, mask_p, stage):
        data = data.reshape(-1, self.obs_dim)
        mask = mask.reshape(-1, self.obs_dim)
        mask_p = mask_p.reshape(-1, self.obs_dim)
        if stage == 'evaluate':
            z_q, mean_q, logvar_q = self.encoder(data, mask)
            x_mean_q, x_logvar_q = self.decoder(z_q)
            z_p, mean_p, logvar_p = self.encoder(data, mask_p)  # not important here
            x_mean_p, x_logvar_p = self.decoder(z_p)
        else:
            z_q, mean_q, logvar_q = self.encoder(data, mask)
            x_mean_q, x_logvar_q = self.decoder(z_q)
            z_p, mean_p, logvar_p = self.encoder(data, mask_p)
            x_mean_p, x_logvar_p = self.decoder(z_p)
        return mean_p, logvar_p, x_mean_p, x_logvar_p, mean_q, logvar_q, x_mean_q, x_logvar_q


class vanilla_EDDI_mnist(nn.Module):
    """ Reg VAE """

    def __init__(self, obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type, num_samples=1,
                 num_estimates=1):
        super(vanilla_EDDI_mnist, self).__init__()
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.emb_dim = K
        self.latent_dim = latent_dim
        self.batch_size = training_parameters['batch_size']
        self.training_parameters = training_parameters
        self.experiment_type = experiment_type
        self.num_samples = num_samples
        self.num_estimates = num_estimates
        # encoder: q(z2 | x)
        self.pnp_encoder1 = nn.Sequential(
            nn.Linear(2 + self.emb_dim, self.emb_dim),
            nn.ReLU()
        )

        self.pnp_encoder2 = nn.Sequential(
            nn.Linear(self.emb_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 2 * latent_dim)
        )
        self.seq_decoder = nn.Sequential(nn.Linear(latent_dim, 200), nn.ReLU(), nn.Linear(200, 500), nn.ReLU(),
                                         nn.Linear(500, 500), nn.ReLU(),
                                         nn.Linear(500, obs_dim),
                                         nn.Sigmoid())
        # self.x_logvar = nn.Sequential(nn.Linear(latent_dim, 50), nn.ELU(), nn.Linear(50, 100), nn.ELU(), nn.Linear(100, obs_dim),
        #                              nn.Hardtanh(min_val=-10.0, max_val=0))
        self.x_logvar = torch.log(torch.square(torch.Tensor([0.1 * np.sqrt(2)])))
        # self.type_pars1 = nn.parameter.Parameter(1e-3 * torch.randn(self.obs_dim, self.emb_dim))
        self.type_pars1 = nn.parameter.Parameter(torch.zeros(self.obs_dim, self.emb_dim), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.type_pars1)
        self.type_bias1 = nn.parameter.Parameter(torch.zeros(self.obs_dim, 1), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.type_bias1)

        # assume factorized prior of normal RVs
        self.prior_mean = torch.nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False)
        self.prior_std = torch.nn.Parameter(torch.ones(self.latent_dim), requires_grad=False)
        self.prior = torch.distributions.Normal(self.prior_mean, self.prior_std)
        self.max_epoch = 2800
        # self.sample = torch.FloatTensor(64, 14).uniform_(-10, -5)

    # TODO add mask again!
    def encoder(self, x, mask, sample=True):
        """
        Perform a point-net like transformation of the data
        """
        if mask.shape[0] == 0:
            return torch.empty(0, 10), torch.empty(0, 10), torch.empty(0, 10)
        else:
            x_flat = x.reshape(-1, 1)
            x_flat = torch.cat(
                [x_flat, x_flat * (self.type_pars1.repeat(x.shape[0], 1, 1)).reshape([x.shape[0] * self.obs_dim, -1]),
                 (self.type_bias1.repeat(x.shape[0], 1, 1)).reshape([x.shape[0] * self.obs_dim, 1])],
                dim=1)
            mask_exp = (mask.reshape(mask.shape[0], mask.shape[1], -1)).repeat(1, 1, self.emb_dim)
            agg = (mask_exp * (self.pnp_encoder1(x_flat.float())).reshape(
                [x.shape[0], self.obs_dim, self.emb_dim])).sum(1)
            mean, logvar = self.pnp_encoder2(agg).chunk(2, dim=1)
            if sample:
                log_std = logvar / 2
                dist = torch.distributions.Normal(mean, torch.exp(log_std))
                z = dist.rsample()
            else:
                z = mean
            return z, mean, logvar

    def decoder(self, z_int):
        decoded_int = self.seq_decoder(z_int)
        x_logvar = self.x_logvar
        # x_logvar = self.x_logvar(z_int)
        return decoded_int, x_logvar

    def loss(self, x, x_recon_q, x_logvar_q, mean_q, logvar_q, epoch, mask,
             vae_elbo=False, llh_eval=False, MI=False, beta_annealing=False,
             beta=1.0, alpha=0.5, stage='train'):
        x = x.reshape(-1, self.obs_dim)
        mask = mask.reshape(-1, self.obs_dim)
        x_logvar_q = torch.ones_like(x_recon_q) * x_logvar_q
        RE_q = self.neg_gaussian_log_likelihood(x * mask, x_recon_q * mask,
                                                x_logvar_q * mask)  # TODO add new_mask again

        RE_q_imputed = self.neg_gaussian_log_likelihood(x * (1 - mask * 1.0), x_recon_q * (1 - mask * 1.0),
                                                        x_logvar_q * (1 - mask * 1.0))

        KL_q = self.kl_diagnormal_stdnormal(mean_q, logvar_q)

        if beta_annealing:
            loss_q = RE_q + (epoch / self.max_epoch) * beta * KL_q
        else:
            loss_q = RE_q + beta * KL_q
        loss = loss_q
        train_loss = loss / x.shape[0]
        print_loss = train_loss
        if llh_eval:
            return print_loss, train_loss, RE_q / x.shape[0], RE_q_imputed / x.shape[0]
        elif MI:
            aggregated_mean = torch.mean(mean_q, 0)
            aggregated_logvar = torch.mean(logvar_q, 0)
            KL_agg = self.kl_diagnormal_stdnormal2(aggregated_mean, aggregated_logvar)
            mut_inf = KL_q / x.shape[0] - KL_agg
            return print_loss, train_loss, mut_inf, KL_q / x.shape[0]
        elif vae_elbo:
            return print_loss, train_loss
        else:
            return print_loss, train_loss

    def kl_diagnormal_diagnormal(self, mean1, log_var1, mean2, log_var2):
        # let dist1 be from reconstructed data
        # let dist2 be from the original data
        dist1 = torch.distributions.Normal(mean1, torch.exp(log_var1 / 2))
        dist2 = torch.distributions.Normal(mean2, torch.exp(log_var2 / 2))
        return torch.sum(torch.distributions.kl_divergence(dist1, dist2))

    def kl_diagnormal_stdnormal(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior))

    def kl_diagnormal_stdnormal2(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior))

    def neg_gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(-dist.log_prob(targets))

    def gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(dist.log_prob(targets))

    def forward(self, data, mask):
        data = data.reshape(-1, self.obs_dim)
        mask = mask.reshape(-1, self.obs_dim)
        z_q, mean_q, logvar_q = self.encoder(data, mask)
        x_mean_q, x_logvar_q = self.decoder(z_q)
        return mean_q, logvar_q, x_mean_q, x_logvar_q


class Reg_VAE(nn.Module):
    """ Reg VAE """

    def __init__(self, obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type, reg_type, num_samples=1,
                 num_estimates=1):
        super(Reg_VAE, self).__init__()
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.latent_dim = latent_dim
        self.K = K
        self.num_samples = num_samples
        self.num_estimates = num_estimates
        self.training_parameters = training_parameters
        self.experiment_type = experiment_type
        self.reg_type = reg_type
        # encoder: q(z2 | x)
        self.seq_encoder = nn.Sequential(
            nn.Linear(obs_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 2 * latent_dim)
        )

        self.seq_decoder = nn.Sequential(nn.Linear(latent_dim, 50), nn.ReLU(), nn.Linear(50, 100), nn.ReLU(),
                                         nn.Linear(100, obs_dim),
                                         nn.Sigmoid())
        # self.x_logvar = nn.Sequential(nn.Linear(latent_dim, hid_dim), nn.ELU(), nn.Linear(hid_dim, obs_dim),
        #                              nn.Hardtanh(min_val=-10.0, max_val=0))
        self.x_logvar = torch.log(torch.square(torch.Tensor([0.1 * np.sqrt(2)])))

        self.prior_mean = torch.nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False)
        self.prior_std = torch.nn.Parameter(torch.ones(self.latent_dim), requires_grad=False)
        self.prior = torch.distributions.Normal(self.prior_mean, self.prior_std)
        self.max_epoch = 2800

    # TODO add mask again!
    def encoder(self, x, mask, sample=True):
        mean, logvar = self.seq_encoder(x.float() * mask).chunk(2, dim=1)
        if sample:
            log_std = logvar / 2
            dist = torch.distributions.Normal(mean, torch.exp(log_std))
            z = dist.rsample()
        else:
            z = mean
        return z, mean, logvar

    def decoder(self, z_int):
        decoded_int = self.seq_decoder(z_int)
        x_logvar = self.x_logvar
        # x_logvar = self.x_logvar(z_int)
        return decoded_int, x_logvar

    def loss(self, x, x_recon_p, x_logvar_p, mean_p, logvar_p, x_recon_q, x_logvar_q, mean_q, logvar_q, mask, mask_p,
             epoch,
             vae_elbo=False, llh_eval=False, MI=False, beta_annealing=False,
             beta=1.0, alpha=0.8, stage='train', alpha_annealing=True):

        x_logvar_q = torch.ones_like(x_recon_q) * x_logvar_q
        x_logvar_p = torch.ones_like(x_recon_p) * x_logvar_p
        if stage == 'evaluate':
            RE_q = self.neg_gaussian_log_likelihood(x * mask, x_recon_q * mask,
                                                    x_logvar_q * mask)
            RE_q_imputed = self.neg_gaussian_log_likelihood(x * ~mask, x_recon_q * ~mask,
                                                            x_logvar_q * ~mask)
            KL_q = self.kl_diagnormal_stdnormal(mean_q, logvar_q)
            if beta_annealing:
                loss_q = RE_q + (epoch / self.max_epoch) * beta * KL_q
            else:
                loss_q = RE_q + beta * KL_q
            loss = loss_q
        else:
            RE_q = self.neg_gaussian_log_likelihood(x * mask, x_recon_q * mask,
                                                    x_logvar_q * mask)
            RE_p = self.neg_gaussian_log_likelihood(x * mask_p,
                                                    x_recon_p * mask_p,
                                                    x_logvar_p * mask_p)
            KL_q = self.kl_diagnormal_stdnormal(mean_q, logvar_q)
            KL_p = self.kl_diagnormal_stdnormal(mean_p, logvar_p)
            if beta_annealing:
                loss_q = RE_q + (epoch / self.max_epoch) * beta * KL_q
                loss_p = RE_p + (epoch / self.max_epoch) * beta * KL_p
            else:
                loss_q = RE_q + beta * KL_q
                loss_p = RE_p + beta * KL_p
            if self.reg_type == 'ml_reg':
                log_std_q = logvar_q / 2
                dist = torch.distributions.Normal(mean_q, torch.exp(log_std_q))
                z_q = dist.rsample()
                z_loglike = self.gaussian_log_likelihood(z_q, mean_p, logvar_p)
                loss = loss_q - (epoch / self.max_epoch) * alpha * z_loglike
            elif self.reg_type == 'kl_reg':
                KL_reg = self.kl_diagnormal_diagnormal(mean_q, logvar_q, mean_p, logvar_p)
                loss = loss_q + alpha * (
                        KL_reg - loss_q + loss_p + self.neg_gaussian_log_likelihood(x * mask * ~mask_p,
                                                                                    x_recon_q * mask * ~mask_p,
                                                                                    x_logvar_q * mask * ~mask_p))
            else:
                loss = 0
                print('Not implemented!')
            RE_q_imputed = 0

        train_loss = loss / x.shape[0]
        print_loss = train_loss

        if llh_eval:
            return print_loss, train_loss, RE_q / x.shape[0], RE_q_imputed / x.shape[0]
        elif MI:
            aggregated_mean = torch.mean(mean_q, 0)
            aggregated_logvar = torch.mean(logvar_q, 0)
            KL_agg = self.kl_diagnormal_stdnormal2(aggregated_mean, aggregated_logvar)
            mut_inf = KL_q / x.shape[0] - KL_agg
            return print_loss, train_loss, mut_inf, KL_q / x.shape[0]

        elif vae_elbo:
            return print_loss, train_loss
        else:
            return print_loss, train_loss

    def kl_diagnormal_diagnormal(self, mean1, log_var1, mean2, log_var2):
        # let dist1 be from reconstructed data
        # let dist2 be from the original data
        dist1 = torch.distributions.Normal(mean1, torch.exp(log_var1 / 2))  # distribution Q
        dist2 = torch.distributions.Normal(mean2, torch.exp(log_var2 / 2))  # distribution P
        return torch.sum(torch.distributions.kl_divergence(dist1, dist2))

    def kl_diagnormal_stdnormal(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior))

    def kl_diagnormal_stdnormal_x(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior_x))

    def kl_diagnormal_stdnormal2(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior))

    def neg_gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(-dist.log_prob(targets))

    def gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(dist.log_prob(targets))

    def forward(self, data, mask, mask_p, stage):
        if stage == 'evaluate':
            z_q, mean_q, logvar_q = self.encoder(data, mask)
            x_mean_q, x_logvar_q = self.decoder(z_q)
            z_p, mean_p, logvar_p = self.encoder(data, mask_p)  # not important here
            x_mean_p, x_logvar_p = self.decoder(z_p)
        else:
            z_q, mean_q, logvar_q = self.encoder(data, mask)
            x_mean_q, x_logvar_q = self.decoder(z_q)
            z_p, mean_p, logvar_p = self.encoder(data, mask_p)
            x_mean_p, x_logvar_p = self.decoder(z_p)
        return mean_p, logvar_p, x_mean_p, x_logvar_p, mean_q, logvar_q, x_mean_q, x_logvar_q


class Reg_VAE_mask(nn.Module):
    """ Reg VAE """

    def __init__(self, obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type, reg_type, num_samples=1,
                 num_estimates=1):
        super(Reg_VAE_mask, self).__init__()
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.latent_dim = latent_dim
        self.K = K
        self.num_samples = num_samples
        self.num_estimates = num_estimates
        self.training_parameters = training_parameters
        self.experiment_type = experiment_type
        self.reg_type = reg_type
        # encoder: q(z2 | x)
        self.seq_encoder = nn.Sequential(
            nn.Linear(2 * obs_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 2 * latent_dim)
        )

        self.seq_decoder = nn.Sequential(nn.Linear(latent_dim, 50), nn.ReLU(), nn.Linear(50, 100), nn.ReLU(),
                                         nn.Linear(100, obs_dim),
                                         nn.Sigmoid())
        self.x_logvar = torch.log(torch.square(torch.Tensor([0.1 * np.sqrt(2)])))

        # assume factorized prior of normal RVs
        self.prior_mean = torch.nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False)
        self.prior_std = torch.nn.Parameter(torch.ones(self.latent_dim), requires_grad=False)
        self.prior = torch.distributions.Normal(self.prior_mean, self.prior_std)
        self.max_epoch = 2800

    def encoder(self, x, mask, sample=True):
        # print(self.sample)
        mean, logvar = self.seq_encoder(torch.stack([x.float() * mask, mask], 1).reshape(-1, self.obs_dim * 2)).chunk(2,
                                                                                                                      dim=1)
        if sample:
            log_std = logvar / 2
            dist = torch.distributions.Normal(mean, torch.exp(log_std))
            z = dist.rsample()
        else:
            z = mean
        return z, mean, logvar

    def decoder(self, z_int):
        decoded_int = self.seq_decoder(z_int)
        # x_logvar = self.x_logvar(z_int)
        x_logvar = self.x_logvar
        return decoded_int, x_logvar

    def loss(self, x, x_recon_p, x_logvar_p, mean_p, logvar_p, x_recon_q, x_logvar_q, mean_q, logvar_q, mask, mask_p,
             epoch,
             vae_elbo=False, llh_eval=False, MI=False, beta_annealing=False,
             beta=1.0, alpha=0.8, alpha_annealing=True, stage='train'):

        x_logvar_q = torch.ones_like(x_recon_q) * x_logvar_q
        x_logvar_p = torch.ones_like(x_recon_p) * x_logvar_p
        if stage == 'evaluate':
            RE_q = self.neg_gaussian_log_likelihood(x * mask, x_recon_q * mask,
                                                    x_logvar_q * mask)
            RE_q_imputed = self.neg_gaussian_log_likelihood(x * ~mask, x_recon_q * ~mask,
                                                            x_logvar_q * ~mask)
            KL_q = self.kl_diagnormal_stdnormal(mean_q, logvar_q)
            if beta_annealing:
                loss_q = RE_q + (epoch / self.max_epoch) * beta * KL_q
            else:
                loss_q = RE_q + beta * KL_q
            loss = loss_q
        else:
            RE_q = self.neg_gaussian_log_likelihood(x * mask, x_recon_q * mask,
                                                    x_logvar_q * mask)
            RE_p = self.neg_gaussian_log_likelihood(x * mask_p,
                                                    x_recon_p * mask_p,
                                                    x_logvar_p * mask_p)
            KL_q = self.kl_diagnormal_stdnormal(mean_q, logvar_q)
            KL_p = self.kl_diagnormal_stdnormal(mean_p, logvar_p)
            if beta_annealing:
                loss_q = RE_q + (epoch / self.max_epoch) * beta * KL_q
                loss_p = RE_p + (epoch / self.max_epoch) * beta * KL_p
            else:
                loss_q = RE_q + beta * KL_q
                loss_p = RE_p + beta * KL_p
            if self.reg_type == 'ml_reg':
                log_std_q = logvar_q / 2
                dist = torch.distributions.Normal(mean_q, torch.exp(log_std_q))
                z_q = dist.rsample()
                z_loglike = self.gaussian_log_likelihood(z_q, mean_p, logvar_p)
                loss = loss_q - (epoch / self.max_epoch) * alpha * z_loglike
            elif self.reg_type == 'kl_reg':
                KL_reg = self.kl_diagnormal_diagnormal(mean_q, logvar_q, mean_p, logvar_p)
                loss = loss_q + alpha * (
                        KL_reg - loss_q + loss_p + self.neg_gaussian_log_likelihood(x * mask * ~mask_p,
                                                                                    x_recon_q * mask * ~mask_p,
                                                                                    x_logvar_q * mask * ~mask_p))
            else:
                loss = 0
                print('Not implemented!')
            RE_q_imputed = 0

        train_loss = loss / x.shape[0]
        print_loss = train_loss

        if llh_eval:
            return print_loss, train_loss, RE_q / x.shape[0], RE_q_imputed / x.shape[0]
        elif MI:
            aggregated_mean = torch.mean(mean_q, 0)
            aggregated_logvar = torch.mean(logvar_q, 0)
            KL_agg = self.kl_diagnormal_stdnormal2(aggregated_mean, aggregated_logvar)
            mut_inf = KL_q / x.shape[0] - KL_agg
            return print_loss, train_loss, mut_inf, KL_q / x.shape[0]

        elif vae_elbo:
            return print_loss, train_loss
        else:
            return print_loss, train_loss

    def kl_diagnormal_diagnormal(self, mean1, log_var1, mean2, log_var2):
        # let dist1 be from reconstructed data
        # let dist2 be from the original data
        dist1 = torch.distributions.Normal(mean1, torch.exp(log_var1 / 2))  # distribution Q
        dist2 = torch.distributions.Normal(mean2, torch.exp(log_var2 / 2))  # distribution P
        return torch.sum(torch.distributions.kl_divergence(dist1, dist2))

    def kl_diagnormal_stdnormal(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior))

    def kl_diagnormal_stdnormal_x(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior_x))

    def kl_diagnormal_stdnormal2(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior))

    def neg_gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(-dist.log_prob(targets))

    def gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(dist.log_prob(targets))

    def forward(self, data, mask, mask_p, stage):
        if stage == 'evaluate':
            z_q, mean_q, logvar_q = self.encoder(data, mask)
            x_mean_q, x_logvar_q = self.decoder(z_q)
            z_p, mean_p, logvar_p = self.encoder(data, mask_p)  # not important here
            x_mean_p, x_logvar_p = self.decoder(z_p)
        else:
            z_q, mean_q, logvar_q = self.encoder(data, mask)
            x_mean_q, x_logvar_q = self.decoder(z_q)
            z_p, mean_p, logvar_p = self.encoder(data, mask_p)
            x_mean_p, x_logvar_p = self.decoder(z_p)
        return mean_p, logvar_p, x_mean_p, x_logvar_p, mean_q, logvar_q, x_mean_q, x_logvar_q


class Reg_EDDI(nn.Module):
    """ Reg VAE """

    def __init__(self, obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type, reg_type, num_samples=1,
                 num_estimates=1):
        super(Reg_EDDI, self).__init__()
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.emb_dim = K
        self.latent_dim = latent_dim
        self.batch_size = training_parameters['batch_size']
        self.training_parameters = training_parameters
        self.experiment_type = experiment_type
        self.num_samples = num_samples
        self.num_estimates = num_estimates
        # encoder: q(z2 | x)
        self.reg_type = reg_type
        self.pnp_encoder1 = nn.Sequential(
            nn.Linear(2 + self.emb_dim, self.emb_dim),
            nn.ReLU()
        )

        self.pnp_encoder2 = nn.Sequential(
            nn.Linear(self.emb_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 2 * latent_dim)
        )
        self.seq_decoder = nn.Sequential(nn.Linear(latent_dim, 50), nn.ReLU(), nn.Linear(50, 100), nn.ReLU(),
                                         nn.Linear(100, obs_dim),
                                         nn.Sigmoid())
        # self.x_logvar = nn.Sequential(nn.Linear(latent_dim, 50), nn.ReLU(), nn.Linear(50, 100), nn.ReLU(), nn.Linear(100, obs_dim),
        #                              nn.Hardtanh(min_val=-4.5, max_val=0))
        self.x_logvar = torch.log(torch.square(torch.Tensor([0.1 * np.sqrt(2)])))
        # self.type_pars1 = nn.parameter.Parameter(1e-3 * torch.randn(self.obs_dim, self.emb_dim))
        self.type_pars1 = nn.parameter.Parameter(torch.zeros(self.obs_dim, self.emb_dim), requires_grad=True)
        torch.nn.init.xavier_uniform(self.type_pars1)
        self.type_bias1 = nn.parameter.Parameter(torch.zeros(self.obs_dim, 1), requires_grad=True)
        torch.nn.init.xavier_uniform(self.type_bias1)

        # assume factorized prior of normal RVs
        self.prior_mean = torch.nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False)
        self.prior_std = torch.nn.Parameter(torch.ones(self.latent_dim), requires_grad=False)
        self.prior = torch.distributions.Normal(self.prior_mean, self.prior_std)
        self.max_epoch = 2800
        # self.sample = torch.FloatTensor(64, 14).uniform_(-10, -5)

    # TODO add mask again!
    def encoder(self, x, mask, sample=True):
        """
        Perform a point-net like transformation of the data
        """
        if mask.shape[0] == 0:
            return torch.empty(0, 10), torch.empty(0, 10), torch.empty(0, 10)
        else:
            x_flat = x.reshape(-1, 1)
            x_flat = torch.cat(
                [x_flat, x_flat * (self.type_pars1.repeat(x.shape[0], 1, 1)).reshape([x.shape[0] * self.obs_dim, -1]),
                 (self.type_bias1.repeat(x.shape[0], 1, 1)).reshape([x.shape[0] * self.obs_dim, 1])],
                dim=1)
            mask_exp = (mask.reshape(mask.shape[0], mask.shape[1], -1)).repeat(1, 1, self.emb_dim)
            agg = (mask_exp * (self.pnp_encoder1(x_flat.float())).reshape(
                [x.shape[0], self.obs_dim, self.emb_dim])).sum(1)
            mean, logvar = self.pnp_encoder2(agg).chunk(2, dim=1)
            if sample:
                log_std = logvar / 2
                dist = torch.distributions.Normal(mean, torch.exp(log_std))
                z = dist.rsample()
            else:
                z = mean
            return z, mean, logvar

    def decoder(self, z_int):
        decoded_int = self.seq_decoder(z_int)
        x_logvar = self.x_logvar
        # x_logvar = self.x_logvar(z_int)
        return decoded_int, x_logvar

    def loss(self, x, x_recon_p, x_logvar_p, mean_p, logvar_p, x_recon_q, x_logvar_q, mean_q, logvar_q, mask, mask_p,
             epoch,
             vae_elbo=False, llh_eval=False, MI=False, beta_annealing=False,
             beta=1.0, alpha=0.5, stage='train', alpha_annealing=False):

        x_logvar_q = torch.ones_like(x_recon_q) * x_logvar_q
        x_logvar_p = torch.ones_like(x_recon_p) * x_logvar_p

        if stage == 'evaluate':
            RE_q = self.neg_gaussian_log_likelihood(x * mask, x_recon_q * mask,
                                                    x_logvar_q * mask)  # TODO add new_mask again
            RE_q_imputed = self.neg_gaussian_log_likelihood(x * ~mask, x_recon_q * ~mask,
                                                            x_logvar_q * ~mask)

            KL_q = self.kl_diagnormal_stdnormal(mean_q, logvar_q)

            if beta_annealing:
                loss_q = RE_q + (epoch / self.max_epoch) * beta * KL_q
            else:
                loss_q = RE_q + beta * KL_q
            loss = loss_q
        else:
            RE_q = self.neg_gaussian_log_likelihood(x * mask, x_recon_q * mask,
                                                    x_logvar_q * mask)  # TODO add new_mask again
            RE_p = self.neg_gaussian_log_likelihood(x * mask_p,
                                                    x_recon_p * mask_p,
                                                    x_logvar_p * mask_p)

            KL_q = self.kl_diagnormal_stdnormal(mean_q, logvar_q)
            KL_p = self.kl_diagnormal_stdnormal(mean_p, logvar_p)

            if beta_annealing:
                loss_q = RE_q + (epoch / self.max_epoch) * beta * KL_q
                loss_p = RE_p + (epoch / self.max_epoch) * beta * KL_p
            else:
                loss_q = RE_q + beta * KL_q
                loss_p = RE_p + beta * KL_p

            if self.reg_type == 'ml_reg':
                log_std_q = logvar_q / 2
                dist = torch.distributions.Normal(mean_q, torch.exp(log_std_q))
                z_q = dist.rsample()
                z_loglike = self.gaussian_log_likelihood(z_q, mean_p, logvar_p)
                loss = loss_q - (epoch / self.max_epoch) * alpha * z_loglike
            elif self.reg_type == 'kl_reg':
                KL_reg = self.kl_diagnormal_diagnormal(mean_q, logvar_q, mean_p, logvar_p)
                loss = loss_q + alpha * (KL_reg - loss_q + loss_p + self.neg_gaussian_log_likelihood(x * mask * ~mask_p,
                                                                                                     x_recon_q * mask * ~mask_p,
                                                                                                     x_logvar_q * mask * ~mask_p))
            else:
                loss = 0
                # print_loss = loss
                print('Not implemented!')
            RE_q_imputed = 0

        train_loss = loss / x.shape[0]
        print_loss = train_loss
        if llh_eval:
            return print_loss, train_loss, RE_q / x.shape[0], RE_q_imputed / x.shape[0]
        elif MI:
            aggregated_mean = torch.mean(mean_q, 0)
            aggregated_logvar = torch.mean(logvar_q, 0)
            KL_agg = self.kl_diagnormal_stdnormal2(aggregated_mean, aggregated_logvar)
            mut_inf = KL_q / x.shape[0] - KL_agg
            return print_loss, train_loss, mut_inf, KL_q / x.shape[0]
        elif vae_elbo:
            return print_loss, train_loss
        else:
            return print_loss, train_loss

    def kl_diagnormal_diagnormal(self, mean1, log_var1, mean2, log_var2):
        # let dist1 be from reconstructed data
        # let dist2 be from the original data
        dist1 = torch.distributions.Normal(mean1, torch.exp(log_var1 / 2))
        dist2 = torch.distributions.Normal(mean2, torch.exp(log_var2 / 2))
        return torch.sum(torch.distributions.kl_divergence(dist1, dist2))

    def kl_diagnormal_stdnormal(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior))

    def kl_diagnormal_stdnormal2(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior))

    def neg_gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(-dist.log_prob(targets))

    def gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(dist.log_prob(targets))

    def forward(self, data, mask, mask_p, stage):
        if stage == 'evaluate':
            z_q, mean_q, logvar_q = self.encoder(data, mask)
            x_mean_q, x_logvar_q = self.decoder(z_q)
            z_p, mean_p, logvar_p = self.encoder(data, mask_p)  # not important here
            x_mean_p, x_logvar_p = self.decoder(z_p)
        else:
            z_q, mean_q, logvar_q = self.encoder(data, mask)
            x_mean_q, x_logvar_q = self.decoder(z_q)
            z_p, mean_p, logvar_p = self.encoder(data, mask_p)
            x_mean_p, x_logvar_p = self.decoder(z_p)
        return mean_p, logvar_p, x_mean_p, x_logvar_p, mean_q, logvar_q, x_mean_q, x_logvar_q


class vanilla_EDDI(nn.Module):
    """ Reg VAE """

    def __init__(self, obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type, num_samples=1,
                 num_estimates=1):
        super(vanilla_EDDI, self).__init__()
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.emb_dim = K
        self.latent_dim = latent_dim
        self.batch_size = training_parameters['batch_size']
        self.training_parameters = training_parameters
        self.experiment_type = experiment_type
        self.num_samples = num_samples
        self.num_estimates = num_estimates
        # encoder: q(z2 | x)
        self.pnp_encoder1 = nn.Sequential(
            nn.Linear(2 + self.emb_dim, self.emb_dim),
            nn.ReLU()
        )

        self.pnp_encoder2 = nn.Sequential(
            nn.Linear(self.emb_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 2 * latent_dim)
        )
        self.seq_decoder = nn.Sequential(nn.Linear(latent_dim, 50), nn.ReLU(), nn.Linear(50, 100), nn.ReLU(),
                                         nn.Linear(100, obs_dim),
                                         nn.Sigmoid())
        # self.x_logvar = nn.Sequential(nn.Linear(latent_dim, 50), nn.ELU(), nn.Linear(50, 100), nn.ELU(), nn.Linear(100, obs_dim),
        #                              nn.Hardtanh(min_val=-10.0, max_val=0))
        self.x_logvar = torch.log(torch.square(torch.Tensor([0.1 * np.sqrt(2)])))
        # self.type_pars1 = nn.parameter.Parameter(1e-3 * torch.randn(self.obs_dim, self.emb_dim))
        self.type_pars1 = nn.parameter.Parameter(torch.zeros(self.obs_dim, self.emb_dim), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.type_pars1)
        self.type_bias1 = nn.parameter.Parameter(torch.zeros(self.obs_dim, 1), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.type_bias1)

        # assume factorized prior of normal RVs
        self.prior_mean = torch.nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False)
        self.prior_std = torch.nn.Parameter(torch.ones(self.latent_dim), requires_grad=False)
        self.prior = torch.distributions.Normal(self.prior_mean, self.prior_std)
        self.max_epoch = 2800

    # TODO add mask again!
    def encoder(self, x, mask, sample=True):
        """
        Perform a point-net like transformation of the data
        """
        if mask.shape[0] == 0:
            return torch.empty(0, 10), torch.empty(0, 10), torch.empty(0, 10)
        else:
            x_flat = x.reshape(-1, 1)
            x_flat = torch.cat(
                [x_flat, x_flat * (self.type_pars1.repeat(x.shape[0], 1, 1)).reshape([x.shape[0] * self.obs_dim, -1]),
                 (self.type_bias1.repeat(x.shape[0], 1, 1)).reshape([x.shape[0] * self.obs_dim, 1])],
                dim=1)
            mask_exp = (mask.reshape(mask.shape[0], mask.shape[1], -1)).repeat(1, 1, self.emb_dim)
            agg = (mask_exp * (self.pnp_encoder1(x_flat.float())).reshape(
                [x.shape[0], self.obs_dim, self.emb_dim])).sum(1)
            mean, logvar = self.pnp_encoder2(agg).chunk(2, dim=1)
            if sample:
                log_std = logvar / 2
                dist = torch.distributions.Normal(mean, torch.exp(log_std))
                z = dist.rsample()
            else:
                z = mean
            return z, mean, logvar

    def decoder(self, z_int):
        decoded_int = self.seq_decoder(z_int)
        x_logvar = self.x_logvar
        # x_logvar = self.x_logvar(z_int)
        return decoded_int, x_logvar

    def loss(self, x, x_recon_q, x_logvar_q, mean_q, logvar_q, epoch, mask,
             vae_elbo=False, llh_eval=False, MI=False, beta_annealing=False,
             beta=1.0, alpha=0.5, stage='train'):

        x_logvar_q = torch.ones_like(x_recon_q) * x_logvar_q
        RE_q = self.neg_gaussian_log_likelihood(x * mask, x_recon_q * mask,
                                                x_logvar_q * mask)  # TODO add new_mask again

        RE_q_imputed = self.neg_gaussian_log_likelihood(x * (1 - mask * 1.0), x_recon_q * (1 - mask * 1.0),
                                                        x_logvar_q * (1 - mask * 1.0))

        KL_q = self.kl_diagnormal_stdnormal(mean_q, logvar_q)

        if beta_annealing:
            loss_q = RE_q + (epoch / self.max_epoch) * beta * KL_q
        else:
            loss_q = RE_q + beta * KL_q
        loss = loss_q
        train_loss = loss / x.shape[0]
        print_loss = train_loss
        if llh_eval:
            return print_loss, train_loss, RE_q / x.shape[0], RE_q_imputed / x.shape[0]
        elif MI:
            aggregated_mean = torch.mean(mean_q, 0)
            aggregated_logvar = torch.mean(logvar_q, 0)
            KL_agg = self.kl_diagnormal_stdnormal2(aggregated_mean, aggregated_logvar)
            mut_inf = KL_q / x.shape[0] - KL_agg
            return print_loss, train_loss, mut_inf, KL_q / x.shape[0]
        elif vae_elbo:
            return print_loss, train_loss
        else:
            return print_loss, train_loss

    def kl_diagnormal_diagnormal(self, mean1, log_var1, mean2, log_var2):
        # let dist1 be from reconstructed data
        # let dist2 be from the original data
        dist1 = torch.distributions.Normal(mean1, torch.exp(log_var1 / 2))
        dist2 = torch.distributions.Normal(mean2, torch.exp(log_var2 / 2))
        return torch.sum(torch.distributions.kl_divergence(dist1, dist2))

    def kl_diagnormal_stdnormal(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior))

    def kl_diagnormal_stdnormal2(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior))

    def neg_gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(-dist.log_prob(targets))

    def gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(dist.log_prob(targets))

    def forward(self, data, mask):
        z_q, mean_q, logvar_q = self.encoder(data, mask)
        x_mean_q, x_logvar_q = self.decoder(z_q)
        return mean_q, logvar_q, x_mean_q, x_logvar_q


class vanilla_VAE_mask(nn.Module):
    """ vanilla_VAE """

    def __init__(self, obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type, num_samples=1,
                 num_estimates=1):
        super(vanilla_VAE_mask, self).__init__()
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.latent_dim = latent_dim
        self.K = K
        self.num_samples = num_samples
        self.num_estimates = num_estimates
        self.training_parameters = training_parameters
        self.experiment_type = experiment_type
        # encoder: q(z2 | x)
        self.seq_encoder = nn.Sequential(
            nn.Linear(2 * obs_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 2 * latent_dim)
        )

        self.seq_decoder = nn.Sequential(nn.Linear(latent_dim, 50), nn.ReLU(), nn.Linear(50, 100), nn.ReLU(),
                                         nn.Linear(100, obs_dim),
                                         nn.Sigmoid())
        # self.x_logvar = nn.Sequential(nn.Linear(latent_dim, hid_dim), nn.ELU(), nn.Linear(hid_dim, obs_dim),
        #                              nn.Hardtanh(min_val=-10.0, max_val=0))
        self.x_logvar = torch.log(torch.square(torch.Tensor([0.1 * np.sqrt(2)])))
        # assume factorized prior of normal RVs
        self.prior_mean = torch.nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False)
        self.prior_std = torch.nn.Parameter(torch.ones(self.latent_dim), requires_grad=False)
        self.prior = torch.distributions.Normal(self.prior_mean, self.prior_std)
        self.max_epoch = 2800

    # TODO add mask again!
    def encoder(self, x, mask, sample=True):
        mean, logvar = self.seq_encoder(torch.stack([x.float() * mask, mask], 1).reshape(-1, self.obs_dim * 2)).chunk(2,
                                                                                                                      dim=1)
        if sample:
            log_std = logvar / 2
            dist = torch.distributions.Normal(mean, torch.exp(log_std))
            z = dist.rsample()
        else:
            z = mean
        # print(z.shape)
        return z, mean, logvar

    def decoder(self, z_int):
        decoded_int = self.seq_decoder(z_int)
        x_logvar = self.x_logvar
        # x_logvar = self.x_logvar(z_int)
        return decoded_int, x_logvar

    def loss(self, x, x_recon_q, x_logvar_q, mean_q, logvar_q, epoch, mask,
             vae_elbo=False, llh_eval=False, MI=False, beta_annealing=False,
             beta=1.0, alpha=0.8, alpha_annealing=True, stage='train'):

        x_logvar_q = torch.ones_like(x_recon_q) * x_logvar_q
        if stage == 'evaluate':
            RE_q = self.neg_gaussian_log_likelihood(x * mask, x_recon_q * mask,
                                                    x_logvar_q * mask)
            RE_q_imputed = self.neg_gaussian_log_likelihood(x * (1 - mask * 1.0), x_recon_q * (1 - mask * 1.0),
                                                            x_logvar_q * (1 - mask * 1.0))
        else:
            RE_q = self.neg_gaussian_log_likelihood(x * mask, x_recon_q * mask,
                                                    x_logvar_q * mask)
            RE_q_imputed = 0

        KL_q = self.kl_diagnormal_stdnormal(mean_q, logvar_q)

        if beta_annealing:
            loss_q = RE_q + (epoch / self.max_epoch) * beta * KL_q
        else:
            loss_q = RE_q + beta * KL_q
        loss = loss_q
        train_loss = loss / x.shape[0]
        print_loss = train_loss
        if llh_eval:
            return print_loss, train_loss, RE_q / x.shape[0], RE_q_imputed / x.shape[0]
        elif MI:
            aggregated_mean = torch.mean(mean_q, 0)
            aggregated_logvar = torch.mean(logvar_q, 0)
            KL_agg = self.kl_diagnormal_stdnormal2(aggregated_mean, aggregated_logvar)
            mut_inf = KL_q / x.shape[0] - KL_agg
            return print_loss, train_loss, mut_inf, KL_q / x.shape[0]
        elif vae_elbo:
            return print_loss, train_loss
        else:
            return print_loss, train_loss

    def kl_diagnormal_diagnormal(self, mean1, log_var1, mean2, log_var2):
        # let dist1 be from reconstructed data
        # let dist2 be from the original data
        dist1 = torch.distributions.Normal(mean1, torch.exp(log_var1 / 2))  # distribution Q
        dist2 = torch.distributions.Normal(mean2, torch.exp(log_var2 / 2))  # distribution P
        return torch.sum(torch.distributions.kl_divergence(dist1, dist2))

    def kl_diagnormal_stdnormal(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior))

    def kl_diagnormal_stdnormal_x(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior_x))

    def kl_diagnormal_stdnormal2(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior))

    def neg_gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(-dist.log_prob(targets))

    def gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(dist.log_prob(targets))

    def forward(self, data, mask):
        z_q, mean_q, logvar_q = self.encoder(data, mask)
        x_mean_q, x_logvar_q = self.decoder(z_q)
        return mean_q, logvar_q, x_mean_q, x_logvar_q


class vanilla_VAE(nn.Module):
    """ vanilla_VAE """

    def __init__(self, obs_dim, hid_dim, K, latent_dim, training_parameters, experiment_type, num_samples=1,
                 num_estimates=1):
        super(vanilla_VAE, self).__init__()
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.latent_dim = latent_dim
        self.K = K
        self.num_samples = num_samples
        self.num_estimates = num_estimates
        self.training_parameters = training_parameters
        self.experiment_type = experiment_type
        # encoder: q(z2 | x)
        self.seq_encoder = nn.Sequential(
            nn.Linear(obs_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 2 * latent_dim)
        )

        self.seq_decoder = nn.Sequential(nn.Linear(latent_dim, 50), nn.ReLU(), nn.Linear(50, 100), nn.ReLU(),
                                         nn.Linear(100, obs_dim),
                                         nn.Sigmoid())
        # self.x_logvar = nn.Sequential(nn.Linear(latent_dim, hid_dim), nn.ELU(), nn.Linear(hid_dim, obs_dim),
        #                              nn.Hardtanh(min_val=-10.0, max_val=0))
        self.x_logvar = torch.log(torch.square(torch.Tensor([0.1 * np.sqrt(2)])))
        # assume factorized prior of normal RVs
        self.prior_mean = torch.nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False)
        self.prior_std = torch.nn.Parameter(torch.ones(self.latent_dim), requires_grad=False)
        self.prior = torch.distributions.Normal(self.prior_mean, self.prior_std)
        self.max_epoch = 2800

    # TODO add mask again!
    def encoder(self, x, mask, sample=True):
        mean, logvar = self.seq_encoder(x.float() * mask).chunk(2, dim=1)
        if sample:
            log_std = logvar / 2
            dist = torch.distributions.Normal(mean, torch.exp(log_std))
            z = dist.rsample()
        else:
            z = mean
        return z, mean, logvar

    def decoder(self, z_int):
        decoded_int = self.seq_decoder(z_int)
        x_logvar = self.x_logvar
        # x_logvar = self.x_logvar(z_int)
        return decoded_int, x_logvar

    def loss(self, x, x_recon_q, x_logvar_q, mean_q, logvar_q, epoch, mask,
             vae_elbo=False, llh_eval=False, MI=False, beta_annealing=False,
             beta=1.0, alpha=0.8, alpha_annealing=True, stage='train'):

        x_logvar_q = torch.ones_like(x_recon_q) * x_logvar_q
        if stage == 'evaluate':
            RE_q = self.neg_gaussian_log_likelihood(x * mask, x_recon_q * mask,
                                                    x_logvar_q * mask)
            RE_q_imputed = self.neg_gaussian_log_likelihood(x * (1 - mask * 1.0), x_recon_q * (1 - mask * 1.0),
                                                            x_logvar_q * (1 - mask * 1.0))

        else:
            RE_q = self.neg_gaussian_log_likelihood(x * mask, x_recon_q * mask,
                                                    x_logvar_q * mask)
            RE_q_imputed = 0

        KL_q = self.kl_diagnormal_stdnormal(mean_q, logvar_q)

        if beta_annealing:
            loss_q = RE_q + (epoch / self.max_epoch) * beta * KL_q
        else:
            loss_q = RE_q + beta * KL_q

        loss = loss_q
        train_loss = loss / x.shape[0]
        print_loss = train_loss
        if llh_eval:
            return print_loss, train_loss, RE_q / x.shape[0], RE_q_imputed / x.shape[0]
        elif MI:
            aggregated_mean = torch.mean(mean_q, 0)
            aggregated_logvar = torch.mean(logvar_q, 0)
            KL_agg = self.kl_diagnormal_stdnormal2(aggregated_mean, aggregated_logvar)
            mut_inf = KL_q / x.shape[0] - KL_agg
            return print_loss, train_loss, mut_inf, KL_q / x.shape[0]
        elif vae_elbo:
            return print_loss, train_loss
        else:
            return print_loss, train_loss

    def kl_diagnormal_diagnormal(self, mean1, log_var1, mean2, log_var2):
        # let dist1 be from reconstructed data
        # let dist2 be from the original data
        dist1 = torch.distributions.Normal(mean1, torch.exp(log_var1 / 2))  # distribution Q
        dist2 = torch.distributions.Normal(mean2, torch.exp(log_var2 / 2))  # distribution P
        return torch.sum(torch.distributions.kl_divergence(dist1, dist2))

    def kl_diagnormal_stdnormal(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior))

    def kl_diagnormal_stdnormal_x(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior_x))

    def kl_diagnormal_stdnormal2(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior))

    def neg_gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(-dist.log_prob(targets))

    def gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(dist.log_prob(targets))

    def forward(self, data, mask):
        z_q, mean_q, logvar_q = self.encoder(data, mask)
        x_mean_q, x_logvar_q = self.decoder(z_q)
        return mean_q, logvar_q, x_mean_q, x_logvar_q


"""Functions that check types."""


def is_bool(x):
    return isinstance(x, bool)


def is_int(x):
    return isinstance(x, int)


def is_positive_int(x):
    return is_int(x) and x > 0


def is_nonnegative_int(x):
    return is_int(x) and x >= 0


def is_power_of_two(n):
    if is_positive_int(n):
        return not n & (n - 1)
    else:
        return False


def tile(x, n):
    if not is_positive_int(n):
        raise TypeError("Argument 'n' must be a positive integer.")
    x_ = x.reshape(-1)
    x_ = x_.repeat(n)
    x_ = x_.reshape(n, -1)
    x_ = x_.transpose(1, 0)
    x_ = x_.reshape(-1)
    return x_


def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    if not is_nonnegative_int(num_batch_dims):
        raise TypeError("Number of batch dimensions must be a non-negative integer.")
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


def split_leading_dim(x, shape):
    """Reshapes the leading dim of `x` to have the given shape."""
    new_shape = torch.Size(shape) + x.shape[1:]
    return torch.reshape(x, new_shape)


def merge_leading_dims(x, num_dims):
    """Reshapes the tensor `x` such that the first `num_dims` dimensions are merged to one."""
    if not is_positive_int(num_dims):
        raise TypeError("Number of leading dims must be a positive integer.")
    if num_dims > x.dim():
        raise ValueError(
            "Number of leading dims can't be greater than total number of dims."
        )
    new_shape = torch.Size([-1]) + x.shape[num_dims:]
    return torch.reshape(x, new_shape)


def repeat_rows(x, num_reps):
    """Each row of tensor `x` is repeated `num_reps` times along leading dimension."""
    if not is_positive_int(num_reps):
        raise TypeError("Number of repetitions must be a positive integer.")
    shape = x.shape
    x = x.unsqueeze(1)
    x = x.expand(shape[0], num_reps, *shape[1:])
    return merge_leading_dims(x, num_dims=2)


def tensor2numpy(x):
    return x.detach().cpu().numpy()


# def logabsdet(x):
#    """Returns the log absolute determinant of square matrix x."""
#    # Note: torch.logdet() only works for positive determinant.
#    _, res = torch.slogdet(x)
#    return res


def random_orthogonal(size):
    """
    Returns a random orthogonal matrix as a 2-dim tensor of shape [size, size].
    """

    # Use the QR decomposition of a random Gaussian matrix.
    x = torch.randn(size, size)
    q, _ = torch.qr(x)
    return q


def get_num_parameters(model):
    """
    Returns the number of trainable parameters in a model of type nets.Module
    :param model: nets.Module containing trainable parameters
    :return: number of trainable parameters in model
    """
    num_parameters = 0
    for parameter in model.parameters():
        num_parameters += torch.numel(parameter)
    return num_parameters


def create_alternating_binary_mask(features, even=True):
    """
    Creates a binary mask of a given dimension which alternates its masking.
    :param features: Dimension of mask.
    :param even: If True, even values are assigned 1s, odd 0s. If False, vice versa.
    :return: Alternating binary mask of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    start = 0 if even else 1
    mask[start::2] += 1
    return mask


def create_mid_split_binary_mask(features):
    """
    Creates a binary mask of a given dimension which splits its masking at the midpoint.
    :param features: Dimension of mask.
    :return: Binary mask split at midpoint of type torch.Tensor
    """
    mask = torch.zeros(features).byte()
    midpoint = features // 2 if features % 2 == 0 else features // 2 + 1
    mask[:midpoint] += 1
    return mask


def create_random_binary_mask(features):
    """
    Creates a random binary mask of a given dimension with half of its entries
    randomly set to 1s.
    :param features: Dimension of mask.
    :return: Binary mask with half of its entries set to 1s, of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    weights = torch.ones(features).float()
    num_samples = features // 2 if features % 2 == 0 else features // 2 + 1
    indices = torch.multinomial(
        input=weights, num_samples=num_samples, replacement=False
    )
    mask[indices] += 1
    return mask


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def cbrt(x):
    """Cube root. Equivalent to torch.pow(x, 1/3), but numerically stable."""
    return torch.sign(x) * torch.exp(torch.log(torch.abs(x)) / 3.0)


def get_temperature(max_value, bound=1 - 1e-3):
    """
    For a dataset with max value 'max_value', returns the temperature such that
        sigmoid(temperature * max_value) = bound.
    If temperature is greater than 1, returns 1.
    :param max_value:
    :param bound:
    :return:
    """
    max_value = torch.Tensor([max_value])
    bound = torch.Tensor([bound])
    temperature = min(-(1 / max_value) * (torch.log1p(-bound) - torch.log(bound)), 1)
    return temperature


def gaussian_kde_log_eval(samples, query):
    N, D = samples.shape[0], samples.shape[-1]
    std = N ** (-1 / (D + 4))
    precision = (1 / (std ** 2)) * torch.eye(D)
    a = query - samples
    b = a @ precision
    c = -0.5 * torch.sum(a * b, dim=-1)
    d = -np.log(N) - (D / 2) * np.log(2 * np.pi) - D * np.log(std)
    c += d
    return torch.logsumexp(c, dim=-1)


class InverseNotAvailable(Exception):
    """Exception to be thrown when a transform does not have an inverse."""

    pass


class InputOutsideDomain(Exception):
    """Exception to be thrown when the input to a transform is not within its domain."""

    pass


class Transform(nn.Module):
    """Base class for all transform objects."""

    def forward(self, inputs, context=None):
        raise NotImplementedError()

    def inverse(self, inputs, context=None):
        raise InverseNotAvailable()


class CompositeTransform(Transform):
    """Composes several transforms into one, in the order they are given."""

    def __init__(self, transforms):
        """Constructor.
        Args:
            transforms: an iterable of `Transform` objects.
        """
        super().__init__()
        self._transforms = nn.ModuleList(transforms)

    @staticmethod
    def _cascade(inputs, funcs, context):
        batch_size = inputs.shape[0]
        outputs = inputs
        total_logabsdet = inputs.new_zeros(batch_size)
        for func in funcs:
            outputs, logabsdet = func(outputs, context)
            total_logabsdet += logabsdet
        return outputs, total_logabsdet

    def forward(self, inputs, context=None):
        funcs = self._transforms
        return self._cascade(inputs, funcs, context)

    def inverse(self, inputs, context=None):
        funcs = (transform.inverse for transform in self._transforms[::-1])
        return self._cascade(inputs, funcs, context)


class MultiscaleCompositeTransform(Transform):
    """A multiscale composite transform as described in the RealNVP paper.
    Splits the outputs along the given dimension after every transform, outputs one half, and
    passes the other half to further transforms. No splitting is done before the last transform.
    Note: Inputs could be of arbitrary shape, but outputs will always be flattened.
    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def __init__(self, num_transforms, split_dim=1):
        """Constructor.
        Args:
            num_transforms: int, total number of transforms to be added.
            split_dim: dimension along which to split.
        """
        if not is_positive_int(split_dim):
            raise TypeError("Split dimension must be a positive integer.")

        super().__init__()
        self._transforms = nn.ModuleList()
        self._output_shapes = []
        self._num_transforms = num_transforms
        self._split_dim = split_dim

    def add_transform(self, transform, transform_output_shape):
        """Add a transform. Must be called exactly `num_transforms` times.
        Parameters:
            transform: the `Transform` object to be added.
            transform_output_shape: tuple, shape of transform's outputs, excl. the first batch
                dimension.
        Returns:
            Input shape for the next transform, or None if adding the last transform.
        """
        assert len(self._transforms) <= self._num_transforms

        if len(self._transforms) == self._num_transforms:
            raise RuntimeError(
                "Adding more than {} transforms is not allowed.".format(
                    self._num_transforms
                )
            )

        if (self._split_dim - 1) >= len(transform_output_shape):
            raise ValueError("No split_dim in output shape")

        if transform_output_shape[self._split_dim - 1] < 2:
            raise ValueError(
                "Size of dimension {} must be at least 2.".format(self._split_dim)
            )

        self._transforms.append(transform)

        if len(self._transforms) != self._num_transforms:  # Unless last transform.
            output_shape = list(transform_output_shape)
            output_shape[self._split_dim - 1] = (
                                                        output_shape[self._split_dim - 1] + 1
                                                ) // 2
            output_shape = tuple(output_shape)

            hidden_shape = list(transform_output_shape)
            hidden_shape[self._split_dim - 1] = hidden_shape[self._split_dim - 1] // 2
            hidden_shape = tuple(hidden_shape)
        else:
            # No splitting for last transform.
            output_shape = transform_output_shape
            hidden_shape = None

        self._output_shapes.append(output_shape)
        return hidden_shape

    def forward(self, inputs, context=None):
        if self._split_dim >= inputs.dim():
            raise ValueError("No split_dim in inputs.")
        if self._num_transforms != len(self._transforms):
            raise RuntimeError(
                "Expecting exactly {} transform(s) "
                "to be added.".format(self._num_transforms)
            )

        batch_size = inputs.shape[0]

        def cascade():
            hiddens = inputs

            for i, transform in enumerate(self._transforms[:-1]):
                transform_outputs, logabsdet = transform(hiddens, context)
                outputs, hiddens = torch.chunk(
                    transform_outputs, chunks=2, dim=self._split_dim
                )
                assert outputs.shape[1:] == self._output_shapes[i]
                yield outputs, logabsdet

            # Don't do the splitting for the last transform.
            outputs, logabsdet = self._transforms[-1](hiddens, context)
            yield outputs, logabsdet

        all_outputs = []
        total_logabsdet = inputs.new_zeros(batch_size)

        for outputs, logabsdet in cascade():
            all_outputs.append(outputs.reshape(batch_size, -1))
            total_logabsdet += logabsdet

        all_outputs = torch.cat(all_outputs, dim=-1)
        return all_outputs, total_logabsdet

    def inverse(self, inputs, context=None):
        if inputs.dim() != 2:
            raise ValueError("Expecting NxD inputs")
        if self._num_transforms != len(self._transforms):
            raise RuntimeError(
                "Expecting exactly {} transform(s) "
                "to be added.".format(self._num_transforms)
            )

        batch_size = inputs.shape[0]

        rev_inv_transforms = [transform.inverse for transform in self._transforms[::-1]]

        split_indices = np.cumsum([np.prod(shape) for shape in self._output_shapes])
        split_indices = np.insert(split_indices, 0, 0)

        split_inputs = []
        for i in range(len(self._output_shapes)):
            flat_input = inputs[:, split_indices[i]: split_indices[i + 1]]
            split_inputs.append(flat_input.view(-1, *self._output_shapes[i]))
        rev_split_inputs = split_inputs[::-1]

        total_logabsdet = inputs.new_zeros(batch_size)

        # We don't do the splitting for the last (here first) transform.
        hiddens, logabsdet = rev_inv_transforms[0](rev_split_inputs[0], context)
        total_logabsdet += logabsdet

        for inv_transform, input_chunk in zip(
                rev_inv_transforms[1:], rev_split_inputs[1:]
        ):
            tmp_concat_inputs = torch.cat([input_chunk, hiddens], dim=self._split_dim)
            hiddens, logabsdet = inv_transform(tmp_concat_inputs, context)
            total_logabsdet += logabsdet

        outputs = hiddens

        return outputs, total_logabsdet


class ActNorm(nn.Module):
    # https://github.com/tonyduan/normalizing-flows/blob/master/nf/flows.py
    """
    ActNorm layer.
    [Kingma and Dhariwal, 2018.]
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mu = nn.Parameter(torch.zeros(dim, dtype=torch.float))
        self.log_sigma = nn.Parameter(torch.zeros(dim, dtype=torch.float))

    # def forward(self, x):
    #     z = x * torch.exp(self.log_sigma) + self.mu
    #     log_det = torch.sum(self.log_sigma)
    #     return z, log_det

    def forward(self, x, cond):
        mu = cond[:, 0:1]
        log_sigma = cond[:, 1:2]
        z = x * torch.exp(log_sigma) + mu
        log_det = torch.sum(log_sigma, dim=1)
        return z, log_det

    def inverse(self, z, cond):
        mu = cond[:, 0:1]
        log_sigma = cond[:, 1:2]
        x = (z - mu) / torch.exp(log_sigma)
        log_det = -torch.sum(log_sigma)
        return x, log_det


class InverseTransform(Transform):
    """Creates a transform that is the inverse of a given transform."""

    def __init__(self, transform):
        """Constructor.
        Args:
            transform: An object of type `Transform`.
        """
        super().__init__()
        self._transform = transform

    def forward(self, inputs, context=None):
        return self._transform.inverse(inputs, context)

    def inverse(self, inputs, context=None):
        return self._transform(inputs, context)


# inputs = latents with dimension 10: [B, latent_dim(10)]
# unnormalized_pdf = smth with fixed dimension 10 : [B, 1, 10]
def unconstrained_linear_spline(
        inputs, unnormalized_pdf, inverse=False, tail_bound=1.0, tails="linear"
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)
    if tails == "linear":
        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    inputs = inputs * inside_interval_mask
    for i in range(10):
        unnormalized_pdf[:, i, :] = unnormalized_pdf[:, i, :] * inside_interval_mask

    if torch.any(inside_interval_mask):
        outputs, logabsdet = linear_spline(
            inputs=inputs,
            unnormalized_pdf=unnormalized_pdf,
            inverse=inverse,
            left=-tail_bound,
            right=tail_bound,
            bottom=-tail_bound,
            top=tail_bound,
        )
    return outputs, logabsdet


def linear_spline(
        inputs, unnormalized_pdf, inverse=False, left=0.0, right=1.0, bottom=0.0, top=1.0
):
    """
    Reference:
    > Müller et al., Neural Importance Sampling, arXiv:1808.03856, 2018.
    """
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise InputOutsideDomain()

    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
    else:
        inputs = (inputs - left) / (right - left)

    num_bins = unnormalized_pdf.size(-1)

    pdf = F.softmax(unnormalized_pdf, dim=-1)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf[..., -1] = 1.0
    cdf = F.pad(cdf, pad=(1, 0), mode="constant", value=0.0)
    if inverse:
        inv_bin_idx = searchsorted(cdf, inputs)

        bin_boundaries = (
            torch.linspace(0, 1, num_bins + 1)
                .view([1] * inputs.dim() + [-1])
                .expand(*inputs.shape, -1)
        )

        slopes = (cdf[..., 1:] - cdf[..., :-1]) / (
                bin_boundaries[..., 1:] - bin_boundaries[..., :-1]
        )
        offsets = cdf[..., 1:] - slopes * bin_boundaries[..., 1:]

        inv_bin_idx = inv_bin_idx.unsqueeze(-1)
        input_slopes = slopes.gather(-1, inv_bin_idx)[..., 0]
        input_offsets = offsets.gather(-1, inv_bin_idx)[..., 0]

        outputs = (inputs - input_offsets) / input_slopes
        outputs = torch.clamp(outputs, 0, 1)

        logabsdet = -torch.log(input_slopes)
    else:
        bin_pos = inputs * num_bins

        bin_idx = torch.floor(bin_pos).long()
        bin_idx[bin_idx >= num_bins] = num_bins - 1
        alpha = bin_pos - bin_idx.float()
        input_pdfs = pdf.gather(-1, bin_idx[..., None])[..., 0]

        outputs = cdf.gather(-1, bin_idx[..., None])[..., 0]
        outputs += alpha * input_pdfs
        outputs = torch.clamp(outputs, 0, 1)

        bin_width = 1.0 / num_bins
        logabsdet = torch.log(input_pdfs) - np.log(bin_width)

    if inverse:
        outputs = outputs * (right - left) + left
    else:
        outputs = outputs * (top - bottom) + bottom

    return outputs, logabsdet


def _share_across_batch(params, batch_size):
    return params[None, ...].expand(batch_size, *params.shape)


class PiecewiseLinearCDF(Transform):
    def __init__(self, shape, num_bins=10, tails=None, tail_bound=1.0):
        super().__init__()

        self.tail_bound = tail_bound
        self.tails = tails

        self.unnormalized_pdf = nn.Parameter(torch.randn(*shape, num_bins))

    def _spline(self, inputs, context, inverse=False):
        batch_size = inputs.shape[0]

        unnormalized_pdf = context.reshape(batch_size, 10, 10)

        if self.tails is None:
            outputs, logabsdet = linear_spline(
                inputs=inputs, unnormalized_pdf=unnormalized_pdf, inverse=inverse
            )
        else:
            outputs, logabsdet = unconstrained_linear_spline(
                inputs=inputs,
                unnormalized_pdf=unnormalized_pdf,
                inverse=inverse,
                tails=self.tails,
                tail_bound=self.tail_bound,
            )
        return outputs, logabsdet

    def forward(self, inputs, context=None):
        return self._spline(inputs, context, inverse=False)

    def inverse(self, inputs, context=None):
        return self._spline(inputs, context, inverse=True)


class Flow(nn.Module):
    def __init__(self, dim, dim_cond):
        super(Flow, self).__init__()
        self.dim = dim
        self.dim_cond = dim_cond

        self.flows = nn.ModuleList()
        self.flows.append(PiecewiseLinearCDF((self.dim,), tails="linear"))
        self.flows.append(PiecewiseLinearCDF((self.dim,), tails="linear"))
        self.flows.append(PiecewiseLinearCDF((self.dim,), tails="linear"))
        # PiecewiseLinearCDF
        # self.flows.append(ActNorm(self.dim))

    def forward(self, cond):
        B = cond.shape[0]  # batch size

        # (1) sample from prior
        dist = torch.distributions.Normal(torch.zeros(B, self.dim), torch.ones(B, self.dim))
        z = dist.rsample()
        log_prob = dist.log_prob(z)

        log_det = 0
        for flow in self.flows:
            z, log_det_ = flow(z, cond)
            log_det += log_det_
        return z, log_prob - log_det

    def backward(self, z, cond):
        """
        Returns the log-prob of z for the given conditioning
        """
        B = cond.shape[0]  # batch size

        log_det = 0
        for flow in self.flows[::-1]:
            z, l = flow.inverse(z, cond)
            log_det += l

        # probability of z under prior
        dist = torch.distributions.Normal(torch.zeros(B, self.dim), torch.ones(B, self.dim))

        return dist.log_prob(z) - log_det


class VAEFlow(nn.Module):
    """ VAE with normalizing flows as the posterior distribution """

    def __init__(self, obs_dim, hid_dim, K, latent_dim, training_parameters, num_samples=1, num_estimates=1):
        super(VAEFlow, self).__init__()
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.latent_dim = latent_dim
        self.K = K
        self.num_samples = num_samples
        self.num_estimates = num_estimates
        # self.missing_rate = missing_rate

        # TODO predict or use fixed value? predict
        self.obs_logvar = -8  # -6 # -3. # None to deactivate

        self.training_parameters = training_parameters

        self.flow = Flow(self.latent_dim, 2 * self.latent_dim)

        # encoder: q(z2 | x)
        activation = nn.ELU
        self.seq_encoder = nn.Sequential(
            nn.Linear(2 * obs_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, 100),
            # activation(),
            # nn.Linear(hid_dim, 2 * latent_dim)
        )

        self.encoder_mean = nn.Linear(hid_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hid_dim, latent_dim)

        self.seq_decoder = nn.Sequential(
            # nn.Identity()
            nn.Linear(latent_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            # nn.Linear(hid_dim, obs_dim),
            # nn.Sigmoid()
        )

        self.decoder_mean = nn.Sequential(
            nn.Linear(hid_dim, obs_dim),
            nn.Sigmoid()
        )
        self.decoder_logvar = nn.Sequential(
            nn.Linear(hid_dim, obs_dim),
        )

        # assume factorized prior of normal RVs
        self.prior_mean = torch.nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False)
        self.prior_std = torch.nn.Parameter(torch.ones(self.latent_dim), requires_grad=False)
        self.prior = torch.distributions.Normal(self.prior_mean, self.prior_std)

        # print(self)

    def encoder(self, x, mask, sample=True):
        x = x.float() * mask
        t = self.seq_encoder(torch.cat([x, mask], dim=1))  # TODO
        if sample:
            z, log_prob = self.flow(t)
        else:
            z = mean
        return z, log_prob  # mean, logvar

    def backward(self, z, x, mask):
        """
        Compute the probabiltiy of z for data x given mask
        """
        x = x.float() * mask
        t = self.seq_encoder(torch.cat([x, mask], dim=1))  # TODO
        log_prob = self.flow.backward(z, t)

        return log_prob

    def decoder(self, z_int):
        t = self.seq_decoder(z_int)
        x_mean, x_logvar = self.decoder_mean(t), self.decoder_logvar(t)
        if self.obs_logvar is not None:
            x_logvar = self.obs_logvar * torch.ones_like(x_logvar)
        return x_mean, x_logvar

    def loss(self, x, x_recon, x_logvar, z, z_log_prob, mask, vae_elbo=False, beta=1.0, llh_eval=False):
        """
        Computes the loss of the given data.
        Returns the reconstruction loss for the observed variables and all variables.
        """
        RE_ = torch.sum(self.neg_gaussian_log_likelihood(x * mask, x_recon * mask, x_logvar * mask))
        RE_q_imputed = torch.sum(self.neg_gaussian_log_likelihood(x * ~mask, x_recon * ~mask, x_logvar * ~mask))

        KL = torch.sum(z_log_prob - self.prior.log_prob(z))

        loss = RE_ + beta * KL
        train_loss = loss / x.shape[0]
        print_loss = loss
        if llh_eval:
            return print_loss, train_loss, RE_ / x.shape[0], RE_q_imputed / x.shape[0]
        else:
            return print_loss, train_loss

    def kl_diagnormal_stdnormal(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        # TODO instead of a sum, we had a mean in the next line. That should have been incorrect
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior), dim=1, keepdim=True)

    def neg_gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return -dist.log_prob(targets)

    def forward(self, data, mask):
        z, z_log_prob = self.encoder(data, mask)
        x_mean, x_logvar = self.decoder(z)
        return z, z_log_prob, x_mean, x_logvar


class REG_VAEFlow(nn.Module):
    """ VAE with normalizing flows as the posterior distribution """

    def __init__(self, obs_dim, hid_dim, K, latent_dim, training_parameters, num_samples=1, num_estimates=1):
        super(REG_VAEFlow, self).__init__()
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.latent_dim = latent_dim
        self.K = K
        self.num_samples = num_samples
        self.num_estimates = num_estimates
        # self.missing_rate = missing_rate

        # TODO predict or use fixed value? predict
        self.obs_logvar = -8  # -6 # -3. # None to deactivate

        self.training_parameters = training_parameters

        self.flow = Flow(self.latent_dim, 2 * self.latent_dim)

        # encoder: q(z2 | x)
        activation = nn.ELU
        self.seq_encoder = nn.Sequential(
            nn.Linear(2 * obs_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, 100),
            # activation(),
            # nn.Linear(hid_dim, 2 * latent_dim)
        )

        self.encoder_mean = nn.Linear(hid_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hid_dim, latent_dim)

        self.seq_decoder = nn.Sequential(
            # nn.Identity()
            nn.Linear(latent_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            # nn.Linear(hid_dim, obs_dim),
            # nn.Sigmoid()
        )

        self.decoder_mean = nn.Sequential(
            nn.Linear(hid_dim, obs_dim),
            nn.Sigmoid()
        )
        self.decoder_logvar = nn.Sequential(
            nn.Linear(hid_dim, obs_dim),
        )

        # assume factorized prior of normal RVs
        self.prior_mean = torch.nn.Parameter(torch.zeros(self.latent_dim), requires_grad=False)
        self.prior_std = torch.nn.Parameter(torch.ones(self.latent_dim), requires_grad=False)
        self.prior = torch.distributions.Normal(self.prior_mean, self.prior_std)

        # print(self)

    def encoder(self, x, mask, sample=True):
        x = x.float() * mask
        t = self.seq_encoder(torch.cat([x, mask], dim=1))  # TODO
        # mean, logvar = self.encoder_mean(t), self.encoder_logvar(t)
        if sample:
            z, log_prob = self.flow(t)
        else:
            z = mean
        # print(z.shape)
        return z, log_prob  # mean, logvar

    def backward(self, z, x, mask):
        """
        Compute the probabiltiy of z for data x given mask
        """
        x = x.float() * mask
        t = self.seq_encoder(torch.cat([x, mask], dim=1))  # TODO
        log_prob = self.flow.backward(z, t)

        return log_prob

    def decoder(self, z_int):
        t = self.seq_decoder(z_int)
        x_mean, x_logvar = self.decoder_mean(t), self.decoder_logvar(t)
        if self.obs_logvar is not None:
            x_logvar = self.obs_logvar * torch.ones_like(x_logvar)
        return x_mean, x_logvar

    def loss(self, x, x_recon_q, x_logvar_q, z_q, z_log_prob_q, x_recon_p, x_logvar_p, z_p, z_log_prob_p, mask, mask_p,
             alpha, beta=1.0, llh_eval=False, stage='train'):
        """
        Computes the loss of the given data.
        Returns the reconstruction loss for the observed variables and all variables.
        """
        if stage == 'train':
            RE_q = torch.sum(self.neg_gaussian_log_likelihood(x * mask, x_recon_q * mask, x_logvar_q * mask))
            RE_p = torch.sum(self.neg_gaussian_log_likelihood(x * mask_p, x_recon_p * mask_p, x_logvar_p * mask_p))
            KL_q = torch.sum(z_log_prob_q - self.prior.log_prob(z_q))
            KL_p = torch.sum(z_log_prob_p - self.prior.log_prob(z_p))
            loss_q = RE_q + beta * KL_q
            loss_p = RE_p + beta * KL_p
            KL_reg = torch.sum(torch.abs(z_log_prob_q - z_log_prob_p))

            loss = loss_q + alpha * (
                    KL_reg - loss_q + loss_p + torch.sum(self.neg_gaussian_log_likelihood(x * mask * ~mask_p,
                                                                                          x_recon_q * mask * ~mask_p,
                                                                                          x_logvar_q * mask * ~mask_p)))
            RE_q_imputed = 0
        else:
            RE_q = torch.sum(self.neg_gaussian_log_likelihood(x * mask, x_recon_q * mask, x_logvar_q * mask))
            RE_q_imputed = torch.sum(self.neg_gaussian_log_likelihood(x * ~mask, x_recon_q * ~mask, x_logvar_q * ~mask))
            KL_q = torch.sum(z_log_prob_q - self.prior.log_prob(z_q))
            loss_q = RE_q + beta * KL_q
            loss = loss_q

        train_loss = loss / x.shape[0]
        print_loss = train_loss

        if llh_eval:
            return print_loss, train_loss, RE_q / x.shape[0], RE_q_imputed / x.shape[0]
        else:
            return print_loss, train_loss

    def kl_diagnormal_stdnormal(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        # TODO instead of a sum, we had a mean in the next line. That should have been incorrect
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior), dim=1, keepdim=True)

    def neg_gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return -dist.log_prob(targets)

    def forward(self, data, mask, mask_p):
        z_q, z_log_prob_q = self.encoder(data, mask)
        z_p, z_log_prob_p = self.encoder(data, mask_p)
        x_mean_q, x_logvar_q = self.decoder(z_q)
        x_mean_p, x_logvar_p = self.decoder(z_p)
        return z_p, z_log_prob_p, x_mean_p, x_logvar_p, z_q, z_log_prob_q, x_mean_q, x_logvar_q


def softmax(x):
    e_x = torch.exp(x - torch.max(x, 1)[0][:, None])
    return e_x / e_x.sum(1)[:, None]


class REG_notMIWAE_new_version(nn.Module):
    def __init__(self, obs_dim, hid_dim, K, latent_dim, training_parameters, num_samples, num_estimates):
        super(REG_notMIWAE_new_version, self).__init__()
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.emb_dim = 10
        self.num_samples = num_samples
        self.num_estimates = num_estimates
        self.latent_dim = latent_dim
        self.batch_size = training_parameters['batch_size']
        self.K = K
        self.obs_std = 0.1
        self.number_components = 500
        self.training_paramters = training_parameters
        # encoder: q(z2 | x)
        self.seq_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU()
            # nn.Linear(128, 2 * latent_dim)
            # nn.LeakyReLU()
        ).float()
        self.q_mu = nn.Sequential(
            nn.Linear(128, latent_dim),

        )
        self.q_logstd = nn.Sequential(
            nn.Linear(128, latent_dim)
            # nn.Hardtanh(min_val=-10, max_val=10)
        )

        self.seq_decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ELU(),
                                         nn.Linear(128, 128), nn.ELU())
        # self.x_logvar = torch.log(torch.square(torch.Tensor([0.1 * np.sqrt(2)])))
        self.x_mean = nn.Sequential(nn.Linear(128, obs_dim), nn.Sigmoid())
        self.x_logvar = nn.Sequential(nn.Linear(128, obs_dim), nn.Hardtanh(min_val=-10, max_val=0))
        emb1 = torch.empty([1, 1, self.obs_dim])
        nn.init.xavier_uniform_(emb1)
        self.W = nn.Parameter(emb1)
        emb2 = torch.empty([1, 1, self.obs_dim])
        nn.init.xavier_uniform_(emb2)
        self.b = nn.Parameter(emb2)
        self.activation = nn.Softplus()
        self.logits = nn.Sequential(nn.Linear(obs_dim, obs_dim)).double()

        # assume factorized prior of normal RVs
        self.prior = torch.distributions.Normal(torch.zeros(self.latent_dim), torch.ones(self.latent_dim))
        self.max_epoch = 2800

    def encoder(self, x, mask, sample=True):
        x = self.seq_encoder(x.float() * mask.float())
        mean = self.q_mu(x)
        log_var = self.q_logstd(x)
        # std = torch.exp(log_std)
        mean = mean.repeat(self.num_samples, 1, 1).permute(1, 0, 2)  # shape [batch_size, n_samples, obs_dim]
        log_var = log_var.repeat(self.num_samples, 1, 1).permute(1, 0, 2)  # shape [batch_size, n_samples, obs_dim]
        if sample:
            dist = torch.distributions.Normal(mean, torch.exp(log_var / 2))
            z = dist.rsample()
        else:
            z = mean
        return z, mean, log_var

    def decoder(self, z_int):
        decoded = self.seq_decoder(z_int)
        x_mean = self.x_mean(decoded)
        x_logvar = self.x_logvar(decoded)
        return x_mean, x_logvar

    def loss(self, x, x_recon_q, x_logvar_q, mean_q, logvar_q, mask,
             epoch, vae_elbo=False, llh_eval=False,
             MI=False,
             beta_annealing=False, beta=1.0, alpha=1.0, alpha_annealing=False, stage='train',
             missing_process='selfmasking_known'):
        new_x = x[:, 0:].repeat(self.num_samples, 1, 1).permute(1, 0, 2)
        new_mask = mask[:, 0:].repeat(self.num_samples, 1, 1).permute(1, 0, 2)
        # new_mask_p = mask_p[:, 0:].repeat(self.num_samples, 1, 1).permute(1, 0, 2)

        out_mixed_q = x_recon_q * (1 - new_mask) + new_x * new_mask
        # out_mixed_p = x_recon_p * (1 - new_mask_p) + new_x * new_mask_p

        # print(out_mixed.type())
        # print(mean.shape)
        # print(logvar.shape)
        if missing_process == 'selfmasking':
            logits_q = - self.W * (out_mixed_q - self.b)
            # logits_p = - self.W * (out_mixed_p - self.b)

        # the type of masking I use
        elif missing_process == 'selfmasking_known':
            # self.W = F.softplus(self.W)
            logits_q = - self.activation(self.W) * (out_mixed_q - self.b)
            # logits_p = - self.activation(self.W) * (out_mixed_p - self.b)


        else:  # linear
            logits_q = self.logits(out_mixed_q)
            # logits_p = self.logits(out_mixed_p)

        p_s_given_x_q = torch.distributions.Bernoulli(logits=logits_q)
        log_p_s_given_x_q = torch.sum(p_s_given_x_q.log_prob(new_mask), 2)
        mask_samples = p_s_given_x_q.sample()
        mask_p = mask_samples[:, 0, :] * mask
        new_mask_p = mask_p[:, 0:].repeat(self.num_samples, 1, 1).permute(1, 0, 2)

        z_p, mean_p, logvar_p = self.encoder(x, mask_p)
        x_recon_p, x_logvar_p = self.decoder(z_p)

        out_mixed_p = x_recon_p * (1 - new_mask_p) + new_x * new_mask_p
        logits_p = - self.activation(self.W) * (out_mixed_p - self.b)

        p_s_given_x_p = torch.distributions.Bernoulli(logits=logits_p)
        log_p_s_given_x_p = torch.sum(p_s_given_x_p.log_prob(new_mask_p), 2)

        RE_q = self.neg_gaussian_log_likelihood(new_x * new_mask, x_recon_q * new_mask,
                                                x_logvar_q * new_mask)  # TODO computed for all variables (also imputed(?) ones)
        RE_p = self.neg_gaussian_log_likelihood(new_x * new_mask_p, x_recon_p * new_mask_p,
                                                x_logvar_p * new_mask_p)


        KL_q = self.kl_diagnormal_stdnormal(mean_q, logvar_q)
        KL_p = self.kl_diagnormal_stdnormal(mean_p, logvar_p)
        l_w_q = RE_q + KL_q - log_p_s_given_x_q
        l_w_p = RE_p + KL_p - log_p_s_given_x_p

        log_sum_w_q = torch.logsumexp(l_w_q, 1)
        log_avg_weight_q = log_sum_w_q - math.log(float(self.num_samples))
        loss_q = torch.mean(log_avg_weight_q)

        log_sum_w_p = torch.logsumexp(l_w_p, 1)
        log_avg_weight_p = log_sum_w_p - math.log(float(self.num_samples))
        KL_reg = self.kl_diagnormal_diagnormal(mean_q, logvar_q, mean_p, logvar_p).mean()

        loss_p = torch.mean(log_avg_weight_p)
        loss = loss_q + alpha * (
                KL_reg - loss_q + loss_p + self.neg_gaussian_log_likelihood(new_x * new_mask * (1 - new_mask_p),
                                                                            x_recon_q * new_mask * (1 - new_mask_p),
                                                                            x_logvar_q * new_mask * (
                                                                                    1 - new_mask_p)).mean())
        train_loss = loss
        print_loss = loss
        if llh_eval:
            wl = softmax(-l_w_q)
            xm = torch.sum((x_recon_q.T * wl.T).T, 1)
            return xm, train_loss, RE_q.mean()
        elif MI:
            aggregated_mean = torch.mean(mean, 0).mean(0).mean(0)
            aggregated_logvar = torch.mean(logvar, 0).mean(0).mean(0)

            KL_agg = self.kl_diagnormal_stdnormal2(aggregated_mean, aggregated_logvar)
            mut_inf = KL.mean() - KL_agg
            return print_loss, train_loss, mut_inf
        elif vae_elbo:
            return print_loss, train_loss
        else:
            return print_loss, train_loss

    def kl_diagnormal_diagnormal(self, mean1, log_var1, mean2, log_var2):
        # let dist1 be from reconstructed data
        # let dist2 be from the original data
        dist1 = torch.distributions.Normal(mean1, torch.exp(log_var1 / 2))  # distribution Q
        dist2 = torch.distributions.Normal(mean2, torch.exp(log_var2 / 2))  # distribution P
        # dist2 = torch.distributions.Normal(((mean2 * torch.exp(logvar_bar)) + mean_bar*torch.exp(log_var2))/(torch.exp(logvar_bar)+ torch.exp(log_var2)),
        #                                   torch.sqrt((torch.exp(log_var2)*torch.exp(logvar_bar))/(torch.exp(logvar_bar) + torch.exp(log_var2))))
        # TODO instead of a sum, we had a mean in the next line. That should have been incorrect
        return torch.distributions.kl_divergence(dist1, dist2)

    def kl_diagnormal_stdnormal(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        # TODO instead of a sum, we had a mean in the next line. That should have been incorrect
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior), 2)

    def kl_diagnormal_stdnormal2(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        # TODO instead of a sum, we had a mean in the next line. That should have been incorrect
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior))

    def neg_gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(-dist.log_prob(targets), 2)

    def reparameterization(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data, mask, stage='train'):
        z_q, mean_q, logvar_q = self.encoder(data, mask)
        x_mean_q, x_logvar_q = self.decoder(z_q)
        # z_p, mean_p, logvar_p = self.encoder(data, mask_p)
        # x_mean_p, x_logvar_p = self.decoder(z_p)
        return mean_q, logvar_q, x_mean_q, x_logvar_q


class REG_notMIWAE_v2(nn.Module):
    def __init__(self, obs_dim, hid_dim, K, latent_dim, training_parameters, num_samples, num_estimates):
        super(REG_notMIWAE_v2, self).__init__()
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.emb_dim = 10
        self.num_samples = num_samples
        self.num_estimates = num_estimates
        self.latent_dim = latent_dim
        self.batch_size = training_parameters['batch_size']
        self.K = K
        self.obs_std = 0.1
        self.number_components = 500
        self.training_paramters = training_parameters
        # encoder: q(z2 | x)
        self.seq_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU()
            # nn.Linear(128, 2 * latent_dim)
            # nn.LeakyReLU()
        ).float()
        self.q_mu = nn.Sequential(
            nn.Linear(128, latent_dim),

        )
        self.q_logstd = nn.Sequential(
            nn.Linear(128, latent_dim)
            # nn.Hardtanh(min_val=-10, max_val=10)
        )

        self.seq_decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ELU(),
                                         nn.Linear(128, 128), nn.ELU())
        # self.x_logvar = torch.log(torch.square(torch.Tensor([0.1 * np.sqrt(2)])))
        self.x_mean = nn.Sequential(nn.Linear(128, obs_dim), nn.Sigmoid())
        self.x_logvar = nn.Sequential(nn.Linear(128, obs_dim), nn.Hardtanh(min_val=-10, max_val=0))
        emb1 = torch.empty([1, 1, self.obs_dim])
        nn.init.xavier_uniform_(emb1)
        self.W = nn.Parameter(emb1)
        emb2 = torch.empty([1, 1, self.obs_dim])
        nn.init.xavier_uniform_(emb2)
        self.b = nn.Parameter(emb2)
        self.activation = nn.Softplus()
        self.logits = nn.Sequential(nn.Linear(obs_dim, obs_dim)).double()

        # assume factorized prior of normal RVs
        self.prior = torch.distributions.Normal(torch.zeros(self.latent_dim), torch.ones(self.latent_dim))
        self.max_epoch = 2800

    def encoder(self, x, mask, sample=True):
        x = self.seq_encoder(x.float() * mask.float())
        mean = self.q_mu(x)
        log_var = self.q_logstd(x)
        # std = torch.exp(log_std)
        mean = mean.repeat(self.num_samples, 1, 1).permute(1, 0, 2)  # shape [batch_size, n_samples, obs_dim]
        log_var = log_var.repeat(self.num_samples, 1, 1).permute(1, 0, 2)  # shape [batch_size, n_samples, obs_dim]
        if sample:
            # log_std = logvar / 2
            dist = torch.distributions.Normal(mean, torch.exp(log_var / 2))
            z = dist.rsample()
        else:
            z = mean
        return z, mean, log_var

    def decoder(self, z_int):
        decoded = self.seq_decoder(z_int)
        x_mean = self.x_mean(decoded)
        x_logvar = self.x_logvar(decoded)
        return x_mean, x_logvar

    def loss(self, x, x_recon_p, x_logvar_p, mean_p, logvar_p, x_recon_q, x_logvar_q, mean_q, logvar_q, mask, mask_p,
             epoch, vae_elbo=False, llh_eval=False,
             MI=False,
             beta_annealing=False, beta=1.0, alpha=1.0, alpha_annealing=False, stage='train',
             missing_process='selfmasking_known'):
        new_x = x[:, 0:].repeat(self.num_samples, 1, 1).permute(1, 0, 2)
        new_mask = mask[:, 0:].repeat(self.num_samples, 1, 1).permute(1, 0, 2)
        new_mask_p = mask_p[:, 0:].repeat(self.num_samples, 1, 1).permute(1, 0, 2)

        out_mixed_q = x_recon_q * (1 - new_mask) + new_x * new_mask
        # out_mixed_p = x_recon_p * (1 - new_mask_p) + new_x * new_mask_p

        if missing_process == 'selfmasking':
            logits_q = - self.W * (out_mixed_q - self.b)
            # logits_p = - self.W * (out_mixed_p - self.b)

        # the type of masking I use
        elif missing_process == 'selfmasking_known':
            # self.W = F.softplus(self.W)
            logits_q = - self.activation(self.W) * (out_mixed_q - self.b)
            # logits_p = - self.activation(self.W) * (out_mixed_p - self.b)


        else:  # linear
            logits_q = self.logits(out_mixed_q)
            # logits_p = self.logits(out_mixed_p)

        RE_q = self.neg_gaussian_log_likelihood(new_x * new_mask, x_recon_q * new_mask,
                                                x_logvar_q * new_mask)  # TODO computed for all variables (also imputed(?) ones)
        RE_p = self.neg_gaussian_log_likelihood(new_x * new_mask_p, x_recon_p * new_mask_p,
                                                x_logvar_p * new_mask_p)


        KL_q = self.kl_diagnormal_stdnormal(mean_q, logvar_q)
        KL_p = self.kl_diagnormal_stdnormal(mean_p, logvar_p)

        p_s_given_x_q = torch.distributions.Bernoulli(logits=logits_q)
        log_p_s_given_x_q = torch.sum(p_s_given_x_q.log_prob(new_mask), 2)

        l_w_q = RE_q + KL_q - log_p_s_given_x_q
        l_w_p = RE_p + KL_p  # - log_p_s_given_x_p

        log_sum_w_q = torch.logsumexp(l_w_q, 1)
        log_avg_weight_q = log_sum_w_q - math.log(float(self.num_samples))
        # print(log_avg_weight.shape)
        loss_q = torch.mean(log_avg_weight_q)

        log_sum_w_p = torch.logsumexp(l_w_p, 1)
        log_avg_weight_p = log_sum_w_p - math.log(float(self.num_samples))
        # print(log_avg_weight.shape)
        KL_reg = self.kl_diagnormal_diagnormal(mean_q, logvar_q, mean_p, logvar_p).mean()

        loss_p = torch.mean(log_avg_weight_p)
        loss = loss_q + alpha * (
                KL_reg - loss_q + loss_p + self.neg_gaussian_log_likelihood(new_x * new_mask * (1 - new_mask_p),
                                                                            x_recon_q * new_mask * (1 - new_mask_p),
                                                                            x_logvar_q * new_mask * (
                                                                                    1 - new_mask_p)).mean())
        train_loss = loss
        print_loss = loss
        if llh_eval:
            wl = softmax(-l_w_q)
            xm = torch.sum((x_recon_q.T * wl.T).T, 1)
            return xm, train_loss, RE_q.mean()
        elif MI:
            aggregated_mean = torch.mean(mean, 0).mean(0).mean(0)
            aggregated_logvar = torch.mean(logvar, 0).mean(0).mean(0)
            KL_agg = self.kl_diagnormal_stdnormal2(aggregated_mean, aggregated_logvar)
            mut_inf = KL.mean() - KL_agg
            return print_loss, train_loss, mut_inf
        elif vae_elbo:
            return print_loss, train_loss
        else:
            return print_loss, train_loss

    def kl_diagnormal_diagnormal(self, mean1, log_var1, mean2, log_var2):
        # let dist1 be from reconstructed data
        # let dist2 be from the original data
        dist1 = torch.distributions.Normal(mean1, torch.exp(log_var1 / 2))  # distribution Q
        dist2 = torch.distributions.Normal(mean2, torch.exp(log_var2 / 2))  # distribution P
        # TODO instead of a sum, we had a mean in the next line. That should have been incorrect
        return torch.distributions.kl_divergence(dist1, dist2)

    def kl_diagnormal_stdnormal(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        # TODO instead of a sum, we had a mean in the next line. That should have been incorrect
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior), 2)

    def kl_diagnormal_stdnormal2(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        # TODO instead of a sum, we had a mean in the next line. That should have been incorrect
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior))

    def neg_gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(-dist.log_prob(targets), 2)

    def reparameterization(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data, mask, mask_p, stage='train'):
        z_q, mean_q, logvar_q = self.encoder(data, mask)
        x_mean_q, x_logvar_q = self.decoder(z_q)
        z_p, mean_p, logvar_p = self.encoder(data, mask_p)
        x_mean_p, x_logvar_p = self.decoder(z_p)
        return mean_p, logvar_p, x_mean_p, x_logvar_p, mean_q, logvar_q, x_mean_q, x_logvar_q


class REG_notMIWAE(nn.Module):
    def __init__(self, obs_dim, hid_dim, K, latent_dim, training_parameters, num_samples, num_estimates):
        super(REG_notMIWAE, self).__init__()
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.emb_dim = 10
        self.num_samples = num_samples
        self.num_estimates = num_estimates
        self.latent_dim = latent_dim
        self.batch_size = training_parameters['batch_size']
        self.K = K
        self.obs_std = 0.1
        self.number_components = 500
        self.training_paramters = training_parameters
        # encoder: q(z2 | x)
        self.seq_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU()
            # nn.Linear(128, 2 * latent_dim)
            # nn.LeakyReLU()
        ).float()
        self.q_mu = nn.Sequential(
            nn.Linear(128, latent_dim),

        )
        self.q_logstd = nn.Sequential(
            nn.Linear(128, latent_dim)
            # nn.Hardtanh(min_val=-10, max_val=10)
        )

        self.seq_decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ELU(),
                                         nn.Linear(128, 128), nn.ELU())
        # self.x_logvar = torch.log(torch.square(torch.Tensor([0.1 * np.sqrt(2)])))
        self.x_mean = nn.Sequential(nn.Linear(128, obs_dim), nn.Sigmoid())
        self.x_logvar = nn.Sequential(nn.Linear(128, obs_dim), nn.Hardtanh(min_val=-10, max_val=0))
        emb1 = torch.empty([1, 1, self.obs_dim])
        nn.init.xavier_uniform_(emb1)
        self.W = nn.Parameter(emb1)
        emb2 = torch.empty([1, 1, self.obs_dim])
        nn.init.xavier_uniform_(emb2)
        self.b = nn.Parameter(emb2)
        self.activation = nn.Softplus()
        self.logits = nn.Sequential(nn.Linear(obs_dim, obs_dim)).double()

        # assume factorized prior of normal RVs
        self.prior = torch.distributions.Normal(torch.zeros(self.latent_dim), torch.ones(self.latent_dim))
        self.max_epoch = 2800

    def encoder(self, x, mask, sample=True):
        x = self.seq_encoder(x.float() * mask.float())
        mean = self.q_mu(x)
        log_var = self.q_logstd(x)
        mean = mean.repeat(self.num_samples, 1, 1).permute(1, 0, 2)  # shape [batch_size, n_samples, obs_dim]
        log_var = log_var.repeat(self.num_samples, 1, 1).permute(1, 0, 2)  # shape [batch_size, n_samples, obs_dim]
        if sample:
            dist = torch.distributions.Normal(mean, torch.exp(log_var / 2))
            z = dist.rsample()
        else:
            z = mean
        return z, mean, log_var

    def decoder(self, z_int):
        # z_int = z_int.reshape(-1, self.latent_dim)
        decoded = self.seq_decoder(z_int)
        x_mean = self.x_mean(decoded)
        x_logvar = self.x_logvar(decoded)
        return x_mean, x_logvar

    def loss(self, x, x_recon_p, x_logvar_p, mean_p, logvar_p, x_recon_q, x_logvar_q, mean_q, logvar_q, mask, mask_p,
             epoch, vae_elbo=False, llh_eval=False,
             MI=False,
             beta_annealing=False, beta=1.0, alpha=1.0, alpha_annealing=False, stage='train',
             missing_process='selfmasking_known'):
        new_x = x[:, 0:].repeat(self.num_samples, 1, 1).permute(1, 0, 2)
        new_mask = mask[:, 0:].repeat(self.num_samples, 1, 1).permute(1, 0, 2)
        new_mask_p = mask_p[:, 0:].repeat(self.num_samples, 1, 1).permute(1, 0, 2)

        out_mixed_q = x_recon_q * (1 - new_mask) + new_x * new_mask
        out_mixed_p = x_recon_p * (1 - new_mask_p) + new_x * new_mask_p

        if missing_process == 'selfmasking':
            logits_q = - self.W * (out_mixed_q - self.b)
            logits_p = - self.W * (out_mixed_p - self.b)

        # the type of masking I use
        elif missing_process == 'selfmasking_known':
            # self.W = F.softplus(self.W)
            logits_q = - self.activation(self.W) * (out_mixed_q - self.b)
            logits_p = - self.activation(self.W) * (out_mixed_p - self.b)


        else:  # linear
            logits_q = self.logits(out_mixed_q)
            logits_p = self.logits(out_mixed_p)

        RE_q = self.neg_gaussian_log_likelihood(new_x * new_mask, x_recon_q * new_mask,
                                                x_logvar_q * new_mask)  # TODO computed for all variables (also imputed(?) ones)
        RE_p = self.neg_gaussian_log_likelihood(new_x * new_mask_p, x_recon_p * new_mask_p,
                                                x_logvar_p * new_mask_p)


        KL_q = self.kl_diagnormal_stdnormal(mean_q, logvar_q)
        KL_p = self.kl_diagnormal_stdnormal(mean_p, logvar_p)

        p_s_given_x_q = torch.distributions.Bernoulli(logits=logits_q)
        log_p_s_given_x_q = torch.sum(p_s_given_x_q.log_prob(new_mask), 2)

        p_s_given_x_p = torch.distributions.Bernoulli(logits=logits_p)
        log_p_s_given_x_p = torch.sum(p_s_given_x_p.log_prob(new_mask_p), 2)

        l_w_q = RE_q + KL_q - log_p_s_given_x_q
        l_w_p = RE_p + KL_p - log_p_s_given_x_p

        log_sum_w_q = torch.logsumexp(l_w_q, 1)
        log_avg_weight_q = log_sum_w_q - math.log(float(self.num_samples))
        loss_q = torch.mean(log_avg_weight_q)

        log_sum_w_p = torch.logsumexp(l_w_p, 1)
        log_avg_weight_p = log_sum_w_p - math.log(float(self.num_samples))
        KL_reg = self.kl_diagnormal_diagnormal(mean_q, logvar_q, mean_p, logvar_p).mean()

        loss_p = torch.mean(log_avg_weight_p)
        loss = loss_q + alpha * (
                KL_reg - loss_q + loss_p + self.neg_gaussian_log_likelihood(new_x * new_mask * (1 - new_mask_p),
                                                                            x_recon_q * new_mask * (1 - new_mask_p),
                                                                            x_logvar_q * new_mask * (
                                                                                    1 - new_mask_p)).mean())
        train_loss = loss
        print_loss = loss
        if llh_eval:
            wl = softmax(-l_w_q)
            xm = torch.sum((x_recon_q.T * wl.T).T, 1)
            return xm, train_loss, RE_q.mean()
        elif MI:
            aggregated_mean = torch.mean(mean, 0).mean(0).mean(0)
            aggregated_logvar = torch.mean(logvar, 0).mean(0).mean(0)
            KL_agg = self.kl_diagnormal_stdnormal2(aggregated_mean, aggregated_logvar)
            mut_inf = KL.mean() - KL_agg
            return print_loss, train_loss, mut_inf
        elif vae_elbo:
            return print_loss, train_loss
        else:
            return print_loss, train_loss

    def kl_diagnormal_diagnormal(self, mean1, log_var1, mean2, log_var2):
        # let dist1 be from reconstructed data
        # let dist2 be from the original data
        dist1 = torch.distributions.Normal(mean1, torch.exp(log_var1 / 2))  # distribution Q
        dist2 = torch.distributions.Normal(mean2, torch.exp(log_var2 / 2))  # distribution P
        # dist2 = torch.distributions.Normal(((mean2 * torch.exp(logvar_bar)) + mean_bar*torch.exp(log_var2))/(torch.exp(logvar_bar)+ torch.exp(log_var2)),
        #                                   torch.sqrt((torch.exp(log_var2)*torch.exp(logvar_bar))/(torch.exp(logvar_bar) + torch.exp(log_var2))))
        # TODO instead of a sum, we had a mean in the next line. That should have been incorrect
        return torch.distributions.kl_divergence(dist1, dist2)

    def kl_diagnormal_stdnormal(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        # TODO instead of a sum, we had a mean in the next line. That should have been incorrect
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior), 2)

    def kl_diagnormal_stdnormal2(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        # TODO instead of a sum, we had a mean in the next line. That should have been incorrect
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior))

    def neg_gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(-dist.log_prob(targets), 2)

    def reparameterization(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data, mask, mask_p):
        z_q, mean_q, logvar_q = self.encoder(data, mask)
        x_mean_q, x_logvar_q = self.decoder(z_q)
        z_p, mean_p, logvar_p = self.encoder(data, mask_p)
        x_mean_p, x_logvar_p = self.decoder(z_p)
        return mean_p, logvar_p, x_mean_p, x_logvar_p, mean_q, logvar_q, x_mean_q, x_logvar_q


class notMIWAE_myversion(nn.Module):
    def __init__(self, obs_dim, hid_dim, K, latent_dim, training_parameters, num_samples, num_estimates):
        super(notMIWAE_myversion, self).__init__()
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.emb_dim = 10
        self.num_samples = num_samples
        self.num_estimates = num_estimates
        self.latent_dim = latent_dim
        self.batch_size = training_parameters['batch_size']
        self.K = K
        self.obs_std = 0.1
        self.number_components = 500
        self.training_paramters = training_parameters
        # encoder: q(z2 | x)
        self.seq_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            # nn.ELU(),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU()
            # nn.Linear(128, 2 * latent_dim)
            # nn.LeakyReLU()
        )
        self.q_mu = nn.Sequential(
            nn.Linear(128, latent_dim),
            # nn.Sigmoid()

        )
        self.q_logstd = nn.Sequential(
            nn.Linear(128, latent_dim),
            # lambda x: torch.clip(x, -10, 10)
            # nn.Hardtanh(min_val=-10, max_val=10)
        )

        self.seq_decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ELU(),
                                         nn.Linear(128, 128), nn.ELU())
        # self.x_logvar = torch.log(torch.square(torch.Tensor([0.1 * np.sqrt(2)])))
        self.x_mean = nn.Sequential(nn.Linear(128, obs_dim), nn.Sigmoid())
        self.x_logvar = nn.Sequential(nn.Linear(128, obs_dim), nn.Hardtanh(min_val=-10.0, max_val=0))

        # nn.Hardtanh(min_val=-10.0, max_val=0))

        # self.x_std = nn.Sequential(nn.Linear(128, obs_dim), nn.Softplus())
        emb1 = torch.empty([1, 1, self.obs_dim])
        nn.init.xavier_uniform_(emb1)
        self.W = nn.Parameter(emb1, requires_grad=True)
        emb2 = torch.empty([1, 1, self.obs_dim])
        nn.init.xavier_uniform_(emb2)
        self.b = nn.Parameter(emb2, requires_grad=True)
        self.activation = nn.Softplus()
        # self.logits = nn.Sequential(nn.Linear(obs_dim, obs_dim)).double()

        # assume factorized prior of normal RVs
        self.prior = torch.distributions.Normal(torch.zeros(self.latent_dim), torch.ones(self.latent_dim))
        # self.max_epoch = 2800

    def encoder(self, x, mask, sample=True):
        x = self.seq_encoder(x.float() * mask)
        mean = self.q_mu(x)
        log_var = self.q_logstd(x)

        mean = mean.unsqueeze(0).expand(self.num_samples, mean.shape[0], mean.shape[1]).permute(1, 0,
                                                                                                2)  # shape [batch_size, n_samples, obs_dim]
        log_var = log_var.unsqueeze(0).expand(self.num_samples, log_var.shape[0], log_var.shape[1]).permute(1, 0,
                                                                                                            2)  # shape [batch_size, n_samples, obs_dim]
        std = torch.exp(log_var / 2)
        if sample:
            dist = torch.distributions.Normal(mean, std)
            z = dist.rsample()
        else:
            z = mean
        return z, mean, log_var

    def decoder(self, z_int):
        # z_int = z_int.reshape(-1, self.latent_dim)
        decoded = self.seq_decoder(z_int)
        x_mean = self.x_mean(decoded)
        x_logvar = self.x_logvar(decoded)
        return x_mean, x_logvar

    def loss(self, x, x_recon, x_logvar, mean, logvar, epoch, mask, vae_elbo=False, llh_eval=False,
             MI=False,
             beta_annealing=False, beta=1.0, stage='train', missing_process='selfmasking_known'):
        new_x = x[:, 0:].unsqueeze(0).expand(self.num_samples, x.shape[0], x.shape[1]).permute(1, 0, 2)
        new_mask = mask[:, 0:].unsqueeze(0).expand(self.num_samples, x.shape[0], x.shape[1]).permute(1, 0, 2)
        out_mixed = x_recon * (1 - new_mask) + new_x * new_mask
        if missing_process == 'selfmasking':
            logits = - self.W * (out_mixed - self.b)

        # the type of masking I use
        elif missing_process == 'selfmasking_known':
            # self.W = F.softplus(self.W)
            logits = - self.activation(self.W) * (out_mixed - self.b)

        else:  # linear
            logits = self.logits(out_mixed)
        RE = self.neg_gaussian_log_likelihood(new_x * new_mask, x_recon * new_mask,
                                              x_logvar * new_mask)  # sum over obs_dim

        q_z = torch.distributions.Normal(loc=mean, scale=torch.exp(logvar / 2))
        dist = torch.distributions.Normal(mean, torch.exp(logvar / 2))
        z = dist.rsample()
        log_q_z_given_x = torch.sum(q_z.log_prob(z), 2)  # sum over obs_dim

        prior = torch.distributions.Normal(loc=0.0, scale=1.0)
        log_p_z = torch.sum(prior.log_prob(z), 2)  # sum over obs_dim
        KL = log_q_z_given_x - log_p_z


        p_s_given_x = torch.distributions.Bernoulli(logits=logits)
        log_p_s_given_x = torch.sum(p_s_given_x.log_prob(new_mask), 2)  # sum over obs_dim
        l_w = RE + KL - log_p_s_given_x

        log_sum_w = torch.logsumexp(l_w, 1)  # sum over n_samples
        log_avg_weight = log_sum_w - math.log(float(self.num_samples))  # average over n_samples
        loss = torch.mean(log_avg_weight)  # mean over batch_size
        train_loss = loss
        print_loss = loss
        if llh_eval:
            wl = softmax(-l_w)
            xm = torch.sum((x_recon.T * wl.T).T, 1)
            return xm, train_loss, RE.mean()
        elif MI:
            aggregated_mean = torch.mean(mean, 0).mean(0).mean(0)
            aggregated_logvar = torch.mean(logvar, 0).mean(0).mean(0)
            KL_agg = self.kl_diagnormal_stdnormal2(aggregated_mean, aggregated_logvar)
            mut_inf = KL.mean() - KL_agg
            return print_loss, train_loss, mut_inf
        elif vae_elbo:
            return print_loss, train_loss
        else:
            return print_loss, train_loss

    def kl_diagnormal_stdnormal(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        # TODO instead of a sum, we had a mean in the next line. That should have been incorrect
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior), 2)

    def kl_diagnormal_stdnormal2(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        # TODO instead of a sum, we had a mean in the next line. That should have been incorrect
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior))

    def neg_gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(-dist.log_prob(targets), 2)

    def reparameterization(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data, mask, stage='train'):
        z, mean, logvar = self.encoder(data, mask)
        x_mean, x_logvar = self.decoder(z)
        return mean, logvar, x_mean, x_logvar


class notMIWAE(nn.Module):
    def __init__(self, obs_dim, hid_dim, K, latent_dim, training_parameters, num_samples, num_estimates):
        super(notMIWAE, self).__init__()
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.emb_dim = 10
        self.num_samples = num_samples
        self.num_estimates = num_estimates
        self.latent_dim = latent_dim
        self.batch_size = training_parameters['batch_size']
        self.K = K
        self.obs_std = 0.1
        self.number_components = 500
        self.training_paramters = training_parameters
        # encoder: q(z2 | x)
        self.seq_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            # nn.ELU(),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
            # nn.Linear(128, 2 * latent_dim)
            # nn.LeakyReLU()
        )
        self.q_mu = nn.Sequential(
            nn.Linear(128, latent_dim),
            # nn.Sigmoid()

        )
        self.q_logstd = nn.Sequential(
            nn.Linear(128, latent_dim),
            # lambda x: torch.clip(x, -10, 10)
            nn.Hardtanh(min_val=-10, max_val=10)  # in my version I don't do any clipping here
        )

        self.seq_decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.Tanh(),
                                         nn.Linear(128, 128), nn.Tanh())
        # self.x_logvar = torch.log(torch.square(torch.Tensor([0.1 * np.sqrt(2)])))
        self.x_mean = nn.Sequential(nn.Linear(128, obs_dim))
        self.x_std = nn.Sequential(nn.Linear(128, obs_dim), nn.Softplus())

        # self.x_logvar = nn.Sequential(nn.Linear(128, obs_dim), nn.Hardtanh(min_val=-10.0, max_val=0))

        # nn.Hardtanh(min_val=-10.0, max_val=0))

        emb1 = torch.empty([1, 1, self.obs_dim])
        nn.init.xavier_uniform_(emb1)
        self.W = nn.Parameter(emb1, requires_grad=True)
        emb2 = torch.empty([1, 1, self.obs_dim])
        nn.init.xavier_uniform_(emb2)
        self.b = nn.Parameter(emb2, requires_grad=True)
        self.activation = nn.Softplus()

        # assume factorized prior of normal RVs
        self.prior = torch.distributions.Normal(torch.zeros(self.latent_dim), torch.ones(self.latent_dim))
        # self.max_epoch = 2800

    def encoder(self, x, mask, sample=True):
        x = self.seq_encoder(x.float() * mask)
        mean = self.q_mu(x)
        log_var = self.q_logstd(x)

        mean = mean.unsqueeze(0).expand(self.num_samples, mean.shape[0], mean.shape[1]).permute(1, 0,
                                                                                                2)  # shape [batch_size, n_samples, obs_dim]
        log_var = log_var.unsqueeze(0).expand(self.num_samples, log_var.shape[0], log_var.shape[1]).permute(1, 0,
                                                                                                            2)  # shape [batch_size, n_samples, obs_dim]
        std = torch.exp(log_var / 2)
        if sample:
            dist = torch.distributions.Normal(mean, std)
            z = dist.rsample()
        else:
            z = mean
        return z, mean, log_var

    def decoder(self, z_int):
        decoded = self.seq_decoder(z_int)
        x_mean = self.x_mean(decoded)
        x_std = self.x_std(decoded)
        return x_mean, torch.log(torch.square(x_std))

    def loss(self, x, x_recon, x_logvar, mean, logvar, epoch, mask, vae_elbo=False, llh_eval=False,
             MI=False,
             beta_annealing=False, beta=1.0, stage='train', missing_process='selfmasking_known'):
        new_x = x[:, 0:].unsqueeze(0).expand(self.num_samples, x.shape[0], x.shape[1]).permute(1, 0, 2)
        new_mask = mask[:, 0:].unsqueeze(0).expand(self.num_samples, x.shape[0], x.shape[1]).permute(1, 0, 2)
        out_mixed = x_recon * (1 - new_mask) + new_x * new_mask
        if missing_process == 'selfmasking':
            logits = - self.W * (out_mixed - self.b)


        # the type of masking I use
        elif missing_process == 'selfmasking_known':
            # self.W = F.softplus(self.W)
            logits = - self.activation(self.W) * (out_mixed - self.b)

        else:  # linear
            logits = self.logits(out_mixed)
        RE = self.neg_gaussian_log_likelihood(new_x * new_mask, x_recon * new_mask,
                                              x_logvar * new_mask)  # sum over obs_dim

        # Different way of calculating of KL-divergence, but still gives the same result.
        q_z = torch.distributions.Normal(loc=mean, scale=torch.exp(logvar / 2))
        dist = torch.distributions.Normal(mean, torch.exp(logvar / 2))
        z = dist.rsample()
        log_q_z_given_x = torch.sum(q_z.log_prob(z), 2)  # sum over obs_dim

        prior = torch.distributions.Normal(loc=0.0, scale=1.0)
        log_p_z = torch.sum(prior.log_prob(z), 2)  # sum over obs_dim
        KL = log_q_z_given_x - log_p_z

        p_s_given_x = torch.distributions.Bernoulli(logits=logits)
        print(p_s_given_x.sample())
        log_p_s_given_x = torch.sum(p_s_given_x.log_prob(new_mask), 2)  # sum over obs_dim
        l_w = RE + KL - log_p_s_given_x

        log_sum_w = torch.logsumexp(l_w, 1)  # sum over n_samples
        log_avg_weight = log_sum_w - math.log(float(self.num_samples))  # average over n_samples
        loss = torch.mean(log_avg_weight)  # mean over batch_size

        train_loss = loss
        print_loss = loss
        if llh_eval:
            wl = softmax(-l_w)
            xm = torch.sum((x_recon.T * wl.T).T, 1)
            return xm, train_loss, RE.mean()
        elif MI:
            aggregated_mean = torch.mean(mean, 0).mean(0).mean(0)
            aggregated_logvar = torch.mean(logvar, 0).mean(0).mean(0)
            KL_agg = self.kl_diagnormal_stdnormal2(aggregated_mean, aggregated_logvar)
            mut_inf = KL.mean() - KL_agg
            return print_loss, train_loss, mut_inf
        elif vae_elbo:
            return print_loss, train_loss
        else:
            return print_loss, train_loss

    def kl_diagnormal_stdnormal(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        # TODO instead of a sum, we had a mean in the next line. That should have been incorrect
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior), 2)

    def kl_diagnormal_stdnormal2(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        # TODO instead of a sum, we had a mean in the next line. That should have been incorrect
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior))

    def neg_gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.sum(-dist.log_prob(targets), 2)

    def reparameterization(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data, mask):
        z, mean, logvar = self.encoder(data, mask)
        x_mean, x_logvar = self.decoder(z)
        return mean, logvar, x_mean, x_logvar


class MIWAE(nn.Module):
    def __init__(self, obs_dim, hid_dim, K, latent_dim, training_parameters, num_samples, num_estimates):
        super(MIWAE, self).__init__()
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.emb_dim = 10
        self.num_samples = num_samples
        self.num_estimates = num_estimates
        self.latent_dim = latent_dim
        self.batch_size = training_parameters['batch_size']
        self.K = K
        self.obs_std = 0.1
        self.number_components = 500
        self.training_paramters = training_parameters
        # encoder: q(z2 | x)
        self.seq_encoder = nn.Sequential(
            torch.nn.Linear(obs_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2 * latent_dim),  # the encoder will output both the mean and the diagonal covariance
        )

        self.seq_decoder = nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3 * obs_dim),
            # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*p)
        )

        # assume factorized prior of normal RVs
        self.prior = torch.distributions.Normal(torch.zeros(self.latent_dim), torch.ones(self.latent_dim))
        self.max_epoch = 2800

    def encoder(self, x, mask, sample=True):
        mean, out_enc2 = self.seq_encoder(x.float() * mask).chunk(2, dim=1)
        scale = nn.Softplus()(out_enc2)
        mean = mean.unsqueeze(0).expand(self.num_samples, mean.shape[0], mean.shape[1]).permute(1, 0,
                                                                                                2)  # shape [batch_size, n_samples, obs_dim]
        scale = scale.unsqueeze(0).expand(self.num_samples, scale.shape[0], scale.shape[1]).permute(1, 0,
                                                                                                    2)  # shape [batch_size, n_samples, obs_dim]
        if sample:
            dist = torch.distributions.Normal(mean, scale)
            z = dist.rsample()
        else:
            z = mean
        return z, mean, scale

    def decoder(self, z_int):
        mean, scale, deg_free = self.seq_decoder(z_int).chunk(3, dim=2)
        mean = nn.Sigmoid()(mean)
        scale = nn.Softplus()(scale) + 0.001
        deg_free = nn.Softplus()(deg_free) + 3
        return mean, scale, deg_free

    def loss(self, x, x_mean, x_scale, deg_free, mean, scale, mask, epoch, vae_elbo=False, llh_eval=False,
             MI=False,
             beta_annealing=True, beta=1.0, stage='train'):
        new_x = x[:, 0:].unsqueeze(0).expand(self.num_samples, x.shape[0], x.shape[1]).permute(1, 0, 2)
        new_mask = mask[:, 0:].unsqueeze(0).expand(self.num_samples, x.shape[0], x.shape[1]).permute(1, 0, 2)
        all_log_pxgivenz_flat = torch.distributions.StudentT(loc=x_mean.reshape([-1, 1]),
                                                             scale=x_scale.reshape([-1, 1]),
                                                             df=deg_free.reshape([-1, 1])).log_prob(
            new_x.reshape(-1, 1))

        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([self.num_samples * x.shape[0], self.obs_dim])

        logpxobsgivenz = torch.sum(all_log_pxgivenz * new_mask.reshape(-1, self.obs_dim), 1).reshape(
            [self.num_samples, x.shape[0]])

        logpxobsgivenz_imp = torch.sum(all_log_pxgivenz * ~new_mask.reshape(-1, self.obs_dim), 1).reshape(
            [self.num_samples, x.shape[0]])
        q_zgivenxobs = torch.distributions.Independent(
            torch.distributions.Normal(loc=mean, scale=scale), 1)
        zgivenx = q_zgivenxobs.rsample()
        p_z = torch.distributions.Independent(
            torch.distributions.Normal(loc=torch.zeros(self.latent_dim), scale=torch.ones(self.latent_dim)), 1)
        logpz = p_z.log_prob(zgivenx).permute(1, 0)
        logq = q_zgivenxobs.log_prob(zgivenx).permute(1, 0)
        neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq, 0))
        train_loss = neg_bound
        print_loss = neg_bound
        if llh_eval:
            imp_weights = F.softmax(logpxobsgivenz + logpz - logq, 0)
            x_mean = x_mean.permute(1, 0, 2)
            xm = torch.einsum('ki,kij->ij', imp_weights.float(), x_mean.float())
            return xm, train_loss, logpxobsgivenz_imp.sum() / (x.shape[0] * 5000)
        elif MI:
            aggregated_mean = torch.mean(mean, 0).mean(0).mean(0)
            aggregated_logvar = torch.mean(logvar, 0).mean(0).mean(0)

            KL_agg = self.kl_diagnormal_stdnormal2(aggregated_mean, aggregated_logvar)
            mut_inf = KL.mean() - KL_agg
            return print_loss, train_loss, mut_inf
        elif vae_elbo:
            return print_loss, train_loss
        else:
            return print_loss, train_loss

    def kl_diagnormal_stdnormal(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        # TODO instead of a sum, we had a mean in the next line. That should have been incorrect
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior), 3)

    def kl_diagnormal_stdnormal2(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        # TODO instead of a sum, we had a mean in the next line. That should have been incorrect
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior))

    def neg_gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.mean(-dist.log_prob(targets), 3)

    def reparameterization(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data, mask):
        z, mean, scale = self.encoder(data, mask)
        x_mean, x_scale, deg_free = self.decoder(z)
        return mean, scale, x_mean, x_scale, deg_free


class Reg_MIWAE(nn.Module):
    def __init__(self, obs_dim, hid_dim, K, latent_dim, training_parameters, num_samples, num_estimates):
        super(Reg_MIWAE, self).__init__()
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.emb_dim = 10
        self.num_samples = num_samples
        self.num_estimates = num_estimates
        self.latent_dim = latent_dim
        self.batch_size = training_parameters['batch_size']
        self.K = K
        self.obs_std = 0.1
        self.number_components = 500
        self.training_paramters = training_parameters
        # encoder: q(z2 | x)
        self.seq_encoder = nn.Sequential(
            torch.nn.Linear(obs_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2 * latent_dim),  # the encoder will output both the mean and the diagonal covariance
        )

        self.seq_decoder = nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3 * obs_dim),
            # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*p)
        )

        # self.x_logvar = nn.Sequential(nn.Linear(latent_dim, hid_dim), nn.LeakyReLU(), nn.Linear(hid_dim, obs_dim),
        #                              nn.Hardtanh(min_val=-10, max_val=0))
        # assume factorized prior of normal RVs
        self.prior = torch.distributions.Normal(torch.zeros(self.latent_dim), torch.ones(self.latent_dim))
        self.max_epoch = 2800

    def encoder(self, x, mask, sample=True):
        mean, out_enc2 = self.seq_encoder(x.float() * mask).chunk(2, dim=1)
        scale = nn.Softplus()(out_enc2)
        mean = mean.unsqueeze(0).expand(self.num_samples, mean.shape[0], mean.shape[1]).permute(1, 0,
                                                                                                2)  # shape [batch_size, n_samples, obs_dim]
        scale = scale.unsqueeze(0).expand(self.num_samples, scale.shape[0], scale.shape[1]).permute(1, 0,
                                                                                                    2)  # shape [batch_size, n_samples, obs_dim]
        if sample:
            # log_std = logvar / 2
            dist = torch.distributions.Normal(mean, scale)
            z = dist.rsample()
        else:
            z = mean
        return z, mean, scale

    def decoder(self, z_int):
        mean, scale, deg_free = self.seq_decoder(z_int).chunk(3, dim=2)
        mean = nn.Sigmoid()(mean)
        scale = nn.Softplus()(scale) + 0.001
        deg_free = nn.Softplus()(deg_free) + 3
        return mean, scale, deg_free

    def loss(self, x, x_mean_p, x_scale_p, deg_free_p, mean_p, scale_p, x_mean_q, x_scale_q, deg_free_q, mean_q,
             scale_q, mask, mask_p, epoch, vae_elbo=False, llh_eval=False,
             MI=False,
             beta_annealing=True, beta=1.0, alpha=1.0, stage='train'):
        new_x = x[:, 0:].unsqueeze(0).expand(self.num_samples, x.shape[0], x.shape[1]).permute(1, 0, 2)
        new_mask = mask[:, 0:].unsqueeze(0).expand(self.num_samples, x.shape[0], x.shape[1]).permute(1, 0, 2)
        new_mask_p = mask_p[:, 0:].unsqueeze(0).expand(self.num_samples, x.shape[0], x.shape[1]).permute(1, 0, 2)
        all_log_pxgivenz_flat_q = torch.distributions.StudentT(loc=x_mean_q.reshape([-1, 1]),
                                                               scale=x_scale_q.reshape([-1, 1]),
                                                               df=deg_free_q.reshape([-1, 1])).log_prob(
            new_x.reshape(-1, 1))

        all_log_pxgivenz_q = all_log_pxgivenz_flat_q.reshape([self.num_samples * x.shape[0], self.obs_dim])

        logpxobsgivenz_q = torch.sum(all_log_pxgivenz_q * new_mask.reshape(-1, self.obs_dim), 1).reshape(
            [self.num_samples, x.shape[0]])

        q_zgivenxobs_q = torch.distributions.Independent(
            torch.distributions.Normal(loc=mean_q, scale=scale_q), 1)
        zgivenx_q = q_zgivenxobs_q.rsample()

        p_z = torch.distributions.Independent(
            torch.distributions.Normal(loc=torch.zeros(self.latent_dim), scale=torch.ones(self.latent_dim)), 1)
        logpz_q = p_z.log_prob(zgivenx_q).permute(1, 0)
        logq_q = q_zgivenxobs_q.log_prob(zgivenx_q).permute(1, 0)

        neg_bound_q = -torch.mean(torch.logsumexp(logpxobsgivenz_q + logpz_q - logq_q, 0))

        all_log_pxgivenz_flat_p = torch.distributions.StudentT(loc=x_mean_p.reshape([-1, 1]),
                                                               scale=x_scale_p.reshape([-1, 1]),
                                                               df=deg_free_p.reshape([-1, 1])).log_prob(
            new_x.reshape(-1, 1))

        all_log_pxgivenz_p = all_log_pxgivenz_flat_p.reshape([self.num_samples * x.shape[0], self.obs_dim])

        logpxobsgivenz_p = torch.sum(all_log_pxgivenz_p * new_mask_p.reshape(-1, self.obs_dim), 1).reshape(
            [self.num_samples, x.shape[0]])

        q_zgivenxobs_p = torch.distributions.Independent(
            torch.distributions.Normal(loc=mean_p, scale=scale_p), 1)
        zgivenx_p = q_zgivenxobs_p.rsample()

        logpz_p = p_z.log_prob(zgivenx_p).permute(1, 0)
        logq_p = q_zgivenxobs_p.log_prob(zgivenx_p).permute(1, 0)

        neg_bound_p = -torch.mean(torch.logsumexp(logpxobsgivenz_p + logpz_p - logq_p, 0))

        reg_like = torch.sum(
            all_log_pxgivenz_q * new_mask.reshape(-1, self.obs_dim) * ~new_mask_p.reshape(-1, self.obs_dim), 1).reshape(
            [self.num_samples, x.shape[0]]).mean()

        KL_reg = self.kl_diagnormal_diagnormal(mean_q, scale_q, mean_p, scale_p).mean()

        loss = neg_bound_q + alpha * (
                KL_reg - neg_bound_q + neg_bound_p - reg_like)
        train_loss = loss
        print_loss = loss
        if llh_eval:
            imp_weights = F.softmax(logpxobsgivenz_q + logpz_q - logq_q, 0)
            x_mean = x_mean_q.permute(1, 0, 2)
            xm = torch.einsum('ki,kij->ij', imp_weights.float(), x_mean.float())
            return xm, train_loss, train_loss
        elif MI:
            aggregated_mean = torch.mean(mean, 0).mean(0).mean(0)
            aggregated_logvar = torch.mean(logvar, 0).mean(0).mean(0)
            KL_agg = self.kl_diagnormal_stdnormal2(aggregated_mean, aggregated_logvar)
            mut_inf = KL.mean() - KL_agg
            return print_loss, train_loss, mut_inf
        elif vae_elbo:
            return print_loss, train_loss
        else:
            return print_loss, train_loss

    def kl_diagnormal_diagnormal(self, mean1, scale1, mean2, scale2):
        # let dist1 be from reconstructed data
        # let dist2 be from the original data
        dist1 = torch.distributions.Normal(mean1, scale1)  # distribution Q
        dist2 = torch.distributions.Normal(mean2, scale2)  # distribution P
        return torch.distributions.kl_divergence(dist1, dist2)

    def kl_diagnormal_stdnormal(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        # TODO instead of a sum, we had a mean in the next line. That should have been incorrect
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior), 3)

    def kl_diagnormal_stdnormal2(self, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        # TODO instead of a sum, we had a mean in the next line. That should have been incorrect
        return torch.sum(torch.distributions.kl_divergence(dist, self.prior))

    def neg_gaussian_log_likelihood(self, targets, mean, log_var):
        dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
        return torch.mean(-dist.log_prob(targets), 3)

    def reparameterization(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data, mask, mask_p, stage='train'):
        z_q, mean_q, scale_q = self.encoder(data, mask)
        x_mean_q, x_scale_q, deg_free_q = self.decoder(z_q)
        z_p, mean_p, scale_p = self.encoder(data, mask_p)
        x_mean_p, x_scale_p, deg_free_p = self.decoder(z_p)
        return mean_p, scale_p, x_mean_p, x_scale_p, deg_free_p, mean_q, scale_q, x_mean_q, x_scale_q, deg_free_q
