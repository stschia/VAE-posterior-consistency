# Taken from lxuechens folder inference-suboptimality

import math
import numpy.linalg as linalg
import sys
import os
import numpy as np
import time
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import grad as torchgrad
from src.utils.utils import neg_gaussian_log_likelihood
from src.utils.loaders import model_loader
import pickle


def linear_schedule(T):
    return np.linspace(0., 1., T)


def log_mean_exp(x):
    max_, _ = torch.max(x, 1, keepdim=True)
    return torch.log(torch.mean(torch.exp(x - max_), 1)) + torch.squeeze(max_)


def safe_repeat(x, n):
    return x.repeat(n, *[1 for _ in range(len(x.size()) - 1)])


def log_normal(x, mean=None, logvar=None, device=torch.device("cpu")):
    """Implementation WITHOUT constant, since the constants in p(z)
    and q(z|x) cancels out.
    Args:
        x: [B,Z]
        mean,logvar: [B,Z]
    Returns:
        output: [B]
    """
    if mean is None:
        mean = Variable(torch.zeros(x.size(), device=device).type(type(x.data)))
    if logvar is None:
        logvar = Variable(torch.zeros(x.size(), device=device).type(type(x.data)))

    return -0.5 * (logvar.sum(1) + ((x - mean).pow(2) / torch.exp(logvar)).sum(1))


def log_bernoulli(logit, target):
    """
    Args:
        logit:  [B, X, ?, ?]
        target: [B, X, ?, ?]

    Returns:
        output:      [B]
    """
    loss = -F.relu(logit) + torch.mul(target, logit) - torch.log(1. + torch.exp(-logit.abs()))
    while len(loss.size()) > 1:
        loss = loss.sum(-1)

    return loss


def sigmoidial_schedule(T, delta=4):
    """From section 6 of BDMC paper."""

    def sigmoid(x):
        return np.exp(x) / (1. + np.exp(x))

    def beta_tilde(t):
        return sigmoid(delta * (2. * t / T - 1.))

    def beta(t):
        return (beta_tilde(t) - beta_tilde(1)) / (beta_tilde(T) - beta_tilde(1))

    return [beta(t) for t in range(1, T + 1)]


def eval_ais(train_loader, valid_loader, test_loader, obs_dim, hid_dim, K, latent_dim, missing_rate, data_type,
             training_parameters, max_epochs, vae_type, num_samples, num_estimates,
             mode='forward',
             schedule=np.linspace(0., 1., 500),
             n_sample=100,
             device=torch.device("cpu")):
    for loader in [train_loader, valid_loader, test_loader]:
        loader, stage = loader
        ais_trajectory(
            loader, obs_dim, hid_dim, K, latent_dim, missing_rate, data_type,
            training_parameters, max_epochs, vae_type, stage, num_samples, num_estimates,
            mode=mode, schedule=schedule, n_sample=n_sample, device=device)


def ais_trajectory(
        loader, obs_dim, hid_dim, K, latent_dim, missing_rate, data_type,
        training_parameters, max_epochs, vae_type, stage, num_samples, num_estimates,
        mode='forward',
        schedule=np.linspace(0., 1., 500),
        n_sample=100,
        device=torch.device("cpu")
):
    """Compute annealed importance sampling trajectories for a batch of data.
    Could be used for *both* forward and reverse chain in bidirectional Monte Carlo
    (default: forward chain with linear schedule).
    Args:
        model (vae.VAE): VAE model
        loader (iterator): iterator that returns pairs, with first component being `x`,
            second would be `z` or label (will not be used)
        mode (string): indicate forward/backward chain; must be either `forward` or
            'backward' schedule (list or 1D np.ndarray): temperature schedule,
            i.e. `p(z)p(x|z)^t`; foward chain has increasing values, whereas
            backward has decreasing values
        n_sample (int): number of importance samples (i.e. number of parallel chains
            for each datapoint)
    Returns:
        A list where each element is a torch.autograd.Variable that contains the
        log importance weights for a single batch of data
    """

    model = model_loader('test', obs_dim, hid_dim, K, latent_dim, missing_rate, data_type,
                         training_parameters, max_epochs, num_samples, num_estimates, vae_type)
    model.to(device)
    assert mode == 'forward' or mode == 'backward', 'Should have forward/backward mode'

    def log_f_i(z, data, t, log_likelihood_fn=neg_gaussian_log_likelihood):
        """Unnormalized density for intermediate distribution `f_i`:
            f_i = p(z)^(1-t) p(x,z)^(t) = p(z) p(x|z)^t
        =>  log f_i = log p(z) + t * log p(x|z)
        """
        # logp1 = log_normal(z, mean, logvar)
        zeros = Variable(torch.zeros(B, z_size, device=device))
        # print(zeros)
        log_prior = log_normal(z, zeros, zeros, device=device)
        # print(log_prior.shape)
        mean, logvar = model.decoder(z)
        # print(data.shape)
        # print(logvar.shape)
        log_likelihood = log_likelihood_fn(data, mean, logvar)
        # print(log_likelihood.shape)
        return log_prior + log_likelihood.mul_(t)

    model.eval()

    # shorter aliases
    z_size = model.latent_dim
    mdtype = torch.FloatTensor

    _time = time.time()
    logws = []  # for output
    latents = []
    # means = []
    # variances = []
    print('In %s mode' % mode)

    for i, (batch, post_z) in enumerate(loader):
        fixed_batch_size = batch.detach().size(0)
        B = batch.size(0) * n_sample
        batch = Variable(batch.type(mdtype))
        batch = batch.to(device) # TODO
        batch = safe_repeat(batch, n_sample)

        # batch of step sizes, one for each chain
        epsilon = Variable(torch.ones(B, device=device)).mul_(0.01)
        # accept/reject history for tuning step size
        accept_hist = Variable(torch.zeros(B, device=device))
        # record log importance weight; volatile=True reduces memory greatly
        logw = Variable(torch.zeros(B, device=device), volatile=True)

        # initial sample of z
        if mode == 'forward':
            current_z = Variable(torch.randn(B, z_size).type(mdtype), requires_grad=True)
        else:
            current_z = Variable(safe_repeat(post_z, n_sample).type(mdtype), requires_grad=True)
        current_z = current_z.to(device) # TODO do it immediately

        # temp = current_z.detach().clone()

        for j, (t0, t1) in tqdm(enumerate(zip(schedule[:-1], schedule[1:]), 1), total=(len(schedule)-1)):
            # update log importance weight
            log_int_1 = log_f_i(current_z, batch, t0)
            log_int_2 = log_f_i(current_z, batch, t1)
            logw.add_(log_int_2 - log_int_1)

            # resample speed
            current_v = Variable(torch.randn(current_z.size(), device=device))

            def U(z):
                return -log_f_i(z, batch, t1)

            def grad_U(z):
                # grad w.r.t. outputs; mandatory in this case
                grad_outputs = torch.ones(B, device=device)
                # torch.autograd.grad default returns volatile
                grad = torchgrad(U(z), z, grad_outputs=grad_outputs)[0]
                # avoid humongous gradients
                grad = torch.clamp(grad, -10000, 10000)
                # needs variable wrapper to make differentiable
                grad = Variable(grad.data, requires_grad=True)
                return grad

            def normalized_kinetic(v):
                zeros = Variable(torch.zeros(B, z_size, device=device))
                # this is superior to the unnormalized version
                return -log_normal(v, zeros, zeros)

            z, v = hmc_trajectory(current_z, current_v, U, grad_U, epsilon)
            # print(current_z)
            # print(temp)
            # accept-reject step
            current_z, epsilon, accept_hist = accept_reject(
                current_z, current_v,
                z, v,
                epsilon,
                accept_hist, j,
                U, K=normalized_kinetic,
                device=device
            )

        # IWAE lower bound
        logw = log_mean_exp(logw.view(n_sample, -1).transpose(0, 1))
        if mode == 'backward':
            logw = -logw
        logws.append(logw.mean())
        # print(current_z.detach().shape)
        latents.append(current_z.detach().reshape(fixed_batch_size, n_sample, z_size))
        print('Time elapse %.4f, last batch stats %.4f' % \
              (time.time() - _time, logw.mean().cpu().data.numpy()))
        _time = time.time()
        sys.stdout.flush()  # for debugging
    torch.save(torch.stack(logws).mean(), 'experiments/' + vae_type + '/' + data_type + '/elbos/' + str(
        missing_rate) + '_missing/' + str(max_epochs) + '_epochs/' + stage + '_ais.pt')
    torch.save(torch.cat(latents, 0), 'experiments/' + vae_type + '/' + data_type + '/latents/' + str(
                   missing_rate) + '_missing/' + str(max_epochs) + '_epochs/' + stage + '_ais_true_latents.pt')
    return logws


def hmc_trajectory(current_z, current_v, U, grad_U, epsilon, L=10):
    """This version of HMC follows https://arxiv.org/pdf/1206.1901.pdf.
    Args:
        U: function to compute potential energy/minus log-density
        grad_U: function to compute gradients w.r.t. U
        epsilon: (adaptive) step size
        L: number of leap-frog steps
        current_z: current position
    """

    # as of `torch-0.3.0.post4`, there still is no proper scalar support
    assert isinstance(epsilon, Variable)

    eps = epsilon.view(-1, 1)
    z = current_z
    v = current_v - grad_U(z).mul(eps).mul_(.5)

    for i in range(1, L + 1):
        z = z + v.mul(eps)
        if i < L:
            v = v - grad_U(z).mul(eps)

    v = v - grad_U(z).mul(eps).mul_(.5)
    v = -v  # this is not needed; only here to conform to the math

    return z.detach(), v.detach()


def accept_reject(current_z, current_v,
                  z, v,
                  epsilon,
                  accept_hist, hist_len,
                  U, K=lambda v: torch.sum(v * v, 1),
                  device=torch.device("cpu")):
    """Accept/reject based on Hamiltonians for current and propose.
    Args:
        current_z: position BEFORE leap-frog steps
        current_v: speed BEFORE leap-frog steps
        z: position AFTER leap-frog steps
        v: speed AFTER leap-frog steps
        epsilon: step size of leap-frog.
                (This is only needed for adaptive update)
        U: function to compute potential energy (MINUS log-density)
        K: function to compute kinetic energy (default: kinetic energy in physics w/ mass=1)
    """

    mdtype = type(current_z.data)

    current_Hamil = K(current_v) + U(current_z)
    propose_Hamil = K(v) + U(z)

    prob = torch.exp(current_Hamil - propose_Hamil)
    uniform_sample = torch.rand(prob.size(), device=device)
    uniform_sample = Variable(uniform_sample)
    accept = 1. * (prob > uniform_sample)
    z = z.mul(accept.view(-1, 1)) + current_z.mul(1. - accept.view(-1, 1))

    accept_hist = accept_hist.add(accept)
    criteria = 1. * (accept_hist / hist_len > 0.65)
    adapt = 1.02 * criteria + 0.98 * (1. - criteria)
    epsilon = epsilon.mul(adapt).clamp(1e-4, .5)

    # clear previous history & save memory, similar to detach
    z = Variable(z.data, requires_grad=True)
    epsilon = Variable(epsilon.data)
    accept_hist = Variable(accept_hist.data)

    return z, epsilon, accept_hist
