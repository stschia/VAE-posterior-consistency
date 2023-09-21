import torch
import numpy as np
import argparse
import copy
from scipy.stats import bernoulli


def check(x, a, b):
    """
    Checks whether x falls inside the interval [a,b]
    """
    if a <= x <= b:
        return torch.BoolTensor([True])
    else:
        return torch.BoolTensor([False])


def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    eps = torch.randn_like(std)
    return mu + eps * std


def create_missing_toy(batch_size, missing_rate):
    missing_rate = missing_rate / 100
    rng = np.random.default_rng(seed=1234)
    given = rng.permutation(batch_size)
    given = given[:int(np.ceil(batch_size * (1. - missing_rate)))]
    mask = np.zeros((batch_size, 2), dtype=np.bool)
    mask[:, 0] = True
    mask[given, 1] = True
    # print(given)
    return torch.from_numpy(mask)


def create_missing_uci(shape, missing_rate):
    missing_rate = missing_rate / 100
    missing_mask = np.random.rand(*shape) < (1 - missing_rate)
    return torch.from_numpy(missing_mask)


def create_missing_uci_drop_eddi(shape):
    temp = np.minimum(np.random.rand(*shape), 0.99)
    mask_drop = bernoulli.rvs(1 - temp)
    return torch.from_numpy(mask_drop)


def introduce_mising_mnar_based_on_mean_half_features(X):
    N, D = X.shape
    Xnan = copy.deepcopy(X)
    mask = torch.ones_like(X)
    # ---- MNAR in D/2 dimensions
    mean = torch.mean(Xnan[:, :int(D / 2)], 0)
    ix_larger_than_mean = Xnan[:, :int(D / 2)] > mean
    Xnan[:, :int(D / 2)][ix_larger_than_mean] = np.nan

    # Xz = Xnan.copy()
    mask[np.isnan(Xnan)] = 0

    return mask == 1.0


def introduce_mising_mnar_based_on_mean_all_features(X):
    N, D = X.shape
    Xnan = copy.deepcopy(X)
    mask = torch.ones_like(X)
    # ---- MNAR in D/2 dimensions
    mean = torch.mean(Xnan[:, :], 0)
    ix_larger_than_mean = Xnan[:, :] > mean
    Xnan[:, :][ix_larger_than_mean] = np.nan

    # Xz = Xnan.copy()
    mask[np.isnan(Xnan)] = 0

    return mask == 1.0


def introduce_mising_mnar_based_on_variance_all_features(X):
    N, D = X.shape
    Xnan = copy.deepcopy(X)
    mask = torch.ones_like(X)
    # ---- MNAR in D/2 dimensions
    var = torch.var(Xnan[:, :], 0)
    ix_larger_than_mean = Xnan[:, :] > var
    Xnan[:, :][ix_larger_than_mean] = np.nan

    # Xz = Xnan.copy()
    mask[np.isnan(Xnan)] = 0

    return mask == 1.0


def introduce_mising_mnar_based_on_variance_half_features(X):
    N, D = X.shape
    Xnan = copy.deepcopy(X)
    mask = torch.ones_like(X)
    # ---- MNAR in D/2 dimensions
    var = torch.var(Xnan[:, :int(D / 2)], 0)
    ix_larger_than_mean = Xnan[:, :int(D / 2)] > var
    Xnan[:, :int(D / 2)][ix_larger_than_mean] = np.nan

    # Xz = Xnan.copy()
    mask[np.isnan(Xnan)] = 0

    return mask == 1.0


def introduce_mising(X):
    N, D = X.shape
    Xnan = copy.deepcopy(X)
    mask = torch.ones_like(X)
    # ---- MNAR in D/2 dimensions
    mean = torch.mean(Xnan[:, :int(D / 2)], 0)
    ix_larger_than_mean = Xnan[:, :int(D / 2)] > mean
    Xnan[:, :int(D / 2)][ix_larger_than_mean] = np.nan

    # Xz = Xnan.copy()
    mask[np.isnan(Xnan)] = 0

    return mask


# def create_missing_uci_v2(shape, missing_rate):
#    missing_rate = missing_rate / 100
#    missing_mask = np.random.rand(*shape) < (1 - missing_rate)
#    return torch.from_numpy(missing_mask)

# from lxuechen
def log_mean_exp(x):
    max_, _ = torch.max(x, -1, keepdim=True)
    # print(x.shape)
    # print(max_.shape)
    # print( torch.log(torch.mean(torch.exp(x - max_), -1)))
    return torch.log(torch.mean(torch.exp(x - max_), -1)) + torch.squeeze(max_)


def logsumexp(x):
    max_x = torch.max(x, 0)[0]
    new_x = x - max_x.unsqueeze(1).expand_as(x)
    return max_x + (new_x.exp().sum(0)).log()


def logsumexp_2(x):
    max_x = torch.max(x, -1, keepdim=True)[0]
    new_x = x - max_x.unsqueeze(1).expand_as(x)
    return max_x + (new_x.exp().sum(0)).log()


def neg_gaussian_log_likelihood(targets, mean, log_var):
    dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
    return torch.sum(-dist.log_prob(targets), 1)


def gaussian_log_likelihood(targets, mean, log_var):
    dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
    return torch.sum(dist.log_prob(targets), 1, keepdim=True)


def gaussian_log_likelihood_2(targets, mean, log_var):
    dist = torch.distributions.Normal(mean, torch.exp(log_var / 2.))
    return torch.sum(dist.log_prob(targets), 3, keepdim=True)


# stackoverflow Maxim
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# stackoverflow
def setup_parser(arguments, title):
    parser = argparse.ArgumentParser(
        description=title,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    for key, value in arguments.items():
        parser.add_argument(
            '-%s' % key,
            type=type(value["default"]),
            help=value["help"],
            default=value["default"]
        )
    return parser


def completion(x, mask, mask_p, M, model):
    '''
    function to generate new samples conditioned on observations
    :param x: underlying partially observed data
    :param mask: mask of missingness
    :param M: number of MC samples
    :param vae: a pre-trained vae
    :return: sampled missing data, a M by N by D matrix, where M is the number of samples.
    '''

    im = torch.zeros((M, x.shape[0], x.shape[1]))
    for m in range(M):
        # tf.reset_default_graph()
        # np.random.seed(42 + m)  ### added for bar plots only
        _, _, _, _, _, _, x_mean_q, _ = model.forward(x, mask, mask_p)
        im[m, :, :] = x_mean_q
    return im
