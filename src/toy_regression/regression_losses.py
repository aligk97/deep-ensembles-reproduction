from math import log, pi

import torch


def mse_loss(mean, target):
    return (target - mean).square().mean()


def gaussian_log_prob(mean, variance, target):
    return -0.5 * (log(2 * pi) + variance.log() + (target - mean).square() / variance)


def gaussian_nll(mean, variance, target):
    return -gaussian_log_prob(mean, variance, target).mean()


def mixture_nll(member_means, member_variances, target):
    log_prob = gaussian_log_prob(member_means, member_variances, target)
    return -(torch.logsumexp(log_prob, dim=0) - log(len(member_means))).mean()
