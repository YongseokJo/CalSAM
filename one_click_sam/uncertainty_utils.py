import copy
import torch
import sys,os
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


def add_uncertainty(observable, covariance_matrix, batch_size=1):
    with torch.no_grad():
        if batch_size > 1:
            res = np.zeros(observable.shape)
            for j in range(batch_size):
                mean   = torch.tensor(observable[j,:])
                mgauss = MultivariateNormal(loc=mean,
                                            covariance_matrix=covariance_matrix.double())
                res[j,:] = mgauss.sample(sample_shape=([1,]))
        else:
            mean   = torch.tensor(observable)
            mgauss = MultivariateNormal(loc=mean,
                                        covariance_matrix=covariance_matrix.double())
            res    = mgauss.sample(sample_shape=([1,]))
    return res
