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

    res = np.zeros(observable.shape)
    with torch.no_grad():
        mean   = observable[j,:]
        mguass = MultivariateNormal(loc=mean,
                                    covariance_matrix=covariance_matrix.double())
        res[j,:] = mgauss.sample(sample_shape=([batch_size,]))


        if batch_size > 1:
            try:
                result = np.hstack(SMF_tmp)
            except:
                pass
        else:
            try:
                result = np.vstack(SMF_tmp)
                result = result.flatten()
            except:
                pass
        result = torch.tensor(result, dtype=torch.float32)
        return result
