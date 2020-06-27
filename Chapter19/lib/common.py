import ptan
import numpy as np
import torch


def unpack_batch_a2c(batch, net, last_val_gamma, device='cpu'):
