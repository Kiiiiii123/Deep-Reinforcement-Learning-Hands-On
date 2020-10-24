import numpy as np
import torch
import torch.nn as nn
import warnings
from datetime import timedelta, datetime
from types import SimpleNamespace
from typing import Iterable, Tuple, List


HYPERPARAMS = {
    'pong': SimpleNamespace(**{
        'env_name':         'PongNoFrameskip-v4',
        'stop_reward':      18.0,
        'run_name':         'pong',
        'replay_size':      100000,
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.2,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32
    }),
}

