import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.determinstic = True
    np.random.seed(seed)
    random.seed(seed)
