import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        import networkit as nk
        nk.setSeed(seed, False)
    except Exception:
        # Some environments have networkit installed but binary-incompatible
        # with the active NumPy build. Seeding should not fail non-networkit runs.
        pass

def set_seed_from_args(args):
    if hasattr(args, 'seed'):
        set_seed(args.seed)
    else:
        set_seed(42)
