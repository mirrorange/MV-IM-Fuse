import torch


def load_local_checkpoint(checkpoint_path, map_location='cpu'):
    """Load a trusted project checkpoint across PyTorch versions."""
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)