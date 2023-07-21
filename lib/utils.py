import torch


def select_device():
    if torch.backends.mps.is_available():
        print("MPS device selected.")
        return torch.device("mps")  # For M1 Macs
    elif torch.cuda.is_available():
        print("CUDA device selected.")
        return torch.device("cuda:0")
    else:
        print("CPU device selected.")
        return torch.device('cpu')
