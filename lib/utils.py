import torch


class Colors:
    ENDC = '\033[0m'
    GRAY = '\033[90m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\33[96m'


class Formats:
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'


def print_style(msg, color=None, formatting=None):
    """ Print to terminal with colour and formatting style,

    params:
        msg: Message to print
        color: string with color for msg. Supported values {'GRAY','RED','GREEN','YELLOW','BLUE','MAGENTA','CYAN'}
        formatting: list of strings with formatting options. Supported values: {'BOLD','ITALIC','UNDERLINE'}
    """

    # Add color
    if color == 'GRAY':
        msg = Colors.GRAY + msg + Colors.ENDC
    elif color == 'RED':
        msg = Colors.RED + msg + Colors.ENDC
    elif color == 'GREEN':
        msg = Colors.GREEN + msg + Colors.ENDC
    elif color == 'YELLOW':
        msg = Colors.YELLOW + msg + Colors.ENDC
    elif color == 'BLUE':
        msg = Colors.BLUE + msg + Colors.ENDC
    elif color == 'MAGENTA':
        msg = Colors.MAGENTA + msg + Colors.ENDC
    elif color == 'CYAN':
        msg = Colors.CYAN + msg + Colors.ENDC
    else:
        msg = msg

    # Add formatting
    if formatting:
        if "BOLD" in formatting:
            msg = Formats.BOLD + msg + Formats.ENDC
        if "ITALIC" in formatting:
            msg = Formats.ITALIC + msg + Formats.ENDC
        if "UNDERLINE" in formatting:
            msg = Formats.UNDERLINE + msg + Formats.ENDC

    print(msg)


def select_device(cuda_device, verbose=True):
    if torch.cuda.is_available():
        if verbose:
            print_style('CUDA (' + cuda_device + ') device selected.', color='CYAN', formatting="ITALIC")
        return torch.device(cuda_device)
    elif torch.backends.mps.is_available():
        if verbose:
            print_style("MPS device selected.", color='CYAN', formatting="ITALIC")
        return torch.device("mps")  # For M1 Macs
    else:
        if verbose:
            print_style("CPU device selected.", color='CYAN', formatting="ITALIC")
        return torch.device('cpu')
