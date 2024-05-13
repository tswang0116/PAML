from .models.alexnet import load_model as alexnet_load_model
from .models.vgg16 import load_model as vgg16_load_model


def load_model(arch, code_length):
    """
    Load CNN model.

    Args
        arch(str): CNN model name.
        code_length(int): Hash code length.

    Returns
        model(torch.nn.Module): CNN model.
    """
    if arch == 'alexnet':
        model = alexnet_load_model(code_length)
    elif arch == 'vgg16':
        model = vgg16_load_model(code_length)
    else:
        raise ValueError('Invalid cnn model name!')

    return model

