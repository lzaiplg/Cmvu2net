from .u2cracknet import U2CrackNet
from .bisenetv2 import BiSeNetV2
from .deeplabv3p import DeepLabV3P
from .unet import UNet
from .hrsegnet import HrSegNetB32
from .cmvu2net import cmvu2net
from .u2netp import U2NETP
from .DeepCrack import DeepCrack
from .CrackSegFormer import CrackSegFormer
__all__ = [
    'U2CrackNet',
    'BiSeNetV2',
    'DeepLabV3P',
    'UNet',
    'HrSegNetB32',
    'cmvu2net',
    'U2NETP',
    'DeepCrack',
    'CrackSegFormer',
]
def get_model(model_name, **kwargs):
    if model_name == 'U2CrackNet':
        return U2CrackNet(**kwargs)
    elif model_name == 'BiSeNetV2':
        return BiSeNetV2(**kwargs)
    elif model_name == 'DeepLabV3P':
        return DeepLabV3P(**kwargs)
    elif model_name == 'UNet':
        return UNet(**kwargs)
    elif model_name == 'HrSegNetB32':
        in_channels = kwargs.pop('in_channels', 3)
        num_classes = kwargs.pop('num_classes', 2)
        pretrained = kwargs.pop('pretrained', None)
        return HrSegNetB32(in_channels=in_channels, num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'cmvu2net':
        in_channels = kwargs.pop('in_channels', 3)
        num_classes = kwargs.pop('num_classes', 2)
        pretrained = kwargs.pop('pretrained', None)
        return cmvu2net(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)
    elif model_name == 'U2NETP' or model_name == 'u2netp':
        in_channels = kwargs.pop('in_channels', 3)
        num_classes = kwargs.pop('num_classes', 2)
        return U2NETP(in_ch=in_channels, out_ch=num_classes)
    elif model_name == 'DeepCrack':
        in_channels = kwargs.pop('in_channels', 3)
        num_classes = kwargs.pop('num_classes', 2)
        return DeepCrack(num_classes=num_classes, in_channels=in_channels)
    elif model_name == 'CrackSegFormer':
        in_channels = kwargs.pop('in_channels', 3)
        num_classes = kwargs.pop('num_classes', 2)
        backbone = kwargs.pop('backbone', 'mit_b0')
        pretrained = kwargs.pop('pretrained', True)
        return CrackSegFormer(num_classes=num_classes, in_channels=in_channels, backbone=backbone, pretrained=pretrained)
    else:
        raise ValueError(f'Unknown model: {model_name}')
