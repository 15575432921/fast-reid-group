from .build import BACKBONE_REGISTRY
from efficientnet_pytorch import EfficientNet
import  torch.nn as nn
@BACKBONE_REGISTRY.register()
def build_efficientnet_backbone(cfg):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    model = EfficientNet.from_pretrained('efficientnet-b0')
    # model._fc = nn.Sequential()
    # return model.extract_features
    return model