import torch
import torchvision.models as models
from torchvision import transforms

def load_backbone(backbone_name: str, pretrained: bool = True):
    """
    Loads a pre-trained backbone model.
    Supported: 'mobilenet_v3_small', 'resnet18'.
    """
    if backbone_name == 'mobilenet_v3_small':
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        # Remove classification head to get features
        model.classifier = torch.nn.Identity()
    elif backbone_name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        # Remove fc layer to get features
        model.fc = torch.nn.Identity()
    else:
        raise ValueError(f"Backbone '{backbone_name}' not supported.")

    model.eval()
    return model

def get_transform(input_size: int = 224):
    """
    Returns standard ImageNet preprocessing transforms.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
