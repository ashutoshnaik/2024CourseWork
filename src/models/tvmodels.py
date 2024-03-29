# Copyright (c) EEEM071, University of Surrey

import torch.nn as nn
import torchvision.models as tvmodels
from torchvision.models import ViT_B_16_Weights

__all__ = ["mobilenet_v3_small", "vgg16", "vit_b_16", "maxvit_t"]


class TorchVisionModel(nn.Module):
    def __init__(self, name, num_classes, loss, pretrained, **kwargs):
        super().__init__()

        if name == "vit_b_16":
            print("vit_b_16")
            self.backbone = tvmodels.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            self.feature_dim = self.backbone.head.in_features
            # Overwrite the head for custom num_classes
            self.backbone.head = nn.Linear(self.feature_dim, num_classes)
        else:
            print("Some other CNN")
            self.backbone = tvmodels.__dict__[name](pretrained=pretrained)
            self.feature_dim = self.backbone.classifier[0].in_features
            # overwrite the classifier used for ImageNet pretrianing
            # nn.Identity() will do nothing, it's just a place-holder
            self.backbone.classifier = nn.Identity()
            self.classifier = nn.Linear(self.feature_dim, num_classes)

        self.loss = loss

    def forward(self, x):
        v = self.backbone(x)

        if not self.training:
            return v

        y = self.classifier(v)

        if self.loss == {"xent"}:
            return y
        elif self.loss == {"xent", "htri"}:
            return y, v
        else:
            raise KeyError(f"Unsupported loss: {self.loss}")


def vgg16(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "vgg16",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model



def mobilenet_v3_small(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "mobilenet_v3_small",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model


# Define any models supported by torchvision bellow
# https://pytorch.org/vision/0.11/models.html

def vit_b_16(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "vit_b_16",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model

def maxvit_t(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "maxvit_t",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model

