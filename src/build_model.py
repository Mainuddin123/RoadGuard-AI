
import torch

from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn


NUM_CLASSES = 4  # Background + pothole + crack + manhole


def build_model(model_type="scratch"):

    if model_type == "scratch":

        # Model 1: Random initialization
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=NUM_CLASSES,
            min_size=320,
            max_size=640
        )

    elif model_type == "pretrained":

        # Model 2: ImageNet-pretrained MobileNetV3 backbone
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=None,
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
            num_classes=NUM_CLASSES,
            min_size=320,
            max_size=640
        )

    else:
        raise ValueError(
            "model_type must be 'scratch' or 'pretrained'"
        )

    return model


def count_parameters(model):

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


if __name__ == "__main__":

    print("===== ROADGUARD AI =====")

    model = build_model("scratch")

    total, trainable = count_parameters(model)

    print("Architecture: Faster R-CNN MobileNetV3-Large-FPN")
    print("Initialization: Random")
    print("Number of classes:", NUM_CLASSES)
    print("Total parameters:", f"{total:,}")
    print("Trainable parameters:", f"{trainable:,}")

    print("\nModel created successfully.")
