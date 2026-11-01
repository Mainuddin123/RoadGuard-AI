
import random
from PIL import ImageEnhance


def augment_image(image, boxes):
    """
    Apply image augmentation and update bounding boxes.
    Boxes format: [xmin, ymin, xmax, ymax]
    """

    boxes = boxes.clone()

    width, height = image.size

    # Random horizontal flip.
    if random.random() < 0.5:
        image = image.transpose(
            ImageTranspose.FLIP_LEFT_RIGHT
        )

        old_xmin = boxes[:, 0].clone()
        old_xmax = boxes[:, 2].clone()

        boxes[:, 0] = width - old_xmax
        boxes[:, 2] = width - old_xmin

    # Mild brightness adjustment.
    brightness = random.uniform(0.85, 1.15)
    image = ImageEnhance.Brightness(image).enhance(brightness)

    # Mild contrast adjustment.
    contrast = random.uniform(0.85, 1.15)
    image = ImageEnhance.Contrast(image).enhance(contrast)

    return image, boxes


# PIL transpose constant
from PIL import Image
ImageTranspose = Image.Transpose
