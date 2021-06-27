import torchvision.transforms as transforms
from PIL import Image


class BaseAugmentation:
    # Only Resize
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
        ])

    def __call__(self, image):
        img = self.transform(image)
        return img


class CenterCropAugmentation(BaseAugmentation):
    def __init__(self):
        super(CenterCropAugmentation, self).__init__()
        self.transform = transforms.Compose([
            transforms.CenterCrop(384),
            transforms.Resize((224, 224), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
        ])

class EffNetAugmentation(BaseAugmentation):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
        ])

class ResNetAugmentation(BaseAugmentation):
    def __init__(self):
        super(ResNetAugmentation, self).__init__()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), Image.BILINEAR),
            transforms.ToTensor(),
        ])