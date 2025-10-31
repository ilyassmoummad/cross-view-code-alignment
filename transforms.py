from PIL import Image, ImageFilter, ImageOps
import torchvision.transforms as T
import numpy as np


class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: a solarized image.
        """

        return ImageOps.solarize(img)
    

class GBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


def IN_normalize_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return T.Normalize(mean=mean, std=std)


class SSLAugment(object):
    def __init__(self, crop_size=224, scale_min=0.2, num_views=2):
    
        self.num_views = num_views
        self.crop_size = crop_size
        self.scale_min = scale_min

    def __call__(self, x):
            
        aug_transform = T.Compose([
            T.RandomResizedCrop(self.crop_size, scale=(self.scale_min, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            T.RandomGrayscale(p=0.2),
            GBlur(p=0.1),
            T.RandomApply([Solarization()], p=0.1),
            T.ToTensor(),  
            IN_normalize_transform(),
        ])
     
        return [aug_transform(x) for i in range(self.num_views)]
    

class EvalTransform(object):
    def __init__(self, resize_size=256, crop_size=224):
        
        self.resize_size = resize_size
        self.crop_size = crop_size

    def __call__(self, x):
            
        val_transform = T.Compose([
            T.Resize(self.resize_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(self.crop_size),
            T.ToTensor(),
            IN_normalize_transform(),
        ])
        
        return val_transform(x)
    

def setup_transforms(args):
    train_transform = SSLAugment(crop_size=args.crop_size, scale_min=args.scale_min, num_views=args.n_views)
    eval_transform = EvalTransform(resize_size=args.resize_size, crop_size=args.crop_size)
    return train_transform, eval_transform