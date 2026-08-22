import glob
import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassRecall,
    MulticlassJaccardIndex,
    MulticlassCohenKappa,
)


class PairCompose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, img, mask):
        for transform in self.transforms:
            img, mask = transform(img, mask)
        return img, mask


class ToTensor01:
    def __call__(self, img, mask):
        return TF.to_tensor(img).clamp(0.0, 1.0), mask


class ResizePair:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __call__(self, img, mask):
        img = TF.resize(img, self.size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)
        return img.clamp(0.0, 1.0), mask


class LoveDASampleTransform:
    def __init__(self, size=512):
        self.transforms = PairCompose([ToTensor01(), ResizePair(size)])

    def __call__(self, img, mask):
        return self.transforms(img, mask)


class LoveDADataset(Dataset):
    def __init__(self, root, split="Val", transform=None, ignore_index=255):
        self.transform = transform
        self.ignore_index = int(ignore_index)

        img_paths = sorted(glob.glob(
            os.path.join(root, split, "**", "images_png", "*.png"),
            recursive=True,
        ))

        self.pairs = []
        for img_path in img_paths:
            mask_path = img_path.replace("images_png", "masks_png")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            self.pairs.append((img_path, mask_path))

        if not self.pairs:
            raise RuntimeError(f"No image/mask pairs found under root={root!r}, split={split!r}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform is not None:
            img, mask = self.transform(img, mask)
        else:
            img = TF.to_tensor(img).clamp(0.0, 1.0)

        mask = np.array(mask, copy=True)
        if mask.ndim == 3:
            mask = mask[..., 0]

        mask = torch.as_tensor(mask, dtype=torch.long)
        ignore = mask == 0
        mask = mask - 1
        mask[ignore] = self.ignore_index

        return img, mask


class Normalize:
    def __init__(self, mean, std):
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)

        if len(self.mean) != len(self.std):
            raise ValueError("mean and std must have the same length")
        if any(v <= 0 for v in self.std):
            raise ValueError("std values must be positive")

    def __call__(self, x):
        if x.ndim != 4:
            raise ValueError("x must have shape [B, C, H, W]")
        channels = x.shape[1]
        if channels != len(self.mean):
            raise ValueError(f"x has {x.shape[1]} channels, mean/std have {len(self.mean)}")

        mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        return (x - mean) / std


def evaluate_metrics(model, loader, device, num_classes, ignore_index=255, normalizer=None):
    model.eval()

    metrics = MetricCollection({
        "Pixel Accuracy": MulticlassAccuracy(
            num_classes=num_classes, average="micro",
            multidim_average="global", ignore_index=ignore_index
        ),
        "AA": MulticlassRecall(
            num_classes=num_classes, average="macro",
            multidim_average="global", ignore_index=ignore_index, zero_division=0
        ),
        "Kappa": MulticlassCohenKappa(
            num_classes=num_classes, ignore_index=ignore_index
        ),
        "mIoU": MulticlassJaccardIndex(
            num_classes=num_classes, average="macro",
            ignore_index=ignore_index, zero_division=0
        ),
        "Per-class IoU": MulticlassJaccardIndex(
            num_classes=num_classes, average=None,
            ignore_index=ignore_index, zero_division=0
        ),
    }).to(device)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if normalizer is not None:
                x = normalizer(x)

            pred = model(x).argmax(dim=1)
            metrics.update(pred, y)

    return metrics.compute()



if __name__ == "__main__":
    data_root = "./"
    test_split = "Val"
    model_path = "./CDFMA_res50UNet_h1024_w1024_bilinearFalse_prob1_LoveDA_200.pth"

    img_size = 1024
    num_classes = 7
    ignore_index = 255

    batch_size = 16
    num_workers = 8

    normalize_mean = (0.485, 0.456, 0.406)
    normalize_std = (0.229, 0.224, 0.225)

    class_names = [
        "background",
        "building",
        "road",
        "water",
        "barren",
        "forest",
        "agriculture",
    ]

    if len(class_names) != num_classes:
        raise ValueError(f"class_names has {len(class_names)} entries, num_classes={num_classes}")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
   

    test_transform = LoveDASampleTransform(size=img_size)
    normalizer = Normalize(normalize_mean, normalize_std)

    test_ds = LoveDADataset(
        data_root,
        split=test_split,
        transform=test_transform,
        ignore_index=ignore_index,
    )

    
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = torch.load(model_path, map_location=device, weights_only=False).to(device)

    print(f"Images:       {len(test_ds)}")
    print(f"Batches:      {len(test_dl)}")
    print(f"Model:        {model_path}")
    print(f"Device:       {device}")
    print(f"Image size:   {img_size}")

    results = evaluate_metrics(
        model, test_dl, device, num_classes,
        ignore_index=ignore_index,
        normalizer=normalizer,
    )

    print("\nOverall metrics:")
    print("Pixel Accuracy:", results["Pixel Accuracy"].item())
    print("AA:", results["AA"].item())
    print("Kappa:", results["Kappa"].item())
    print("mIoU:", results["mIoU"].item())
    print("\nPer-class IoU:")
    for name, iou in zip(class_names, results["Per-class IoU"]):
        print(name, ":", iou.item())