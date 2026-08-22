import os
import glob
import numpy as np
from PIL import Image

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as tvT
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class PairCompose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, img, mask):
        for transform in self.transforms:
            img, mask = transform(img, mask)
        return img, mask


class ToTensor01:
    def __call__(self, img, mask):
        img = TF.to_tensor(img).clamp(0.0, 1.0)
        return img, mask


class ResizePair:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __call__(self, img, mask):
        img = TF.resize(img, self.size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)
        return img.clamp(0.0, 1.0), mask


class RandomResizedCropPair:
    def __init__(self, size, scale=(0.7, 1.0), ratio=(0.75, 1.33)):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.scale = tuple(scale)
        self.ratio = tuple(ratio)

    def __call__(self, img, mask):
        i, j, h, w = tvT.RandomResizedCrop.get_params(
            img,
            scale=self.scale,
            ratio=self.ratio,
        )
        img = TF.resized_crop(
            img,
            i,
            j,
            h,
            w,
            self.size,
            interpolation=InterpolationMode.BILINEAR,
        )
        mask = TF.resized_crop(
            mask,
            i,
            j,
            h,
            w,
            self.size,
            interpolation=InterpolationMode.NEAREST,
        )
        return img.clamp(0.0, 1.0), mask


class RandomHorizontalFlipPair:
    def __init__(self, p=0.5):
        self.p = float(p)

    def __call__(self, img, mask):
        if torch.rand(1).item() < self.p:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        return img, mask


class RandomVerticalFlipPair:
    def __init__(self, p=0.5):
        self.p = float(p)

    def __call__(self, img, mask):
        if torch.rand(1).item() < self.p:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        return img, mask


class LoveDAChessMixDataset(Dataset):
    def __init__(
        self,
        root,
        split="Train",
        transform=None,
        ignore_index=255,
        synthetic_root=None,
        include_synthetic=False,
    ):
        self.ignore_index = int(ignore_index)
        self.transform = transform
        self.root = root
        self.split = split
        self.synthetic_root = synthetic_root
        self.include_synthetic = bool(include_synthetic)

        self.pairs = []

        original_img_paths = sorted(
            glob.glob(
                os.path.join(root, split, "**", "images_png", "*.png"),
                recursive=True,
            )
        )
        for p in original_img_paths:
            mp = p.replace("images_png", "masks_png")
            if os.path.exists(mp):
                self.pairs.append((p, mp, "original"))
            else:
                print(f"[WARN] mask not found for image: {p}")

        if self.include_synthetic:
            if synthetic_root is None:
                raise ValueError("synthetic_root must be provided when include_synthetic=True")

            synthetic_img_paths = sorted(
                glob.glob(
                    os.path.join(synthetic_root, "images_png", "*.png"),
                    recursive=False,
                )
            )
            for p in synthetic_img_paths:
                mp = p.replace("images_png", "masks_png")
                if os.path.exists(mp):
                    self.pairs.append((p, mp, "synthetic"))
                else:
                    print(f"[WARN] synthetic mask not found for image: {p}")

        if not self.pairs:
            raise RuntimeError("No image/mask pairs found for dataset.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp, _source = self.pairs[idx]
        img = Image.open(ip).convert("RGB")
        mask = Image.open(mp)

        if self.transform is not None:
            img, mask = self.transform(img, mask)
        else:
            img = TF.to_tensor(img).clamp(0.0, 1.0)

        mask = np.array(mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = torch.as_tensor(mask, dtype=torch.long)

        # LoveDA/ChessMix: 0=ignore, 1..7=classes -> 255=ignore, 0..6=classes。
        mask = mask - 1
        mask[mask == -1] = self.ignore_index

        return img, mask

    def count_sources(self):
        counts = {"original": 0, "synthetic": 0}
        for _, _, source in self.pairs:
            counts[source] = counts.get(source, 0) + 1
        return counts


class LoveDASampleTransform:
    def __init__(
        self,
        size=512,
        training=True,
        crop_scale=(0.7, 1.0),
        crop_ratio=(0.75, 1.33),
        hflip_prob=0.5,
        vflip_prob=0.5,
    ):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.training = bool(training)

        if self.training:
            self.transforms = PairCompose(
                [
                    ToTensor01(),
                    RandomResizedCropPair(
                        self.size,
                        scale=crop_scale,
                        ratio=crop_ratio,
                    ),
                    RandomHorizontalFlipPair(p=hflip_prob),
                    RandomVerticalFlipPair(p=vflip_prob),
                ]
            )
        else:
            self.transforms = PairCompose(
                [
                    ToTensor01(),
                    ResizePair(self.size),
                ]
            )

    def __call__(self, img, mask):
        return self.transforms(img, mask)


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
            raise ValueError("x must be a 4D tensor with shape [B, C, H, W]")

        channels = x.shape[1]
        if channels != len(self.mean):
            raise ValueError(
                f"x has {channels} channels, but mean/std have {len(self.mean)} values"
            )

        mean = torch.as_tensor(
            self.mean,
            device=x.device,
            dtype=x.dtype,
        ).view(1, channels, 1, 1)
        std = torch.as_tensor(
            self.std,
            device=x.device,
            dtype=x.dtype,
        ).view(1, channels, 1, 1)

        return (x - mean) / std


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    ignore_index,
    batch_augmenter=None,
    normalizer=None,
):
    model.train()

    running_loss = 0.0
    num_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if batch_augmenter is not None:
            x = batch_augmenter(x)

        if normalizer is not None:
            x = normalizer(x)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y, ignore_index=ignore_index)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected at epoch={epoch}")

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / max(num_batches, 1)
    return avg_loss


if __name__ == "__main__":
    data_root = "./"
    train_split = "Train"
    synthetic_root = "./ChessMix_synthetic"
    include_synthetic = True

    img_size = 1024
    num_classes = 7
    ignore_index = 255

    batch_size = 16
    epochs = 200
    lr = 1e-3
    lr_min = 1e-5
    num_workers = 8

    crop_scale = (0.7, 1)
    crop_ratio = (3 / 4, 4 / 3)
    hflip_prob = 0.5
    vflip_prob = 0.5

    normalize_mean = (0.485, 0.456, 0.406)
    normalize_std = (0.229, 0.224, 0.225)

    save_path = f"ChessMix_res50UNet_LoveDA_{epochs}.pth"

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = LoveDASampleTransform(
        size=img_size,
        training=True,
        crop_scale=crop_scale,
        crop_ratio=crop_ratio,
        hflip_prob=hflip_prob,
        vflip_prob=vflip_prob,
    )

    normalizer = Normalize(
        mean=normalize_mean,
        std=normalize_std,
    )

    train_ds = LoveDAChessMixDataset(
        data_root,
        split=train_split,
        transform=train_transform,
        ignore_index=ignore_index,
        synthetic_root=synthetic_root,
        include_synthetic=include_synthetic,
    )

    source_counts = train_ds.count_sources()
    print(f"Loaded original train images: {source_counts.get('original', 0)}")
    print(f"Loaded ChessMix synthetic images: {source_counts.get('synthetic', 0)}")
    print(f"Loaded total train images: {len(train_ds)}")

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Train batches per epoch: {len(train_dl)}")

    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
    ).to(device)

    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - 1, eta_min=lr_min)


    print(f"synthetic_root = {synthetic_root}")
    print(f"include_synthetic = {include_synthetic}")
    print(f"crop_scale = {crop_scale}")
    print(f"crop_ratio = {crop_ratio}")
    print(f"hflip_prob = {hflip_prob}")
    print(f"vflip_prob = {vflip_prob}")
    print(f"normalize_mean = {normalize_mean}")
    print(f"normalize_std = {normalize_std}")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_dl,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            ignore_index=ignore_index,
            batch_augmenter=None,
            normalizer=normalizer,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"epoch: {epoch} | "
            f"train_loss={train_loss:.6f} | "
            f"lr={current_lr:.8f}"
        )

        scheduler.step()

    torch.save(model, save_path)
    print(f"Training complete and last model saved to: {save_path}")
