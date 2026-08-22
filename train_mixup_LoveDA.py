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


class LoveDADataset(Dataset):
    def __init__(self, root, split="Train", transform=None, ignore_index=255):
        self.ignore_index = int(ignore_index)
        self.transform = transform

        img_paths = sorted(
            glob.glob(
                os.path.join(root, split, "**", "images_png", "*.png"),
                recursive=True,
            )
        )

        self.pairs = []
        for p in img_paths:
            mp = p.replace("images_png", "masks_png")
            self.pairs.append((p, mp))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
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
        mask = mask - 1
        mask[mask == -1] = self.ignore_index

        return img, mask


def build_batch_random_index(batch_size, device):
    return torch.randperm(batch_size, device=device)


def make_one_hot_segmentation(y, num_classes, ignore_index=255):
    if y.ndim != 3:
        raise ValueError("y must be a 3D tensor with shape [B, H, W]")

    valid = y != ignore_index
    invalid = valid & ((y < 0) | (y >= num_classes))
    if torch.any(invalid):
        bad_value = int(y[invalid][0].item())
        raise ValueError(f"label value {bad_value} is outside [0, {num_classes - 1}]")

    batch_size, height, width = y.shape
    y_safe = y.clamp(min=0, max=num_classes - 1)

    one_hot = torch.zeros(
        batch_size,
        num_classes,
        height,
        width,
        device=y.device,
        dtype=torch.float32,
    )
    one_hot.scatter_(1, y_safe.unsqueeze(1), 1.0)
    one_hot = one_hot * valid.unsqueeze(1).to(dtype=one_hot.dtype)
    return one_hot


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

class MixUpAugmenter:
    def __init__(
        self,
        num_classes,
        mixup_alpha=1.0,
        mixup_prob=1.0,
        enabled=True,
    ):
        if not 0.0 <= mixup_prob <= 1.0:
            raise ValueError("mixup_prob must be in [0, 1]")

        self.num_classes = int(num_classes)
        self.mixup_alpha = float(mixup_alpha)
        self.mixup_prob = float(mixup_prob)
        self.enabled = bool(enabled)

    def sample_lambda(self, device, dtype):
        if self.mixup_alpha <= 0.0:
            return torch.ones(
                (),
                device=device,
                dtype=dtype,
            )

        alpha = torch.tensor(
            self.mixup_alpha,
            device=device,
            dtype=dtype,
        )

        beta_dist = torch.distributions.Beta(
            alpha,
            alpha,
        )

        lam = beta_dist.sample()
        return lam.clamp(0.0, 1.0)

    def __call__(self, x, y, ignore_index=255):
        if not self.enabled:
            return x, y

        if x.ndim != 4:
            raise ValueError(
                "x must be a 4D tensor with shape [B, C, H, W]"
            )

        if y.ndim != 3:
            raise ValueError(
                "y must be a 3D tensor with shape [B, H, W]"
            )

        if (
            x.shape[0] != y.shape[0]
            or x.shape[2:] != y.shape[1:]
        ):
            raise ValueError(
                "x and y must have matching batch and spatial dimensions"
            )

        batch_size = x.shape[0]
        x01 = x.clamp(0.0, 1.0)

        if self.mixup_prob <= 0.0:
            return x01, y

        if self.mixup_alpha <= 0.0:
            return x01, y

        if (
            torch.rand(
                (),
                device=x.device,
            ).item()
            >= self.mixup_prob
        ):
            return x01, y

        ref_index = build_batch_random_index(
            batch_size=batch_size,
            device=x.device,
        )

        x_ref = x01[ref_index]
        y_ref = y[ref_index]
        lam = self.sample_lambda(
            device=x.device,
            dtype=x01.dtype,
        )

        mix_mask = lam.view(1, 1, 1, 1)

        x_mix = (
            mix_mask * x01
            + (1.0 - mix_mask) * x_ref
        )
        x_mix = x_mix.clamp(0.0, 1.0)

        target = make_one_hot_segmentation(
            y,
            self.num_classes,
            ignore_index=ignore_index,
        ).to(
            device=x.device,
            dtype=x01.dtype,
        )

        target_ref = make_one_hot_segmentation(
            y_ref,
            self.num_classes,
            ignore_index=ignore_index,
        ).to(
            device=x.device,
            dtype=x01.dtype,
        )

        target_mix = (
            mix_mask * target
            + (1.0 - mix_mask) * target_ref
        )

        return x_mix, target_mix


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


def segmentation_soft_cross_entropy(logits, target_soft):
    if logits.shape != target_soft.shape:
        raise ValueError("logits and target_soft must have the same shape")

    log_probs = F.log_softmax(logits, dim=1)
    loss_map = -(target_soft * log_probs).sum(dim=1)
    target_mass = target_soft.sum(dim=1)
    return loss_map.sum() / target_mass.sum().clamp_min(1.0)


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
            x, target = batch_augmenter(
                x,
                y,
                ignore_index=ignore_index,
            )
        else:
            target = y

        if normalizer is not None:
            x = normalizer(x)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        if target.ndim == 4:
            loss = segmentation_soft_cross_entropy(logits, target)
        else:
            loss = F.cross_entropy(logits, target, ignore_index=ignore_index)

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
    img_size = 1024
    num_classes = 7
    ignore_index = 255

    batch_size = 16
    epochs = 200
    lr = 1e-3
    lr_min = 1e-5
    num_workers = 8

    crop_scale = (0.7, 1.0)
    crop_ratio = (3 / 4, 4 / 3)
    hflip_prob = 0.5
    vflip_prob = 0.5

    mixup_alpha = 1
    mixup_prob = 1

    normalize_mean = (0.485, 0.456, 0.406)
    normalize_std = (0.229, 0.224, 0.225)

    save_path = f"MixUp_res50UNet_LoveDA_{epochs}.pth"
    
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

    mixup_augmenter = MixUpAugmenter(
        num_classes=num_classes,
        mixup_alpha=mixup_alpha,
        mixup_prob=mixup_prob,
        enabled=True,
    )

    normalizer = Normalize(
        mean=normalize_mean,
        std=normalize_std,
    )

    train_ds = LoveDADataset(
        data_root,
        split=train_split,
        transform=train_transform,
        ignore_index=ignore_index,
    )

    print(f"Loaded train images: {len(train_ds)}")

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

    print(f"crop_scale = {crop_scale}")
    print(f"crop_ratio = {crop_ratio}")
    print(f"hflip_prob = {hflip_prob}")
    print(f"vflip_prob = {vflip_prob}")
    print(f"mixup_alpha = {mixup_alpha}")
    print(f"mixup_prob = {mixup_prob}")
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
            batch_augmenter=mixup_augmenter,
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
