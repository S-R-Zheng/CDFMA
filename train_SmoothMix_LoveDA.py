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

        # LoveDA: 0=ignore, 1..7=classes -> 255=ignore, 0..6=classes
        mask = mask - 1
        mask[mask == -1] = self.ignore_index

        return img, mask


def build_batch_random_index(batch_size, device):
    return torch.randperm(batch_size, device=device)



def parse_sigma_value(sigma):
    
    if isinstance(sigma, (tuple, list)):
        if len(sigma) != 2:
            raise ValueError("sigma tuple/list must have length 2")
        low, high = float(sigma[0]), float(sigma[1])
        if low < 0.0 or high <= 0.0:
            raise ValueError("sigma range must satisfy low >= 0 and high > 0")
        if high < low:
            raise ValueError("sigma range must satisfy high >= low")
        return (low, high)

    sigma = float(sigma)
    if sigma < 0.0:
        raise ValueError("sigma must be >= 0")
    return sigma


def sample_sigma(batch_size, sigma, device, dtype):
    if isinstance(sigma, tuple):
        low, high = sigma

        if high == low:
            return torch.full((1,),low,device=device,dtype=dtype,)

        return torch.empty(1,device=device,dtype=dtype,).uniform_(low, high)

    return torch.full((1,),float(sigma),device=device,dtype=dtype,)


def build_smoothmix_circle_masks(batch_size, height, width, sigma, device, dtype):
    eps = torch.finfo(dtype).eps
    sigma_values = sample_sigma(batch_size=batch_size,sigma=sigma,device=device,dtype=dtype,).clamp_min(eps)
    mu_x = torch.empty(1,device=device,dtype=dtype,).uniform_(0.0, float(width))
    mu_y = torch.empty(1,device=device,dtype=dtype,).uniform_(0.0, float(height))
    xs = torch.arange(width,device=device,dtype=dtype,).view(1, 1, width)
    ys = torch.arange(height,device=device,dtype=dtype,).view(1, height, 1)
    sigma_x = (sigma_values * float(width)).view(1, 1, 1).clamp_min(eps)
    sigma_y = (sigma_values * float(height)).view(1, 1, 1).clamp_min(eps)
    mu_x = mu_x.view(1, 1, 1)
    mu_y = mu_y.view(1, 1, 1)
    gx = torch.exp(-0.5 * ((xs - mu_x) / sigma_x) ** 2)
    gy = torch.exp(-0.5 * ((ys - mu_y) / sigma_y) ** 2)
    masks = (gy * gx).unsqueeze(1).clamp(0.0, 1.0)
    return masks.expand(batch_size,-1,-1,-1,)



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


class SmoothMixAugmenter:
    def __init__(
        self,
        num_classes,
        smoothmix_prob=0.5,
        sigma=(0.25, 0.5),
        enabled=True,
    ):
        if not 0.0 <= smoothmix_prob <= 1.0:
            raise ValueError("smoothmix_prob must be in [0, 1]")

        self.num_classes = int(num_classes)
        self.smoothmix_prob = float(smoothmix_prob)
        self.sigma = parse_sigma_value(sigma)
        self.enabled = bool(enabled)

    def build_masks(self, batch_size, height, width, device, dtype):
        return build_smoothmix_circle_masks(
            batch_size=batch_size,
            height=height,
            width=width,
            sigma=self.sigma,
            device=device,
            dtype=dtype,
        )

    def __call__(self, x, y, ignore_index=255):
        if not self.enabled:
            return x, y

        if x.ndim != 4:
            raise ValueError("x must be a 4D tensor with shape [B, C, H, W]")
        if y.ndim != 3:
            raise ValueError("y must be a 3D tensor with shape [B, H, W]")

        batch_size, _, height, width = x.shape

        x01 = x.clamp(0.0, 1.0)

        if torch.rand((), device=x.device).item() >= self.smoothmix_prob:
            return x01, y

        ref_index = build_batch_random_index(
            batch_size=batch_size,
            device=x.device,
        )

        x_ref = x01[ref_index]
        y_ref = y[ref_index]

        masks = self.build_masks(
            batch_size=batch_size,
            height=height,
            width=width,
            device=x.device,
            dtype=x01.dtype,
        )

        x_mix = masks * x01 + (1.0 - masks) * x_ref
        x_mix = x_mix.clamp(0.0, 1.0)

        target = make_one_hot_segmentation(
            y,
            self.num_classes,
            ignore_index=ignore_index,
        ).to(device=x.device, dtype=x01.dtype)
        target_ref = make_one_hot_segmentation(
            y_ref,
            self.num_classes,
            ignore_index=ignore_index,
        ).to(device=x.device, dtype=x01.dtype)

        target_mix = masks * target + (1.0 - masks) * target_ref
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

    crop_scale = (0.7, 1)
    crop_ratio = (3 / 4, 4 / 3)
    hflip_prob = 0.5
    vflip_prob = 0.5
    smoothmix_prob = 1
    sigma = (0.25, 0.5)

    normalize_mean = (0.485, 0.456, 0.406)
    normalize_std = (0.229, 0.224, 0.225)

    save_path = f"SmoothMix_res50UNet_LoveDA_{epochs}.pth"
    

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

    smoothmix_augmenter = SmoothMixAugmenter(
        num_classes=num_classes,
        smoothmix_prob=smoothmix_prob,
        sigma=sigma,
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
    print(f"smoothmix_prob = {smoothmix_prob}")
    print(f"sigma = {sigma}")
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
            batch_augmenter=smoothmix_augmenter,
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
