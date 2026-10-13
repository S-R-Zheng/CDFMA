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
import time

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


def quantize_to_bins(values, num_bins):
    idx = torch.floor(values * num_bins).long()
    idx = idx.clamp(min=0, max=num_bins - 1)
    return idx


def compute_normalized_histogram_from_indices(indices, num_bins, dtype):
    hist = torch.bincount(indices, minlength=num_bins).to(dtype=dtype)
    hist = hist / hist.sum().clamp_min(1.0)
    return hist


def build_tile_meta(length, tile_size, device, dtype):
    if tile_size < 1:
        raise ValueError("tile_size must be >= 1")

    starts = list(range(0, length, tile_size))
    ends = [min(s + tile_size, length) for s in starts]
    centers = [0.5 * (s + e - 1) for s, e in zip(starts, ends)]

    starts_t = torch.tensor(starts, device=device, dtype=torch.long)
    ends_t = torch.tensor(ends, device=device, dtype=torch.long)
    centers_t = torch.tensor(centers, device=device, dtype=dtype)
    return starts_t, ends_t, centers_t


def compute_cdf(ref_ch, num_bins, tile_h, tile_w):
    device = ref_ch.device
    dtype = ref_ch.dtype
    height, width = ref_ch.shape

    y_starts, y_ends, centers_y = build_tile_meta(
        height,
        tile_h,
        device=device,
        dtype=dtype,
    )
    x_starts, x_ends, centers_x = build_tile_meta(
        width,
        tile_w,
        device=device,
        dtype=dtype,
    )

    num_tiles_y = y_starts.numel()
    num_tiles_x = x_starts.numel()

    ref_cdfs = torch.empty(
        (num_tiles_y, num_tiles_x, num_bins),
        device=device,
        dtype=dtype,
    )

    for iy in range(num_tiles_y):
        ys = int(y_starts[iy].item())
        ye = int(y_ends[iy].item())

        for ix in range(num_tiles_x):
            xs = int(x_starts[ix].item())
            xe = int(x_ends[ix].item())

            tile = ref_ch[ys:ye, xs:xe].reshape(-1)
            tile_idx = quantize_to_bins(tile, num_bins=num_bins)
            tile_hist = compute_normalized_histogram_from_indices(
                tile_idx,
                num_bins=num_bins,
                dtype=dtype,
            )

            tile_cdf = torch.cumsum(tile_hist, dim=0).clamp(max=1.0)
            ref_cdfs[iy, ix] = tile_cdf

    return ref_cdfs, y_starts, y_ends, centers_y, x_starts, x_ends, centers_x


def apply_interpolated_cdf_mapping(
    src_ch,
    ref_cdfs,
    num_bins,
    tile_h,
    tile_w,
):
    if src_ch.ndim != 2:
        raise ValueError("src_ch must be a 2D tensor with shape [H, W]")
    if ref_cdfs.ndim != 3:
        raise ValueError("ref_cdfs must have shape [num_tiles_y, num_tiles_x, num_bins]")
    if tile_h < 1 or tile_w < 1:
        raise ValueError("tile_h and tile_w must be >= 1")

    height, width = src_ch.shape
    num_tiles_y, num_tiles_x, cdf_bins = ref_cdfs.shape
    if cdf_bins != num_bins:
        raise ValueError("ref_cdfs last dimension must match num_bins")

    device = src_ch.device
    dtype = src_ch.dtype

    src_idx = quantize_to_bins(src_ch, num_bins=num_bins)
    out = torch.empty_like(src_ch)

    x = torch.arange(width, device=device, dtype=dtype)
    txf = (x + 0.5) / float(tile_w) - 0.5
    tx1_raw = torch.floor(txf).long()
    tx2_raw = tx1_raw + 1
    xa = (txf - tx1_raw.to(dtype)).view(1, width)
    xa1 = 1.0 - xa
    tx1 = tx1_raw.clamp(min=0, max=num_tiles_x - 1)
    tx2 = tx2_raw.clamp(min=0, max=num_tiles_x - 1)

    for y in range(height):
        tyf = (float(y) + 0.5) / float(tile_h) - 0.5
        ty1_raw = int(torch.floor(torch.tensor(tyf)).item())
        ty2_raw = ty1_raw + 1
        ya = tyf - float(ty1_raw)
        ya1 = 1.0 - ya

        ty1 = min(max(ty1_raw, 0), num_tiles_y - 1)
        ty2 = min(max(ty2_raw, 0), num_tiles_y - 1)

        row_idx = src_idx[y]

        top_left = ref_cdfs[ty1, tx1, row_idx]
        top_right = ref_cdfs[ty1, tx2, row_idx]
        bottom_left = ref_cdfs[ty2, tx1, row_idx]
        bottom_right = ref_cdfs[ty2, tx2, row_idx]

        top = top_left * xa1.view(width) + top_right * xa.view(width)
        bottom = bottom_left * xa1.view(width) + bottom_right * xa.view(width)
        out[y] = top * ya1 + bottom * ya

    return out.clamp(0.0, 1.0)


def apply_cdf_mapping(
    src_ch,
    ref_ch,
    num_bins=256,
    tile_h=128,
    tile_w=128,
    use_bilinear=True,
):

    (
        ref_cdfs,
        y_starts,
        y_ends,
        _centers_y,
        x_starts,
        x_ends,
        _centers_x,
    ) = compute_cdf(
        ref_ch=ref_ch,
        num_bins=num_bins,
        tile_h=tile_h,
        tile_w=tile_w,
    )

    if use_bilinear:
        return apply_interpolated_cdf_mapping(
            src_ch=src_ch,
            ref_cdfs=ref_cdfs,
            num_bins=num_bins,
            tile_h=tile_h,
            tile_w=tile_w,
        )

    mapped = torch.empty_like(src_ch)

    num_tiles_y = y_starts.numel()
    num_tiles_x = x_starts.numel()

    for iy in range(num_tiles_y):
        ys = int(y_starts[iy].item())
        ye = int(y_ends[iy].item())

        for ix in range(num_tiles_x):
            xs = int(x_starts[ix].item())
            xe = int(x_ends[ix].item())

            src_tile = src_ch[ys:ye, xs:xe]
            src_idx = quantize_to_bins(src_tile.reshape(-1), num_bins=num_bins)

            tile_cdf = ref_cdfs[iy, ix]
            mapped_tile = tile_cdf[src_idx].view(ye - ys, xe - xs)

            mapped[ys:ye, xs:xe] = mapped_tile

    return mapped.clamp(0.0, 1.0)


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


class CDFMAugmenter:
    def __init__(
        self,
        num_bins=256,
        local_region_h=128,
        local_region_w=128,
        CDFMA_prob=0.5,
        use_bilinear=True,
        enabled=True,
    ):
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2")
        if local_region_h < 1 or local_region_w < 1:
            raise ValueError("local_region_h and local_region_w must be >= 1")
        if not 0.0 <= CDFMA_prob <= 1.0:
            raise ValueError("CDFMA_prob must be in [0, 1]")

        self.num_bins = int(num_bins)
        self.local_region_h = int(local_region_h)
        self.local_region_w = int(local_region_w)
        self.CDFMA_prob = float(CDFMA_prob)
        self.use_bilinear = bool(use_bilinear)
        self.enabled = bool(enabled)

    def __call__(self, x):
        if not self.enabled:
            return x

        if x.ndim != 4:
            raise ValueError("x must be a 4D tensor with shape [B, C, H, W]")

        batch_size, channels, _, _ = x.shape

        x01 = x.clamp(0.0, 1.0)

        if torch.rand((), device=x.device).item() >= self.CDFMA_prob:
            return x01

        ref_index = build_batch_random_index(
            batch_size,
            device=x.device,
        )

        x01_aug = x01.clone()

        for bi in range(batch_size):
            bj = int(ref_index[bi].item())

            for ch in range(channels):
                src_ch = x01[bi, ch]
                ref_ch = x01[bj, ch]

                mapped_ch = apply_cdf_mapping(
                    src_ch=src_ch,
                    ref_ch=ref_ch,
                    num_bins=self.num_bins,
                    tile_h=self.local_region_h,
                    tile_w=self.local_region_w,
                    use_bilinear=self.use_bilinear,
                )

                x01_aug[bi, ch] = mapped_ch

        return x01_aug.clamp(0.0, 1.0)


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
    img_size = 1024
    num_classes = 7
    ignore_index = 255
    batch_size = 16
    lr = 1e-3
    num_workers = 8

    crop_scale = (0.7, 1.0)
    crop_ratio = (3 / 4, 4 / 3)
    hflip_prob = 0.5
    vflip_prob = 0.5

    num_bins = 256
    local_region_h = 1024
    local_region_w = 1024
    CDFMA_prob = 1

    use_bilinear = False

    normalize_mean = (0.485, 0.456, 0.406)
    normalize_std = (0.229, 0.224, 0.225)

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

    cdfma_augmenter =CDFMAugmenter(
        num_bins=num_bins,
        local_region_h=local_region_h,
        local_region_w=local_region_w,
        CDFMA_prob=CDFMA_prob,
        use_bilinear=use_bilinear,
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
  
    print(f"crop_scale = {crop_scale}")
    print(f"crop_ratio = {crop_ratio}")
    print(f"hflip_prob = {hflip_prob}")
    print(f"vflip_prob = {vflip_prob}")
    print(f"num_bins = {num_bins}")
    print(f"local_region_h = {local_region_h}")
    print(f"local_region_w = {local_region_w}")
    print(f"CDFMA_prob = {CDFMA_prob}")
    print(f"use_bilinear = {use_bilinear}")
    print(f"normalize_mean = {normalize_mean}")
    print(f"normalize_std = {normalize_std}")

    train_loss = train_one_epoch(
            model=model,
            loader=train_dl,
            optimizer=optimizer,
            device=device,
            epoch=1,
            ignore_index=ignore_index,
            batch_augmenter=cdfma_augmenter,
            normalizer=normalizer,
        )

    start = time.perf_counter()
    train_loss = train_one_epoch(
            model=model,
            loader=train_dl,
            optimizer=optimizer,
            device=device,
            epoch=2,
            ignore_index=ignore_index,
            batch_augmenter=cdfma_augmenter,
            normalizer=normalizer,
        )
    
    end=time.perf_counter()
    print(f"CDFMA_bins{num_bins}_h{local_region_h}_w{local_region_w}_bilinear{use_bilinear}_prob{CDFMA_prob}_time: {end-start}s")

       
