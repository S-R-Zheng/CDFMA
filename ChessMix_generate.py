import os
import glob
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import albumentations as A
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ChessMixPatch:
    image_path: str
    mask_path: str
    x: int
    y: int
    size: int
    weight: float


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_random_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)


def as_hw(size):
    if isinstance(size, int):
        return int(size), int(size)
    if len(size) != 2:
        raise ValueError("size must be an int or a tuple/list with length 2")
    return int(size[0]), int(size[1])


def build_axis_starts(length, patch_size, stride):
   
    if length < patch_size:
        return []
    return list(range(0, length - patch_size+1, stride))


def find_loveda_pairs(root, split="Train"):
    img_paths = sorted(glob.glob(os.path.join(root, split, "**", "images_png", "*.png"), recursive=True))

    pairs = []
    for ip in img_paths:
        mp = ip.replace("images_png", "masks_png")
        if os.path.exists(mp):
            pairs.append((ip, mp))
        else:
            print(f"[WARN] mask not found for image: {ip}")

    if not pairs:
        raise RuntimeError(
            f"No image/mask pairs found under {os.path.join(root, split)}. "
            "Expected **/images_png/*.png and corresponding **/masks_png/*.png."
        )

    return pairs


def read_mask_array(mask_path):
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int64, copy=False)


def compute_global_class_pixels(pairs, num_classes):
    class_pixels = np.zeros(num_classes + 1, dtype=np.float64)

    for _, mask_path in pairs:
        mask = read_mask_array(mask_path)
        counts = np.bincount(mask.reshape(-1), minlength=num_classes + 1)
        class_pixels[1:num_classes + 1] += counts[1:num_classes + 1]

    if class_pixels[1:].sum() <= 0:
        raise RuntimeError(
            "No labeled pixels found. Check mask values: "
            "expected 1..num_classes as valid classes and 0 as ignore."
        )

    return class_pixels


def compute_class_weights(class_pixels, eps=1e-12):
    weights = np.zeros_like(class_pixels, dtype=np.float64)
    labeled = class_pixels[1:]
    positive = labeled > 0
    c_max = labeled[positive].max()

    for cls in range(1, len(class_pixels)):
        if class_pixels[cls] > 0:
            weights[cls] = c_max / max(class_pixels[cls], eps)

    return weights


def compute_patch_weight(mask_patch, class_weights, num_classes):
    counts = np.bincount(mask_patch.reshape(-1), minlength=num_classes + 1)
    return float(np.dot(counts[1:num_classes + 1], class_weights[1:num_classes + 1]))


def build_multiscale_patch_pools(
    pairs,
    patch_size,
    n_scale,
    patch_overlap,
    num_classes,
):

    patch_h, patch_w = as_hw(patch_size)
    if patch_h != patch_w:
        raise ValueError(
            f"patch_size must be square for the official ChessMix transforms, got {(patch_h, patch_w)}. "
            "RandomRotate90/Transpose would swap patch height and width."
        )
    patch_size = patch_h

    patch_overlap = float(patch_overlap)
    if not 0.0 <= patch_overlap < 1.0:
        raise ValueError("patch_overlap must be in [0, 1)")
    
    class_pixels = compute_global_class_pixels(pairs, num_classes=num_classes)
    class_weights = compute_class_weights(class_pixels)

    pools: Dict[int, List[ChessMixPatch]] = {scale: [] for scale in range(1, int(n_scale) + 1)}

    for image_path, mask_path in pairs:
        mask = read_mask_array(mask_path)
        height, width = mask.shape

        for scale in range(1, int(n_scale) + 1):
            current_patch_size = patch_size * scale
            stride = int(current_patch_size * (1.0 - patch_overlap))
            if stride < 1:
                raise ValueError("patch_overlap is too large: computed stride must be >= 1.")

            y_starts = build_axis_starts(height, current_patch_size, stride)
            x_starts = build_axis_starts(width, current_patch_size, stride)

            if not y_starts or not x_starts:
                continue

            for y in y_starts:
                for x in x_starts:
                    mask_patch = mask[y:y + current_patch_size, x:x + current_patch_size]
                    weight = compute_patch_weight(mask_patch, class_weights, num_classes)

                    if weight <= 0.0:
                        continue

                    pools[scale].append(
                        ChessMixPatch(
                            image_path=image_path,
                            mask_path=mask_path,
                            x=int(x),
                            y=int(y),
                            size=int(current_patch_size),
                            weight=float(weight),
                        )
                    )

    empty_scales = [scale for scale in range(1, int(n_scale) + 1) if not pools[scale]]

    if empty_scales:
        raise RuntimeError(
            f"No valid patches were extracted for scale(s): {empty_scales}. "
            "Check patch_size, n_scale, patch_overlap, image size, and mask labels."
        )

    return pools, class_pixels, class_weights


def weighted_sample_patch(patches: Sequence[ChessMixPatch]):
    weights = np.asarray([p.weight for p in patches], dtype=np.float64)
    probabilities = weights / weights.sum()
    idx = np.random.choice(len(patches), p=probabilities)
    return patches[idx]


def apply_random_patch_transform(img_patch, mask_patch):
    aug = A.Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.OneOf([
            A.GridDistortion(p=0.8),
            A.Perspective(p=0.8),
        ], p=0.5),
    ])

    augmented = aug(image=img_patch, mask=mask_patch)
    return np.ascontiguousarray(augmented["image"]), np.ascontiguousarray(augmented["mask"])


def read_patch(patch: ChessMixPatch):
    img = np.array(Image.open(patch.image_path).convert("RGB"))
    mask = read_mask_array(patch.mask_path).astype(np.uint8, copy=False)

    y0, x0, size = patch.y, patch.x, patch.size
    return img[y0:y0 + size, x0:x0 + size], mask[y0:y0 + size, x0:x0 + size]


def generate_one_chessmix_image(pools, crop_size, n_scale, mirror=True):
    crop_h, crop_w = as_hw(crop_size)
    scale = random.randint(1, int(n_scale))

    patches = pools[scale]
    block = patches[0].size
    rows = crop_h // block
    cols = crop_w // block

    image_canvas = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
    mask_canvas = np.zeros((crop_h, crop_w), dtype=np.uint8)  # 0=ignore

    for row in range(rows):
        active_cols = range(row % 2, cols, 2)
        mirror_offset = 1 if row % 2 == 0 else -1

        for col in active_cols:
            patch = weighted_sample_patch(patches)
            img_patch, mask_patch = read_patch(patch)
            img_patch, mask_patch = apply_random_patch_transform(img_patch, mask_patch)

            y0, x0 = row * block, col * block
            image_canvas[y0:y0 + block, x0:x0 + block] = img_patch
            mask_canvas[y0:y0 + block, x0:x0 + block] = mask_patch

            if mirror:
                mirror_col = col + mirror_offset
                if 0 <= mirror_col < cols:
                    mx0 = mirror_col * block
                    mirrored_img_patch = np.ascontiguousarray(np.flip(img_patch, axis=1))
                    image_canvas[y0:y0 + block, mx0:mx0 + block] = mirrored_img_patch

    return image_canvas, mask_canvas, scale


def generate_chessmix_dataset(
    data_root,
    output_root,
    split="Train",
    crop_size=1024,
    patch_size=256,
    n_scale=2,
    patch_overlap=0.5,
    n_images=1000,
    num_classes=7,
    mirror=True,
):

    image_out_dir = os.path.join(output_root, "images_png")
    mask_out_dir = os.path.join(output_root, "masks_png")
    ensure_dir(image_out_dir)
    ensure_dir(mask_out_dir)

    pairs = find_loveda_pairs(data_root, split=split)
    print(f"Found source image/mask pairs: {len(pairs)}")

    pools, class_pixels, class_weights = build_multiscale_patch_pools(
        pairs, patch_size, n_scale, patch_overlap, num_classes
    )

    crop_h, crop_w = as_hw(crop_size)
    for scale, patches in sorted(pools.items()):
        block = patches[0].size
        print(f"scale={scale} | patch_size={block} | candidate patches={len(patches)}")

    scale_counts = {scale: 0 for scale in sorted(pools)}

    for idx in range(1, int(n_images) + 1):
        image_canvas, mask_canvas, scale = generate_one_chessmix_image(
            pools, crop_size, n_scale, mirror
        )

        name = f"chessmix_{idx:06d}.png"
        Image.fromarray(image_canvas).save(os.path.join(image_out_dir, name))
        Image.fromarray(mask_canvas).save(os.path.join(mask_out_dir, name))
        scale_counts[scale] = scale_counts.get(scale, 0) + 1

        if idx % 100 == 0 or idx == int(n_images):
            print(f"Generated {idx}/{n_images}")

    print(f"ChessMix generation complete: {output_root}")
    print(f"Images: {image_out_dir}")
    print(f"Masks:  {mask_out_dir}")
    print("Generated synthetic images by scale:")

    for scale in sorted(scale_counts):
        print(f"  scale={scale}: {scale_counts[scale]}")


if __name__ == "__main__":
    data_root = "./"
    train_split = "Train"
    synthetic_output_root = "./ChessMix_synthetic"
    crop_size = 1024
    patch_size = 256
    n_scale = 2
    patch_overlap = 0.5
    n_images = len(find_loveda_pairs(data_root, split=train_split))

    mirror = True

    # LoveDA: 0=ignore, 1..7=classes。
    num_classes = 7
    seed = 42

    set_random_seed(seed)

    generate_chessmix_dataset(
        data_root=data_root,
        output_root=synthetic_output_root,
        split=train_split,
        crop_size=crop_size,
        patch_size=patch_size,
        n_scale=n_scale,
        patch_overlap=patch_overlap,
        n_images=n_images,
        num_classes=num_classes,
        mirror=mirror,
    )