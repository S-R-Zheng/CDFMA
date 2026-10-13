import os
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


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
            raise ValueError(
                f"x has {channels} channels, "
                f"mean/std have {len(self.mean)}"
            )

        mean = torch.as_tensor(
            self.mean,
            device=x.device,
            dtype=x.dtype,
        ).view(1, -1, 1, 1)

        std = torch.as_tensor(
            self.std,
            device=x.device,
            dtype=x.dtype,
        ).view(1, -1, 1, 1)

        return (x - mean) / std


def load_image(image_path, img_size):
    image = Image.open(image_path).convert("RGB")
    image = TF.to_tensor(image).clamp(0.0, 1.0)
    image = TF.resize(
        image,
        (img_size, img_size),
        interpolation=InterpolationMode.BILINEAR,
    )
    image = image.clamp(0.0, 1.0)
    image = image.unsqueeze(0)

    return image


def load_label(label_path, img_size, ignore_index=255):
    label = Image.open(label_path)
    label = TF.resize(
        label,
        (img_size, img_size),
        interpolation=InterpolationMode.NEAREST,
    )

    label = np.array(label, copy=True)
    if label.ndim == 3:
        label = label[..., 0]

    label = torch.as_tensor(
        label,
        dtype=torch.long,
    )

    ignore = label == 0

    label = label - 1
    label[ignore] = ignore_index

    return label


def colorize_label(
    label,
    palette,
    ignore_index=255,
):

    if torch.is_tensor(label):
        label = label.detach().cpu().numpy()

    label = np.asarray(label)

    if label.ndim != 2:
        raise ValueError(
            f"label must have shape [H, W], "
            f"but got {label.shape}"
        )

    palette = np.asarray(
        palette,
        dtype=np.uint8,
    )

    rgb = np.zeros(
        (label.shape[0], label.shape[1], 3),
        dtype=np.uint8,
    )

    valid = (
        (label >= 0)
        & (label < len(palette))
    )

    rgb[valid] = palette[label[valid]]

    allowed = valid | (label == ignore_index)

    if not np.all(allowed):
        invalid_values = np.unique(
            label[~allowed]
        )

        raise ValueError(
            "Unexpected values in LoveDA label: "
            f"{invalid_values.tolist()}"
        )

    return Image.fromarray(rgb)


def colorize_prediction(prediction,palette,):
    if torch.is_tensor(prediction):
        prediction = prediction.detach().cpu().numpy()

    prediction = np.asarray(prediction)

    if prediction.ndim != 2:
        raise ValueError(
            f"prediction must have shape [H, W], "
            f"but got {prediction.shape}"
        )

    palette = np.asarray(
        palette,
        dtype=np.uint8,
    )

    if np.any(prediction < 0) or np.any(
        prediction >= len(palette)
    ):
        invalid_values = np.unique(
            prediction[
                (prediction < 0)
                | (prediction >= len(palette))
            ]
        )

        raise ValueError(
            "Unexpected class indices in prediction: "
            f"{invalid_values.tolist()}"
        )

    rgb = palette[prediction]

    return Image.fromarray(
        rgb.astype(np.uint8)
    )


if __name__ == "__main__":
    image_path = "./2721.png"
    label_path = "./label_2721.png"
    model_path = "./CDFMA_res50UNet_bins256_h1024_w1024_bilinearFalse_prob1_LoveDA_200.pth"
    label_output_path = "./visualization_label_2721.png"
    prediction_output_path = "./visualization_prediction_2721_CDFMA_bins256_h1024_w1024_bilinearFalse_prob1.png"

    img_size = 1024
    num_classes = 7
    ignore_index = 255

    normalize_mean = (0.485,0.456,0.406,)

    normalize_std = (0.229,0.224,0.225,)

    class_names = [
        "background",
        "building",
        "road",
        "water",
        "barren",
        "forest",
        "agriculture",
    ]

    palette = np.array([
    [255, 255, 255],   # 0 background  - white
    [255,   0,   0],   # 1 building    - red
    [128, 128, 128],   # 2 road        - gray
    [  0,   0, 255],   # 3 water       - blue
    [159, 129, 183],   # 4 barren      - purple
    [  0, 255,   0],   # 5 forest      - green
    [255, 165,   0],   # 6 agriculture - orange
                        ], dtype=np.uint8)

    if len(class_names) != num_classes:
        raise ValueError(
            f"class_names has {len(class_names)} entries, "
            f"num_classes={num_classes}"
        )

    if len(palette) != num_classes:
        raise ValueError(
            f"palette has {len(palette)} colors, "
            f"num_classes={num_classes}"
        )


    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    normalizer = Normalize(
        normalize_mean,
        normalize_std,
    )

    model = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    ).to(device)

    model.eval()

    label = load_label(
        label_path,
        img_size=img_size,
        ignore_index=ignore_index,
    )

    label_vis = colorize_label(
        label,
        palette,
        ignore_index=ignore_index,
    )

    label_vis.save(
        label_output_path
    )

    image = load_image(
        image_path,
        img_size=img_size,
    )

    image = image.to(
        device,
        non_blocking=True,
    )

    image = normalizer(image)

    with torch.no_grad():
        prediction = model(image).argmax(dim=1)

    prediction = prediction[0]
    prediction_vis = colorize_prediction(
        prediction,
        palette,
    )
    prediction_vis.save(
        prediction_output_path
    )

    print(f"Image:                  {image_path}")
    print(f"Label:                  {label_path}")
    print(f"Model:                  {model_path}")
    print(f"Device:                 {device}")
    print(f"Image size:             {img_size}")

    print(
        f"Label visualization:    "
        f"{os.path.abspath(label_output_path)}"
    )

    print(
        f"Prediction visualization: "
        f"{os.path.abspath(prediction_output_path)}"
    )
