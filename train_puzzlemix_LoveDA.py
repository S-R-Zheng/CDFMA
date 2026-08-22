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
import gco


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


def segmentation_soft_cross_entropy(logits, target_soft):
    if logits.shape != target_soft.shape:
        raise ValueError("logits and target_soft must have the same shape")

    log_probs = F.log_softmax(logits, dim=1)
    loss_map = -(target_soft * log_probs).sum(dim=1)
    target_mass = target_soft.sum(dim=1)
    return loss_map.sum() / target_mass.sum().clamp_min(1.0)


def get_transport_cost_matrix(block_num, device, dtype):
    coords_y, coords_x = torch.meshgrid(
        torch.arange(block_num, device=device, dtype=dtype),
        torch.arange(block_num, device=device, dtype=dtype),
        indexing="ij",
    )
    coords = torch.stack([coords_y.reshape(-1), coords_x.reshape(-1)], dim=1)
    diff = coords[:, None, :] - coords[None, :, :]
    cost = (diff.pow(2).sum(dim=-1)) / float(max(block_num - 1, 1) ** 2)
    return cost.unsqueeze(0)


def graphcut_multi(unary1, unary2, pw_x, pw_y, alpha, beta, eta, n_labels=3, eps=1e-8):
    block_num = unary1.shape[0]
    large_val = 1000 * block_num ** 2

    if n_labels == 2:
        prior = np.array([-np.log(alpha + eps), -np.log(1 - alpha + eps)])
    elif n_labels == 3:
        prior = np.array(
            [
                -np.log(alpha ** 2 + eps),
                -np.log(2 * alpha * (1 - alpha) + eps),
                -np.log((1 - alpha) ** 2 + eps),
            ]
        )
    elif n_labels == 4:
        prior = np.array(
            [
                -np.log(alpha ** 3 + eps),
                -np.log(3 * alpha ** 2 * (1 - alpha) + eps),
                -np.log(3 * alpha * (1 - alpha) ** 2 + eps),
                -np.log((1 - alpha) ** 3 + eps),
            ]
        )
    else:
        raise ValueError("n_labels must be 2, 3, or 4, matching the official code")

    prior = eta * prior / block_num ** 2
    label_values = np.linspace(0, 1, n_labels)
    unary_cost = np.stack(
        [(1 - lam) * unary1 + lam * unary2 + prior[i] for i, lam in enumerate(label_values)],
        axis=-1,
    ).astype(np.float32)

    pairwise_cost = np.zeros(shape=(n_labels, n_labels), dtype=np.float32)
    for i in range(n_labels):
        for j in range(n_labels):
            pairwise_cost[i, j] = (i - j) ** 2 / float((n_labels - 1) ** 2)

    edge_x = pw_x + beta
    edge_y = pw_y + beta

    unary_cost_int = (large_val * unary_cost).astype(np.int32)
    edge_x_int = (large_val * edge_x).astype(np.int32)
    edge_y_int = (large_val * edge_y).astype(np.int32)
    labels = gco.cut_grid_graph(
        unary_cost_int,
        pairwise_cost,
        edge_x_int,
        edge_y_int,
        algorithm="swap",
    )
    labels = labels.reshape(block_num, block_num)
    return 1.0 - labels.astype(np.float32) / float(n_labels - 1)


def neigh_penalty(input1, input2, k):
    pw_x = input1[:, :, :-1, :] - input2[:, :, 1:, :]
    pw_y = input1[:, :, :, :-1] - input2[:, :, :, 1:]

    pw_x = pw_x[:, :, k - 1::k, :]
    pw_y = pw_y[:, :, :, k - 1::k]

    pw_x = F.avg_pool2d(pw_x.abs().mean(1, keepdim=True), kernel_size=(1, k)).squeeze(1)
    pw_y = F.avg_pool2d(pw_y.abs().mean(1, keepdim=True), kernel_size=(k, 1)).squeeze(1)
    return pw_x, pw_y


def mask_transport(mask, grad_pool, xi=0.8):
    batch_size = mask.shape[0]
    block_num = mask.shape[-1]
    n_iter = int(block_num)

    cost_matrix = get_transport_cost_matrix(
        block_num=block_num,
        device=mask.device,
        dtype=mask.dtype,
    )

    z = (mask > 0).to(dtype=mask.dtype)
    cost = xi * cost_matrix - grad_pool.reshape(batch_size, block_num ** 2, 1) * z.reshape(
        batch_size, 1, block_num ** 2
    )

    for _ in range(n_iter):
        row_best = cost.min(-1)[1]
        plan = torch.zeros_like(cost).scatter_(-1, row_best.unsqueeze(-1), 1.0)

        cost_fight = plan * cost
        col_best = cost_fight.min(-2)[1]
        plan_win = torch.zeros_like(cost).scatter_(-2, col_best.unsqueeze(-2), 1.0) * plan
        plan_lose = (1.0 - plan_win) * plan
        cost = cost + plan_lose

    return plan_win


def transport_tensor(tensor, plan, block_num, block_size):
    batch_size, channels, height, width = tensor.shape
    expected = block_num * block_size
    if height != expected or width != expected:
        raise ValueError("height and width must equal block_num * block_size for transport")

    patches = tensor.reshape(
        batch_size,
        channels,
        block_num,
        block_size,
        block_num,
        block_size,
    )
    patches = patches.permute(0, 1, 2, 4, 3, 5).reshape(
        batch_size,
        channels,
        block_num ** 2,
        block_size,
        block_size,
    )

    transported = torch.einsum("bij,bcihw->bcjhw", plan.to(dtype=tensor.dtype), patches)
    transported = transported.reshape(
        batch_size,
        channels,
        block_num,
        block_num,
        block_size,
        block_size,
    )
    transported = transported.permute(0, 1, 2, 4, 3, 5).reshape(
        batch_size,
        channels,
        height,
        width,
    )
    return transported


def apply_puzzlemix_graph(
    x,
    y_one_hot,
    saliency,
    indices,
    graph_size=2,
    alpha=0.5,
    beta=1.2,
    gamma=0.5,
    eta=0.2,
    xi=0.8,
    neigh_size=4,
    n_labels=3,
    transport=True,
    transport_size=-1,
    noise=None,
    adv_mask1=False,
    adv_mask2=False,
):
    if x.ndim != 4:
        raise ValueError("x must be a 4D tensor with shape [B, C, H, W]")
    if y_one_hot.ndim != 4:
        raise ValueError("y_one_hot must be a 4D tensor with shape [B, K, H, W]")
    if x.shape[0] != y_one_hot.shape[0] or x.shape[2:] != y_one_hot.shape[2:]:
        raise ValueError("x and y_one_hot must have matching batch and spatial dimensions")

    batch_size, _, height, width = x.shape
    if height != width:
        raise ValueError("this PuzzleMix implementation expects square inputs")
    if width % graph_size != 0:
        raise ValueError("input width must be divisible by graph_size")

    device = x.device
    dtype = x.dtype
    block_size = width // graph_size
    neigh_size = int(min(neigh_size, block_size))
    if block_size % neigh_size != 0:
        divisors = [d for d in range(neigh_size, 0, -1) if block_size % d == 0]
        neigh_size = divisors[0]
    k = block_size // neigh_size

    x1 = x.clamp(0.0, 1.0)
    x2 = x1[indices].clone()
    y1 = y_one_hot
    y2 = y_one_hot[indices].clone()

    beta_norm = float(beta) / graph_size / 16.0

    grad_pool = F.avg_pool2d(saliency.unsqueeze(1), block_size).squeeze(1)
    grad_sum = grad_pool.reshape(batch_size, -1).sum(1).reshape(batch_size, 1, 1)
    unary1_torch = grad_pool / grad_sum.clamp_min(1e-12)
    unary2_torch = unary1_torch[indices]

    x1_pool = F.avg_pool2d(x1, neigh_size)
    x2_pool = x1_pool[indices]

    pw_x = torch.zeros(
        batch_size,
        2,
        2,
        graph_size - 1,
        graph_size,
        device=device,
        dtype=dtype,
    )
    pw_y = torch.zeros(
        batch_size,
        2,
        2,
        graph_size,
        graph_size - 1,
        device=device,
        dtype=dtype,
    )

    pw_x[:, 0, 0], pw_y[:, 0, 0] = neigh_penalty(x2_pool, x2_pool, k)
    pw_x[:, 0, 1], pw_y[:, 0, 1] = neigh_penalty(x2_pool, x1_pool, k)
    pw_x[:, 1, 0], pw_y[:, 1, 0] = neigh_penalty(x1_pool, x2_pool, k)
    pw_x[:, 1, 1], pw_y[:, 1, 1] = neigh_penalty(x1_pool, x1_pool, k)

    pw_x = beta_norm * float(gamma) * pw_x
    pw_y = beta_norm * float(gamma) * pw_y

    unary1 = unary1_torch.clone()
    unary2 = unary2_torch.clone()

    unary2[:, :-1, :] += (pw_x[:, 1, 0] + pw_x[:, 1, 1]) / 2.0
    unary1[:, :-1, :] += (pw_x[:, 0, 1] + pw_x[:, 0, 0]) / 2.0
    unary2[:, 1:, :] += (pw_x[:, 0, 1] + pw_x[:, 1, 1]) / 2.0
    unary1[:, 1:, :] += (pw_x[:, 1, 0] + pw_x[:, 0, 0]) / 2.0

    unary2[:, :, :-1] += (pw_y[:, 1, 0] + pw_y[:, 1, 1]) / 2.0
    unary1[:, :, :-1] += (pw_y[:, 0, 1] + pw_y[:, 0, 0]) / 2.0
    unary2[:, :, 1:] += (pw_y[:, 0, 1] + pw_y[:, 1, 1]) / 2.0
    unary1[:, :, 1:] += (pw_y[:, 1, 0] + pw_y[:, 0, 0]) / 2.0

    pw_x = (pw_x[:, 1, 0] + pw_x[:, 0, 1] - pw_x[:, 1, 1] - pw_x[:, 0, 0]) / 2.0
    pw_y = (pw_y[:, 1, 0] + pw_y[:, 0, 1] - pw_y[:, 1, 1] - pw_y[:, 0, 0]) / 2.0

    unary1_np = unary1.detach().cpu().numpy()
    unary2_np = unary2.detach().cpu().numpy()
    pw_x_np = pw_x.detach().cpu().numpy()
    pw_y_np = pw_y.detach().cpu().numpy()

    masks = []
    for bi in range(batch_size):
        masks.append(
            graphcut_multi(
                unary1=unary2_np[bi],
                unary2=unary1_np[bi],
                pw_x=pw_x_np[bi],
                pw_y=pw_y_np[bi],
                alpha=alpha,
                beta=beta_norm,
                eta=eta,
                n_labels=n_labels,
            )
        )

    mask = torch.as_tensor(np.stack(masks, axis=0), device=device, dtype=dtype).unsqueeze(1)

    if noise is not None and adv_mask1:
        x1 = (x1 + noise).clamp(0.0, 1.0)
    if noise is not None and adv_mask2:
        x2 = (x2 + noise[indices]).clamp(0.0, 1.0)

    if transport:
        if transport_size == -1:
            transport_block_num = graph_size
            t_size = block_size
            transport_mask = mask
            transport_unary1 = unary1_torch
            transport_unary2 = unary2_torch
        elif transport_size < block_size:
            t_size = int(transport_size)
            if t_size < 1 or width % t_size != 0 or block_size % t_size != 0:
                raise ValueError(
                    "transport_size must divide both input width and graph block size"
                )
            transport_block_num = width // t_size
            transport_mask = F.interpolate(
                mask,
                size=(transport_block_num, transport_block_num),
                mode="nearest",
            )
            grad_pool_t = F.avg_pool2d(saliency.unsqueeze(1), t_size).squeeze(1)
            grad_sum_t = grad_pool_t.reshape(batch_size, -1).sum(1).reshape(batch_size, 1, 1)
            transport_unary1 = grad_pool_t / grad_sum_t.clamp_min(1e-12)
            transport_unary2 = transport_unary1[indices]
        else:
            transport_block_num = graph_size
            t_size = block_size
            transport_mask = mask
            transport_unary1 = unary1_torch
            transport_unary2 = unary2_torch

        plan1 = mask_transport(transport_mask, transport_unary1, xi=xi)
        x1 = transport_tensor(x1, plan1, transport_block_num, t_size)
        y1 = transport_tensor(y1, plan1, transport_block_num, t_size)

        plan2 = mask_transport(1.0 - transport_mask, transport_unary2, xi=xi)
        x2 = transport_tensor(x2, plan2, transport_block_num, t_size)
        y2 = transport_tensor(y2, plan2, transport_block_num, t_size)

        mask = transport_mask

    mask_up = F.interpolate(mask, size=(height, width), mode="nearest")
    x_mix = mask_up * x1 + (1.0 - mask_up) * x2
    y_mix = mask_up * y1 + (1.0 - mask_up) * y2

    return x_mix.clamp(0.0, 1.0), y_mix


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


class PuzzleMixAugmenter:
    def __init__(
        self,
        num_classes,
        mix_alpha=1.0,
        puzzlemix_prob=1.0,
        beta=1.2,
        gamma=0.5,
        eta=0.2,
        xi=0.8,
        graph_sizes=((2, 2), (4, 4), (8, 8), (16, 16)),
        n_labels=3,
        neigh_size=4,
        p_adversary=0.1,
        adv_epsilon=10.0 / 255.0,
        adv_step_size=12.0 / 255.0,
        transport=True,
        transport_size=-1,
        enabled=True,
    ):
        if mix_alpha < 0.0:
            raise ValueError("mix_alpha must be >= 0")
        if not (0.0 <= puzzlemix_prob <= 1.0):
            raise ValueError("puzzlemix_prob must be in [0, 1]")
        n_labels = int(n_labels)
        if n_labels not in (2, 3, 4):
            raise ValueError("n_labels must be 2, 3, or 4, matching the official graphcut_multi options")
        if not (0.0 <= p_adversary <= 1.0):
            raise ValueError("p_adversary must be in [0, 1]")
        if adv_epsilon < 0.0 or adv_step_size < 0.0:
            raise ValueError("adv_epsilon and adv_step_size must be non-negative")

        self.num_classes = int(num_classes)
        self.mix_alpha = float(mix_alpha)
        self.puzzlemix_prob = float(puzzlemix_prob)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.eta = float(eta)
        self.xi = float(xi)
        self.graph_sizes = tuple((int(h), int(w)) for h, w in graph_sizes)
        self.graph_size_values = tuple(int(h) for h, w in self.graph_sizes if int(h) == int(w))
        if len(self.graph_size_values) != len(self.graph_sizes):
            raise ValueError("only square graph sizes are supported")
        self.n_labels = int(n_labels)
        self.neigh_size = int(neigh_size)
        self.p_adversary = float(p_adversary)
        self.adv_epsilon = float(adv_epsilon)
        self.adv_step_size = float(adv_step_size)
        self.transport = bool(transport)
        self.transport_size = int(transport_size)
        self.enabled = bool(enabled)

    def sample_lambda(self):
        if self.mix_alpha <= 0.0:
            return 1.0
        return float(np.random.beta(self.mix_alpha, self.mix_alpha))

    def sample_graph_size(self, device):
        idx = int(torch.randint(0, len(self.graph_size_values), (1,), device=device).item())
        return self.graph_size_values[idx]

    def compute_saliency_and_noise(self, x, y, model, normalizer,ignore_index):
        
        model.eval()
        x_base = x.detach()
        adv_mask1 = torch.rand(1, device=x.device).item() < self.p_adversary
        adv_mask2 = torch.rand(1, device=x.device).item() < self.p_adversary

        noise = None
        x_for_grad = x_base
        if adv_mask1 or adv_mask2:
            noise = torch.empty_like(x_base).uniform_(-self.adv_epsilon, self.adv_epsilon)
            x_for_grad = (x_base + noise).clamp(0.0, 1.0)
            noise = x_for_grad - x_base
   
        x_ = normalizer(x_for_grad).detach().requires_grad_(True)
        logits = model(x_)
        loss_for_saliency_and_noise = F.cross_entropy(logits,y,ignore_index=ignore_index,)
        grad = torch.autograd.grad(
                loss_for_saliency_and_noise,
                x_,
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
            )[0].detach()
            
        saliency = torch.sqrt(torch.mean(grad.pow(2), dim=1) + 1e-12)
        
        if noise is not None:
            noise = noise + self.adv_step_size * grad.sign()
            noise = noise.clamp(-self.adv_epsilon, self.adv_epsilon)
            adv_mix_coef = torch.rand(1, device=x.device, dtype=x.dtype)
            noise = adv_mix_coef * noise
            noise = noise.detach()
        return saliency.detach(), noise, adv_mask1, adv_mask2

    def __call__(self, x, y, model,normalizer,ignore_index=255):
        if not self.enabled:
            return x, y,False

        if x.ndim != 4:
            raise ValueError("x must be a 4D tensor with shape [B, C, H, W]")
        if y.ndim != 3:
            raise ValueError("y must be a 3D tensor with shape [B, H, W]")
        if x.shape[0] != y.shape[0] or x.shape[2:] != y.shape[1:]:
            raise ValueError("x and y must have matching batch and spatial dimensions")

        if torch.rand(1, device=x.device).item() >= self.puzzlemix_prob:
            return x,y,False

        batch_size = x.shape[0]
        ref_index = build_batch_random_index(batch_size, device=x.device)

        lam = self.sample_lambda()
        graph_size = self.sample_graph_size(device=x.device)

        y_one_hot = make_one_hot_segmentation(y, self.num_classes, ignore_index=ignore_index)
        saliency, noise, adv_mask1, adv_mask2 = self.compute_saliency_and_noise(
            x=x,
            y=y,
            model=model,
            normalizer=normalizer,
            ignore_index=ignore_index,
        )

        x_mix, y_mix = apply_puzzlemix_graph(
            x=x,
            y_one_hot=y_one_hot,
            saliency=saliency,
            indices=ref_index,
            graph_size=graph_size,
            alpha=lam,
            beta=self.beta,
            gamma=self.gamma,
            eta=self.eta,
            xi=self.xi,
            neigh_size=self.neigh_size,
            n_labels=self.n_labels,
            transport=self.transport,
            transport_size=self.transport_size,
            noise=noise,
            adv_mask1=adv_mask1,
            adv_mask2=adv_mask2,
        )

        return x_mix, y_mix,True


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
    clean_lam,
    batch_augmenter=None,
    normalizer=None,
):
    clean_lam = float(clean_lam)
    
    if clean_lam < 0.0:
        raise ValueError("clean_lam must be non-negative")

    model.train()

    running_loss = 0.0
    running_clean_loss = 0.0
    running_mixed_loss = 0.0
    num_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        x_clean = x
        target_clean = y

        if batch_augmenter is not None:
            x_mix, target_mix,mix_applied = batch_augmenter(
                x_clean,
                target_clean,
                model=model,
                normalizer=normalizer,
                ignore_index=ignore_index,
            )
            model.train()
        else:
            x_mix,target_mix,mix_applied = x_clean, target_clean,False

        if normalizer is not None:
            x_clean_ = normalizer(x_clean)
            x_mix_ = normalizer(x_mix)
        else:
            x_clean_ = x_clean
            x_mix_ = x_mix

        if clean_lam > 0.0 and mix_applied:
            clean_logits = model(x_clean_)
            clean_loss = F.cross_entropy(clean_logits,target_clean,ignore_index=ignore_index,)
            weighted_clean_loss = clean_lam * clean_loss
            weighted_clean_loss.backward()
        else:
            weighted_clean_loss =  x_clean_.new_zeros(())

        mixed_logits = model(x_mix_)

        if target_mix.ndim == 4:
            mixed_loss = segmentation_soft_cross_entropy(
                mixed_logits,
                target_mix,
            )
        elif target_mix.ndim == 3:
            mixed_loss = F.cross_entropy(mixed_logits,target_mix,ignore_index=ignore_index,)
        else:
            raise ValueError(
                "target_mix must be either a hard label tensor "
                "with shape [B, H, W] or a soft label tensor "
                "with shape [B, K, H, W]"
            )

        mixed_loss.backward()

        optimizer.step()

       
        batch_total_loss = (
            weighted_clean_loss.detach()
            + mixed_loss.detach()
        )

        running_loss += batch_total_loss.item()
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

    puzzlemix_alpha = 1
    puzzlemix_prob = 1
    puzzlemix_beta = 1.2
    puzzlemix_gamma = 0.5
    puzzlemix_eta = 0.2
    puzzlemix_xi = 0.8
    puzzlemix_graph_sizes = ((2, 2), (4, 4), (8, 8), (16, 16))
    puzzlemix_n_labels = 3  # L={0,1/2,1}; official graphcut_multi also supports 2 or 4 labels
    puzzlemix_neigh_size = 4
    puzzlemix_clean_lam = 1
    puzzlemix_p_adversary = 0.1
    puzzlemix_adv_epsilon = 10 / 255
    puzzlemix_adv_step_size = 12 / 255
    puzzlemix_transport = True
    puzzlemix_transport_size = -1

    normalize_mean = (0.485, 0.456, 0.406)
    normalize_std = (0.229, 0.224, 0.225)

    save_path = f"PuzzleMix_res50UNet_LoveDA_{epochs}.pth"

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

    puzzlemix_augmenter = PuzzleMixAugmenter(
        num_classes=num_classes,
        mix_alpha=puzzlemix_alpha,
        puzzlemix_prob=puzzlemix_prob,
        beta=puzzlemix_beta,
        gamma=puzzlemix_gamma,
        eta=puzzlemix_eta,
        xi=puzzlemix_xi,
        graph_sizes=puzzlemix_graph_sizes,
        n_labels=puzzlemix_n_labels,
        neigh_size=puzzlemix_neigh_size,
        p_adversary=puzzlemix_p_adversary,
        adv_epsilon=puzzlemix_adv_epsilon,
        adv_step_size=puzzlemix_adv_step_size,
        transport=puzzlemix_transport,
        transport_size=puzzlemix_transport_size,
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
    print(f"puzzlemix_alpha = {puzzlemix_alpha}")
    print(f"puzzlemix_prob = {puzzlemix_prob}")
    print(f"puzzlemix_beta = {puzzlemix_beta}")
    print(f"puzzlemix_gamma = {puzzlemix_gamma}")
    print(f"puzzlemix_eta = {puzzlemix_eta}")
    print(f"puzzlemix_xi = {puzzlemix_xi}")
    print(f"puzzlemix_graph_sizes = {puzzlemix_graph_sizes}")
    print(f"puzzlemix_n_labels = {puzzlemix_n_labels}")
    print(f"puzzlemix_clean_lam = {puzzlemix_clean_lam}")
    print(f"puzzlemix_p_adversary = {puzzlemix_p_adversary}")
    print(f"puzzlemix_adv_epsilon = {puzzlemix_adv_epsilon}")
    print(f"puzzlemix_adv_step_size = {puzzlemix_adv_step_size}")
    print(f"puzzlemix_transport = {puzzlemix_transport}")
    print(f"puzzlemix_transport_size = {puzzlemix_transport_size}")
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
            clean_lam=puzzlemix_clean_lam,
            batch_augmenter=puzzlemix_augmenter,
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