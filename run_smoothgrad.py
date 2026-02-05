
import os
import datetime
import numpy as np
import cv2
from PIL import Image

import torch

from xai_core import (
    add_common_cli, build_model, load_val_meta, ensure_dir, iter_images,
    run_core_outputs, tfm, device, _predict_logits_tensor
)

def _robust_norm01(m2d: np.ndarray, lo=80.0, hi=99.5, gamma=0.8, eps=1e-8) -> np.ndarray:
    lo_v = np.percentile(m2d, lo)
    hi_v = np.percentile(m2d, hi)
    if hi_v <= lo_v + eps:
        return np.zeros_like(m2d)
    m = np.clip(m2d, lo_v, hi_v)
    m = (m - lo_v) / (hi_v - lo_v + eps)
    if gamma != 1.0:
        m = np.power(m, gamma)
    return m

def _pool_to_patches(m2d: np.ndarray, patch: int | None) -> np.ndarray:
    if patch is None or patch <= 0:
        return m2d
    H, W = m2d.shape
    Hc, Wc = (H // patch) * patch, (W // patch) * patch
    if Hc <= 0 or Wc <= 0:
        return m2d
    m2d = m2d[:Hc, :Wc]
    blocks = m2d.reshape(Hc // patch, patch, Wc // patch, patch).transpose(0, 2, 1, 3)
    return blocks.mean(axis=(2, 3))

def run_smoothgrad(
    model,
    img_path,
    target_idx=None,
    n_samples=25,
    sigma=0.15,
    batch_size=8,
    agg="abs",        # "abs" | "sq" | "raw" | "gi"
    patch=16,
    p_low=80.0,
    p_high=99.5,
    gamma_corr=0.8,
    colormap=cv2.COLORMAP_VIRIDIS
):
    model.eval()
    dev = next(model.parameters()).device

    pil = Image.open(img_path).convert("RGB")
    x0 = tfm(pil).unsqueeze(0).to(dev)

    with torch.no_grad():
        logits = _predict_logits_tensor(model, x0)
        pred_idx = int(logits.argmax(1).item())
    target_idx = pred_idx if target_idx is None else int(target_idx)

    # noise scale in normalized space
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    noise_scale = float(sigma) * imagenet_std

    total_unsigned = torch.zeros_like(x0)
    n_done = 0
    n_left = int(n_samples)

    while n_left > 0:
        b = min(int(batch_size), n_left)
        x = x0.repeat(b, 1, 1, 1)
        noise = torch.randn_like(x) * noise_scale
        x = (x + noise).clamp(-5, 5)
        x.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        y = _predict_logits_tensor(model, x)[:, target_idx].sum()
        y.backward()

        grads = x.grad.detach()

        if agg == "abs":
            u = grads.abs()
        elif agg == "sq":
            u = grads.pow(2)
        elif agg == "gi":
            u = (grads * x).abs()
        else:  # "raw"
            u = grads.abs()

        total_unsigned += u.mean(dim=0, keepdim=True)
        n_done += b
        n_left -= b

        del x, grads, u
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_unsigned = total_unsigned / max(1, n_done)
    sal = total_unsigned.sum(1, keepdim=False)[0].detach().cpu().numpy()  # [H,W]

    sal = _pool_to_patches(sal, None if patch == 0 else patch)
    sal = np.maximum(sal, 0.0)
    sal01 = _robust_norm01(sal, lo=p_low, hi=p_high, gamma=gamma_corr)

    raw = cv2.imread(img_path)
    sal01 = cv2.resize(sal01, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)
    heat_u8 = (sal01 * 255).astype(np.uint8)
    cm = cv2.applyColorMap(heat_u8, colormap)
    out = cv2.addWeighted(raw, 0.55, cm, 0.45, 0)
    return out, target_idx

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run SmoothGrad only.")
    add_common_cli(parser)
    parser.add_argument('--smoothgrad_samples', type=int, default=25)
    parser.add_argument('--smoothgrad_sigma', type=float, default=0.15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--agg', choices=['abs', 'sq', 'raw', 'gi'], default='abs')

    parser.add_argument('--patch', type=int, default=16, help="Patch pooling size; set 0 to disable.")
    parser.add_argument('--p_low', type=float, default=80.0)
    parser.add_argument('--p_high', type=float, default=99.5)
    parser.add_argument('--gamma', type=float, default=0.8)
    args = parser.parse_args()

    model, _ = build_model()
    path2meta = load_val_meta(data_dir=args.data_dir)

    imgs = iter_images(args.data_dir)
    print(f"\nProcessing {len(imgs)} images...\n")
    t0 = datetime.datetime.now()

    ensure_dir(args.out_root)
    for p in imgs:
        if args.no_core:
            x = tfm(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_idx = int(_predict_logits_tensor(model, x).argmax(1).item())
            img_dir = ensure_dir(os.path.join(args.out_root, os.path.splitext(os.path.basename(p))[0] + "_unknown", "images"))
            stem = os.path.splitext(os.path.basename(p))[0]
        else:
            info = run_core_outputs(model, p, path2meta, args.out_root)
            pred_idx, img_dir, stem = info["pred_idx"], info["img_dir"], info["stem"]

        out, _ = run_smoothgrad(
            model, p, target_idx=pred_idx,
            n_samples=args.smoothgrad_samples,
            sigma=args.smoothgrad_sigma,
            batch_size=args.batch_size,
            agg=args.agg,
            patch=args.patch,
            p_low=args.p_low, p_high=args.p_high, gamma_corr=args.gamma
        )
        save_to = os.path.join(img_dir, f"{stem}_smoothgrad.jpg")
        cv2.imwrite(save_to, out)
        print("Saved:", save_to)

    print("Done. Runtime:", datetime.datetime.now() - t0)

if __name__ == "__main__":
    main()
