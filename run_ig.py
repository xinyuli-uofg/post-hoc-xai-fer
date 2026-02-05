

import os
import datetime
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from captum.attr import IntegratedGradients, LayerIntegratedGradients, NoiseTunnel

from xai_core import (
    add_common_cli, build_model, load_val_meta, ensure_dir, iter_images,
    run_core_outputs, tfm, device, _predict_logits_tensor, resolve_module_by_name
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def _normalized_midgray_like(x: torch.Tensor, dev: torch.device) -> torch.Tensor:
    return ((0.5 - IMAGENET_MEAN.to(dev)) / IMAGENET_STD.to(dev)).expand_as(x)

def _robust_norm01(m2d: np.ndarray, lo=80.0, hi=99.5, gamma=0.9, eps=1e-8) -> np.ndarray:
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

def run_ig_pixel(
    model,
    img_path,
    target_idx=None,
    steps=50,
    use_noise=True,
    nt_samples=8,
    nt_type="smoothgrad_sq",
    p_low=80.0,
    p_high=99.5,
    gamma_corr=0.9,
    colormap=cv2.COLORMAP_VIRIDIS
):
    model.eval()
    dev = next(model.parameters()).device

    pil = Image.open(img_path).convert("RGB")
    x = tfm(pil).unsqueeze(0).to(dev)

    with torch.no_grad():
        logits = _predict_logits_tensor(model, x)
        pred_idx = int(logits.argmax(1).item())
    target_idx = pred_idx if target_idx is None else int(target_idx)

    def forward_fn(t):
        return _predict_logits_tensor(model, t)[:, target_idx]

    ig = IntegratedGradients(forward_fn)
    baseline = _normalized_midgray_like(x, dev)

    if use_noise:
        nt = NoiseTunnel(ig)
        attr = nt.attribute(
            inputs=x, baselines=baseline, n_steps=int(steps),
            nt_samples=int(nt_samples), nt_type=str(nt_type)
        )
    else:
        attr = ig.attribute(inputs=x, baselines=baseline, n_steps=int(steps))

    attr = torch.relu(attr.sum(dim=1, keepdim=True))  # [1,1,H,W]
    heat = attr[0, 0].detach().cpu().numpy()
    heat01 = _robust_norm01(heat, lo=p_low, hi=p_high, gamma=gamma_corr)

    raw = cv2.imread(img_path)
    heat_u8 = (cv2.resize(heat01, (raw.shape[1], raw.shape[0])) * 255).astype(np.uint8)
    cm = cv2.applyColorMap(heat_u8, colormap)
    out = cv2.addWeighted(raw, 0.55, cm, 0.45, 0)
    return out, target_idx

def run_ig_layer(
    model,
    img_path,
    target_idx=None,
    layer_name="pre_concept_model.patch_embed.proj",
    steps=50,
    use_noise=True,
    nt_samples=4,
    nt_type="smoothgrad_sq",
    attr_to_input=False,
    internal_batch_size=8,
    patch=16,
    p_low=80.0,
    p_high=99.5,
    gamma_corr=0.9,
    colormap=cv2.COLORMAP_VIRIDIS
):
    model.eval()
    dev = next(model.parameters()).device

    pil = Image.open(img_path).convert("RGB")
    x = tfm(pil).unsqueeze(0).to(dev)
    H_in, W_in = x.shape[-2], x.shape[-1]

    with torch.no_grad():
        logits = _predict_logits_tensor(model, x)
        pred_idx = int(logits.argmax(1).item())
    target_idx = pred_idx if target_idx is None else int(target_idx)

    layer = resolve_module_by_name(model, layer_name)

    def forward_fn(t):
        return _predict_logits_tensor(model, t)[:, target_idx]

    lig = LayerIntegratedGradients(forward_fn, layer)
    baseline = _normalized_midgray_like(x, dev)

    kwargs = dict(
        inputs=x,
        baselines=baseline,
        n_steps=int(steps),
        internal_batch_size=int(internal_batch_size),
        attribute_to_layer_input=bool(attr_to_input),
    )

    if use_noise:
        nt = NoiseTunnel(lig)
        attr = nt.attribute(nt_samples=int(nt_samples), nt_type=str(nt_type), **kwargs)
    else:
        attr = lig.attribute(**kwargs)

    if attr.dim() == 4:
        attr = attr.sum(dim=1, keepdim=True)
    else:
        attr = attr.unsqueeze(1)
    attr = torch.relu(attr)

    heat = attr[0, 0].detach().cpu().numpy()

    # Patch-aware resize (optional): map to patch grid then upsample once with nearest
    if patch is not None and patch > 0:
        Hp = H_in // patch
        Wp = W_in // patch
        if Hp > 0 and Wp > 0 and heat.shape != (Hp, Wp):
            heat = cv2.resize(heat, (Wp, Hp), interpolation=cv2.INTER_AREA)

    heat01 = _robust_norm01(heat, lo=p_low, hi=p_high, gamma=gamma_corr)

    raw = cv2.imread(img_path)
    heat_u8 = (cv2.resize(heat01, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST) * 255).astype(np.uint8)
    cm = cv2.applyColorMap(heat_u8, colormap)
    out = cv2.addWeighted(raw, 0.55, cm, 0.45, 0)
    return out, target_idx

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Integrated Gradients only (pixel IG or LayerIG).")
    add_common_cli(parser)

    parser.add_argument('--ig_steps', type=int, default=50)
    parser.add_argument('--ig_layer', type=str, default=None,
                        help="If set, run LayerIntegratedGradients at this layer; otherwise run pixel IG.")
    parser.add_argument('--ig_attr_to_input', action='store_true',
                        help="For LayerIG: attribute to layer input (often best for conv/patch-proj).")
    parser.add_argument('--ig_internal_bs', type=int, default=8)
    parser.add_argument('--ig_nt_samples', type=int, default=4)

    parser.add_argument('--patch', type=int, default=16, help="Patch pooling/upsample size for LayerIG; set 0 to disable.")
    parser.add_argument('--p_low', type=float, default=80.0)
    parser.add_argument('--p_high', type=float, default=99.5)
    parser.add_argument('--gamma', type=float, default=0.9)
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

        if args.ig_layer:
            out, _ = run_ig_layer(
                model, p, target_idx=pred_idx,
                layer_name=args.ig_layer,
                steps=args.ig_steps,
                use_noise=True,
                nt_samples=args.ig_nt_samples,
                attr_to_input=args.ig_attr_to_input,
                internal_batch_size=args.ig_internal_bs,
                patch=(None if args.patch == 0 else args.patch),
                p_low=args.p_low, p_high=args.p_high, gamma_corr=args.gamma
            )
            save_to = os.path.join(img_dir, f"{stem}_ig_layer.jpg")
        else:
            out, _ = run_ig_pixel(
                model, p, target_idx=pred_idx,
                steps=args.ig_steps,
                use_noise=True,
                nt_samples=max(8, args.ig_nt_samples),
                p_low=args.p_low, p_high=args.p_high, gamma_corr=args.gamma
            )
            save_to = os.path.join(img_dir, f"{stem}_ig.jpg")

        cv2.imwrite(save_to, out)
        print("Saved:", save_to)

    print("Done. Runtime:", datetime.datetime.now() - t0)

if __name__ == "__main__":
    main()