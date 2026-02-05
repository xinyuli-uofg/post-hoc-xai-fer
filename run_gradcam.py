
import os
import datetime
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from captum.attr import LayerGradCam, LayerAttribution

from xai_core import (
    add_common_cli, build_model, load_val_meta, ensure_dir, iter_images,
    run_core_outputs, tfm, device, _predict_logits_tensor, resolve_module_by_name
)

def _autoselect_vit_conv_layer(model: nn.Module):
    candidates = [
        "pre_concept_model.patch_embed.proj",
        "pre_concept_model.patch_embed.conv",
    ]
    for name in candidates:
        try:
            mod = resolve_module_by_name(model, name)
            if isinstance(mod, nn.Conv2d):
                return mod
        except Exception:
            pass

    last_conv = None
    if hasattr(model, "pre_concept_model"):
        for m in model.pre_concept_model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
    return last_conv

def _gradcam_reduce_and_normalize(up_attr: torch.Tensor) -> np.ndarray:
    m = up_attr.squeeze(0).detach().cpu().numpy()
    heat = m.sum(axis=0) if m.ndim == 3 else m
    heat = np.maximum(heat, 0.0)
    p99 = np.percentile(heat, 99)
    if p99 > 1e-8:
        heat = np.clip(heat, 0, p99) / p99
    else:
        heat = np.zeros_like(heat)
    return heat

def run_gradcam(model, img_path, target_idx=None, layer_name=None):
    model.eval()
    x = tfm(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    x.requires_grad_(True)

    with torch.no_grad():
        logits = _predict_logits_tensor(model, x)
        pred_idx = int(logits.argmax(1).item())
    target_idx = pred_idx if target_idx is None else int(target_idx)

    if layer_name:
        layer = resolve_module_by_name(model, layer_name)
    else:
        layer = _autoselect_vit_conv_layer(model)
        if layer is None:
            raise RuntimeError(
                "Grad-CAM requires a Conv2d layer. Provide --gradcam_layer for ViT/MAE (e.g., pre_concept_model.patch_embed.proj)."
            )

    forward_fn = lambda t: _predict_logits_tensor(model, t)
    gradcam = LayerGradCam(forward_fn, layer)

    attr = gradcam.attribute(x, target=target_idx)
    up_attr = LayerAttribution.interpolate(attr, x.shape[2:])

    heat = _gradcam_reduce_and_normalize(up_attr)

    raw = cv2.imread(img_path)
    heat_u8 = (cv2.resize(heat, (raw.shape[1], raw.shape[0])) * 255).astype(np.uint8)
    cm = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    out = cv2.addWeighted(raw, 0.6, cm, 0.4, 0)
    return out, target_idx

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Grad-CAM only.")
    add_common_cli(parser)
    parser.add_argument('--gradcam_layer', type=str, default=None,
                        help="Dotted module path for layer (recommended for ViT/MAE: pre_concept_model.patch_embed.proj).")
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

        out, _ = run_gradcam(model, p, target_idx=pred_idx, layer_name=args.gradcam_layer)
        save_to = os.path.join(img_dir, f"{stem}_gradcam.jpg")
        cv2.imwrite(save_to, out)
        print("Saved:", save_to)

    print("Done. Runtime:", datetime.datetime.now() - t0)

if __name__ == "__main__":
    main()
