

import os
import datetime
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from xai_core import (
    add_common_cli, build_model, load_val_meta, ensure_dir, iter_images,
    run_core_outputs, tfm, device, _predict_logits_tensor, resolve_module_by_name
)

def run_layercam(model, img_path, target_idx=None, layer_name="pre_concept_model.patch_embed.proj"):
    model.eval()
    x = tfm(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

    layer = resolve_module_by_name(model, layer_name)

    feats = {}
    grads = {}

    def fwd_hook(m, inp, out):
        feats["act"] = out.detach()

    def bwd_hook(m, ginp, gout):
        grads["grad"] = gout[0].detach()

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    try:
        with torch.no_grad():
            logits = _predict_logits_tensor(model, x)
            pred_idx = int(logits.argmax(1).item())
        target_idx = pred_idx if target_idx is None else int(target_idx)

        model.zero_grad(set_to_none=True)
        x.requires_grad_(True)
        y = _predict_logits_tensor(model, x)[:, target_idx].sum()
        y.backward()

        A = feats["act"]
        G = grads["grad"]

        cam = (torch.relu(G) * torch.relu(A)).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam_up = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam_np = cam_up.detach().cpu().numpy()

        p99 = np.percentile(cam_np, 99)
        cam_np = np.clip(cam_np, 0, p99) / (p99 + 1e-8)

        raw = cv2.imread(img_path)
        heat_u8 = (cv2.resize(cam_np, (raw.shape[1], raw.shape[0])) * 255).astype(np.uint8)
        cm = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        out = cv2.addWeighted(raw, 0.6, cm, 0.4, 0)
        return out, target_idx

    finally:
        h1.remove()
        h2.remove()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run LayerCAM only.")
    add_common_cli(parser)
    parser.add_argument('--layercam_layer', type=str, default='pre_concept_model.patch_embed.proj',
                        help="Target module for LayerCAM (recommended for ViT/MAE patch embedding conv).")
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

        out, _ = run_layercam(model, p, target_idx=pred_idx, layer_name=args.layercam_layer)
        save_to = os.path.join(img_dir, f"{stem}_layercam.jpg")
        cv2.imwrite(save_to, out)
        print("Saved:", save_to)

    print("Done. Runtime:", datetime.datetime.now() - t0)

if __name__ == "__main__":
    main()
