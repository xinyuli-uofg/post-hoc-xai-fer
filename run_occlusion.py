

import os
import datetime
import numpy as np
import cv2
from PIL import Image

import torch
from captum.attr import Occlusion

from xai_core import (
    add_common_cli, build_model, load_val_meta, ensure_dir, iter_images,
    run_core_outputs, tfm, device, _predict_logits_tensor
)

def run_occlusion(model, img_path, target_idx=None, kernel_size=21, stride=10):
    model.eval()
    x = tfm(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = _predict_logits_tensor(model, x)
        pred_idx = int(logits.argmax(1).item())
    target_idx = pred_idx if target_idx is None else int(target_idx)

    occ = Occlusion(lambda t: _predict_logits_tensor(model, t)[:, target_idx])

    attributions = occ.attribute(
        x,
        strides=(1, int(stride), int(stride)),
        sliding_window_shapes=(3, int(kernel_size), int(kernel_size)),
        baselines=0
    )  # [1,3,H,W]

    heat = np.abs(attributions.mean(1, keepdim=True)[0, 0].detach().cpu().numpy())
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    raw = cv2.imread(img_path)
    heat_u8 = (cv2.resize(heat, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_CUBIC) * 255).astype(np.uint8)
    cm = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    out = cv2.addWeighted(raw, 0.6, cm, 0.4, 0)
    return out, target_idx

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Occlusion Sensitivity only.")
    add_common_cli(parser)
    parser.add_argument('--occ_kernel', type=int, default=21)
    parser.add_argument('--occ_stride', type=int, default=10)
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

        out, _ = run_occlusion(model, p, target_idx=pred_idx, kernel_size=args.occ_kernel, stride=args.occ_stride)
        save_to = os.path.join(img_dir, f"{stem}_occlusion.jpg")
        cv2.imwrite(save_to, out)
        print("Saved:", save_to)

    print("Done. Runtime:", datetime.datetime.now() - t0)

if __name__ == "__main__":
    main()
